import argparse
import json
import os
from pathlib import Path
import torch
from tqdm import tqdm
from lib.config import Config
from lib import setup_model, utils
from lib.representations import Representation
from lib.runtime import text_and_diffusion, embed_text
from data.load_data import load_data, build_data_loader, unwrap_batch_data


@torch.no_grad()
def predict_video(predictor, representation, text_embedder, diffusion, videos, others, params):
    context = params["num_context"]
    horizon = params["num_preds"]
    x = representation.encode(videos[:, :context])
    y, mask = embed_text(text_embedder, others, videos.device)
    context_tokens = x.shape[1]
    future_tokens = (representation.frames_to_tokens(horizon) if representation.kind == "videovae"
                     else horizon)
    history = min(representation.frames_to_tokens(params["history_context"]), predictor.buffer_size - 1)
    latents = predictor(x, num_context=context_tokens, num_preds=future_tokens, history_context=history,
                        token_embeddings=y, mask=mask, diffusion=diffusion, training=False,
                        cfg_scale=params["cfg_scale"], using_cfg=params["cfg_scale"] > 1.0)
    if representation.kind == "videovae":
        factor = representation.temporal_factor
        causal = representation.model.is_causal
        length = (context_tokens + future_tokens) * factor - (factor - 1 if causal else 0)
        full = representation.decode(torch.cat([x, latents], dim=1), num_frames=length)
        offset = 0 if causal else context_tokens * factor - context
        return full[:, offset + context:offset + context + horizon]
    return representation.decode(latents)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--exp-directory", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-preds", type=int, default=29)
    parser.add_argument("--history-context", type=int, default=4)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--metrics", nargs="+", choices=["mse", "psnr", "ssim", "lpips", "fvd"], default=["lpips"])
    parser.add_argument("--i3d-checkpoint")
    parser.add_argument("--vlm-model")
    parser.add_argument("--vlm-url", default="http://localhost:8000/v1")
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()
    if args.num_preds < 1 or args.history_context < 1:
        parser.error("Prediction and history lengths must be positive")
    if "fvd" in args.metrics and not args.i3d_checkpoint:
        parser.error("FVD requires --i3d-checkpoint")
    utils.set_random_seed(14)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = Config(args.exp_directory).load_exp_config_file()
    params["evaluation"].update(num_preds=args.num_preds, history_context=args.history_context, cfg_scale=args.cfg_scale)
    params["training_prediction"]["sample_length"] = args.num_preds + 1
    for key in ("slots_path", "latents_path", "precomputed_slots_path", "precomputed_latents_path", "t5_embeddings_path"):
        params["dataset"].pop(key, None)
    dataset = load_data(params, split="valid")
    if len(dataset) == 0:
        raise ValueError("The evaluation dataset is empty")
    loader = build_data_loader(dataset, args.batch_size)
    representation = Representation(params, device)
    predictor = setup_model.setup_predictor(params).to(device).eval()
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    name = params["model"]["predictor"]["predictor_name"]
    key = "ema" if params["model"]["predictor"][name].get("load_ema", True) else "model_state_dict"
    predictor.load_state_dict(state[key])
    text, diffusion = text_and_diffusion(params, device)
    from lib.metrics import MSE, PSNR, SSIM, LPIPS, FVD
    classes = {"mse": MSE, "psnr": PSNR, "ssim": SSIM, "lpips": LPIPS, "fvd": FVD}
    metrics = {name: classes[name](args.i3d_checkpoint) if name == "fvd" else classes[name]() for name in args.metrics}
    judgments = []
    if args.vlm_model:
        from openai import OpenAI
        from lib.vlm import judge_video
        client = OpenAI(base_url=args.vlm_url, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    with torch.no_grad():
        for batch in tqdm(loader):
            videos, targets, _, others = unwrap_batch_data(params, batch)
            videos, targets = videos.to(device), targets.to(device)
            predictions = predict_video(predictor, representation, text, diffusion, videos, others, params["evaluation"])
            targets = targets[:, 1:args.num_preds + 1].contiguous()
            if representation.kind == "rae" and predictions.shape[-2:] != targets.shape[-2:]:
                shape = targets.shape
                targets = torch.nn.functional.interpolate(targets.flatten(0, 1), size=predictions.shape[-2:], mode="bilinear", align_corners=False)
                targets = targets.reshape(shape[0], shape[1], shape[2], *predictions.shape[-2:])
            if predictions.shape != targets.shape:
                raise ValueError(f"Prediction/target shape mismatch: {predictions.shape}, {targets.shape}")
            for metric in metrics.values():
                metric.accumulate(preds=predictions.contiguous(), targets=targets)
            if args.vlm_model:
                for i, caption in enumerate(others["captions"]):
                    frames = torch.cat([videos[i, :1], predictions[i]], dim=0)
                    judgments.append({"instruction": caption, **judge_video(client, args.vlm_model, frames, caption)})
    results = {name: {"mean": float(value[0]), "per_frame": value[1].tolist()} for name, metric in metrics.items() for value in [metric.aggregate()]}
    if judgments:
        results["task_success"] = {"rate": sum(j["success"] for j in judgments) / len(judgments), "episodes": judgments}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
