import argparse
import os
import torch
import torch.distributed as dist
from data.t5 import T5TextEmbedder
from diffusion import create_diffusion_dfot
from models.model_utils import freeze_params
from lib import utils
from CONFIG import CONFIG


def training_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--exp-directory", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--resume-training", action="store_true")
    args = parser.parse_args()
    if args.resume_training and not args.checkpoint:
        parser.error("--resume-training requires --checkpoint")
    return args


def distributed_setup():
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    utils.set_random_seed(CONFIG["random_seed"] + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return rank, world_size, device


def text_and_diffusion(params, device):
    name = params["model"]["predictor"]["predictor_name"]
    model = params["model"]["predictor"][name]
    text_params = model["text_encoder_params"]
    text = T5TextEmbedder(device=device, model_name=text_params["model_name"],
                          model_max_length=text_params["token_max_length"],
                          use_text_preprocessing=text_params["use_text_preprocessing"]).to(device)
    text = freeze_params(text.eval())
    diffusion = create_diffusion_dfot(**model["diffusion_params"], learn_sigma=model["learn_sigma"])
    diffusion.sampling_timesteps = params["evaluation"]["num_sampling_steps"]
    return text, diffusion


def embed_text(text_embedder, others, device):
    if "t5_embedding" in others:
        return others["t5_embedding"].to(device), others["t5_attention_mask"].to(device)
    return text_embedder.embed_texts(others["captions"])
