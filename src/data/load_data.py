import os
from torch.utils.data import DataLoader
from CONFIG import CONFIG


def load_data(exp_params, split="train"):
    params = dict(exp_params["dataset"])
    name = params.pop("dataset_name")
    root = params.pop("root")
    frames = exp_params["training_prediction"]["sample_length"]
    for key in ("shuffle_train", "shuffle_eval", "variant", "target", "vocab_size", "use_segmentation", "num_frames"):
        params.pop(key, None)
    if name == "CLIPort":
        from data.cliport import CLIPort
        return CLIPort(datapath=root, split=split, num_frames=frames, **params)
    if name == "LanguageTable-Synthetic":
        from data.language_table import LanguageTableSynthetic
        return LanguageTableSynthetic(datapath=root, split=split, num_frames=frames, **params)
    if name == "LanguageTable-Real":
        from data.language_table import LanguageTableReal
        return LanguageTableReal(root_dir=root, split=split, num_frames=frames, **params)
    if name == "BridgeV2":
        from data.bridge import BridgeV2, BridgeV2_precomputed_slots, BridgeV2_precomputed_latents
        if params.get("precomputed_latents_path"):
            cls = BridgeV2_precomputed_latents
        elif params.get("precomputed_slots_path"):
            cls = BridgeV2_precomputed_slots
        else:
            cls = BridgeV2
            for key in ("precomputed_slots_path", "precomputed_latents_path", "t5_embeddings_path", "load_imgs"):
                params.pop(key, None)
        return cls(datapath=root, split=split, num_frames=frames, **params)
    raise ValueError(f"Unknown dataset: {name}")


def build_data_loader(dataset, batch_size, shuffle=False, sampler=None, drop_last=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                      drop_last=drop_last, num_workers=CONFIG["num_workers"],
                      collate_fn=getattr(dataset, "collate_fn", None), pin_memory=True)


def unwrap_batch_data(exp_params, batch_data):
    if exp_params["dataset"]["dataset_name"] == "LanguageTable-Synthetic":
        videos, targets, sample = batch_data
    else:
        videos, sample = batch_data
        targets = videos
    others = dict(sample)
    others["captions"] = sample["caption"]
    others["slots"] = sample.get("slots", sample.get("latents"))
    if "imgs" in sample and sample["imgs"] is not None:
        videos = sample["imgs"]
        targets = videos
    return videos, targets, {}, others
