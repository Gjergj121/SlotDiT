import os
import json
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import re

from lib.logger import print_

from transformers import T5Tokenizer

class LanguageTableSynthetic(Dataset):
    def __init__(self, split, datapath, num_frames=20, sample_rate=1,
                 random_start=True, img_size=(128, 128), use_string_captions=False,
                 eval_min_num_frames=None, **kwargs):
        assert split in ["train", "val", "valid", "eval", "test"], f"Unknown {split = }..."
        split = "val" if split in ["val", "valid", "test", "eval"] else split
        assert split in ['train', 'val'], f"Unknown dataset split {split}..."
        assert os.path.exists(datapath), f"{datapath = } does not exist..."

        self.split = split
        self.root = datapath
        self.datapath = os.path.join(datapath, split)
        self.num_frames = num_frames
        self.random_start = random_start
        self.img_size = img_size
        self.sample_rate = sample_rate
        self.random_start = random_start if split == "train" else False
        self.to_tensor = transforms.ToTensor()
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.use_string_captions = use_string_captions
        self.latents_path = kwargs.get("latents_path", None)

        self.eval_min_num_frames = eval_min_num_frames if split != "train" else None

        if self.latents_path is not None:
            print_(f"Using precomputed latents from path: {self.latents_path}")

            split_path = "valid" if split == "val" or split == "test" else "train"
            self.latents_dir = os.path.join(self.latents_path, split_path)
            self.latents_files = [f for f in os.listdir(self.latents_dir) if f.endswith('.pt')]

            self.latents_files.sort(key=lambda x: int(re.search(r'video(\d+)\.pt', x).group(1)))

        self.episodes = []
        for ep in sorted(os.listdir(self.datapath)):
            num_imgs = len(os.listdir(os.path.join(self.datapath, ep)))
            if self.num_frames is None:
                if num_imgs < 9:
                    continue
            elif num_imgs < self.num_frames * self.sample_rate + 1:
                continue
            if self.eval_min_num_frames is not None and num_imgs < self.eval_min_num_frames:
                continue
            self.episodes.append(ep)
        self.num_episodes = len(self.episodes)
        self.print_db()
        return

    def __len__(self):
        return self.num_episodes

    def decode_inst(self, inst):
        return bytes(inst[np.where(inst != 0)].tolist()).decode("utf-8")

    def __getitem__(self, index):
        episode = self.episodes[index]
        img_path = os.path.join(self.datapath, episode)

        img_names = self.get_imgs_names_to_load(img_path)
        num_imgs = len(img_names)

        if self.num_frames is not None and num_imgs < self.num_frames * self.sample_rate + 1:
            raise ValueError(f"Episode {episode} with {num_imgs = } is not valid...")
            return self.__getitem__(np.random.randint(0, self.num_episodes))

        if self.num_frames is not None:
            if self.random_start:
                start_idx = np.random.randint(0, num_imgs - self.num_frames * self.sample_rate)
            else:
                start_idx = 0
            end_idx = start_idx + self.num_frames * self.sample_rate
        else:
            start_idx = 0
            end_idx = num_imgs

        img_names = img_names[start_idx:end_idx:self.sample_rate]
        imgs = self.load_imgs(img_path, img_names)

        if self.latents_path is not None:
            latents_path = os.path.join(self.latents_dir, self.latents_files[index])
            latents = torch.load(latents_path, weights_only=True)
            latents = latents[start_idx:end_idx:self.sample_rate]
        else:
            latents = None

        actions_path = os.path.join(self.root, "actions")
        if os.path.exists(actions_path):
            actions = torch.from_numpy(
                    np.load(os.path.join(actions_path, f"{episode}.npy"))
                )[start_idx:end_idx:self.sample_rate]
        else:
            actions = []

        captions_path = os.path.join(self.root, "labels")
        if os.path.exists(captions_path):
            if self.use_string_captions:
                captions = np.load(
                    os.path.join(captions_path, f"{episode}.npy"),
                    allow_pickle=True
                )
                text_caption = str(captions)

                captions = torch.from_numpy(
                    np.frombuffer(text_caption.encode('utf-8'), dtype=np.uint8)
                )
            else:
                captions = torch.from_numpy(
                    np.load(os.path.join(captions_path, f"{episode}.npy"))
                )
                text_caption = self.decode_inst(captions)
        else:
            captions = []
            text_caption = ""

        targets = imgs
        all_reps = {
            "videos": imgs,
            "episode": episode,
            "text_tokens": captions,
            "caption": text_caption,
            "img_paths": img_names,
            "latents": latents,
            "actions": actions
        }
        return imgs, targets, all_reps

    def get_imgs_names_to_load(self, img_path):
        img_names = sorted(
                [f for f in os.listdir(img_path) if f.endswith(".png")],
                key=lambda f: int(f.split(".")[0].split("_")[-1])
            )

        return img_names

    def print_db(self):
        print_("Instantiating LanguageTable-Synthetic dataset:")
        print_(f"  --> datapath: {self.datapath}")
        print_(f"  --> split: {self.split}")
        print_(f"  --> NumEpisodes: {self.num_episodes}")
        print_(f"  --> NumFrames: {self.num_frames}")
        print_(f"  --> Sample Rate: {self.sample_rate}")
        print_(f"  --> Random Start: {self.random_start}")
        print_(f"  --> Img Size: {self.img_size}")
        print_(f"  --> Eval Min Num Frames: {self.eval_min_num_frames}")
        return

    def load_imgs(self, episode, img_names):
        imgs = []
        for img_name in img_names:
            cur_p = os.path.join(episode, img_name)
            img = Image.open(cur_p)
            img = img.resize(self.img_size)
            img = self.to_tensor(img)[:3]
            imgs.append(img)
        imgs = torch.stack(imgs, dim=0).float()
        return imgs

    def get_num_frames_per_episode(self):
        num_imgs = []
        for i in tqdm(range(len(self))):
            episode = self.episodes[i]
            ep_path = os.path.join(self.datapath, episode)
            img_names = self.get_imgs_names_to_load(ep_path)
            num_imgs.append(len(img_names))
        return num_imgs

    def collate_fn(self, data):
        caption_tokens = self.tokenizer([d[2]['caption'] for d in data], padding=True, return_tensors="pt")

        d0 = torch.stack([d[0] for d in data], dim=0)
        d1 = torch.stack([d[1] for d in data], dim=0)

        text_tokens_list = [torch.as_tensor(d[2]['text_tokens']) for d in data]
        if all(len(t) == len(text_tokens_list[0]) for t in text_tokens_list):
            text_tokens = torch.stack(text_tokens_list, dim=0)
        else:
            max_len = max(len(t) for t in text_tokens_list)
            text_tokens = torch.stack([
                torch.nn.functional.pad(t, (0, max_len - len(t)), value=0)
                for t in text_tokens_list
            ], dim=0)

        d2 = {
            "videos": d0,
            "episode": [d[2]['episode'] for d in data],
            "text_tokens": text_tokens,
            "caption": [d[2]['caption'] for d in data],
            "caption_tokens": caption_tokens,
            "img_paths": [d[2]['img_paths'] for d in data],
            "actions": torch.stack([torch.as_tensor(d[2]["actions"]) for d in data]),
            "latents": torch.stack([d[2]['latents'] for d in data], dim=0) if self.latents_path is not None else None
        }

        return d0, d1, d2


import random
from glob import glob


class LanguageTableReal(Dataset):
    def __init__(
        self,
        root_dir="datasets/lt_real",
        split="train",
        num_frames=None,
        min_num_frames=None,
        split_ratio=0.8,
        step_size=1,
        img_size=None,
        resize=True,
        random_start=False,
        max_episodes=None,
        seed=42,
        verbose=True,
        partition_num_frames=None,
        partition_step_size=None,
        eval_min_num_frames=None,
        eval_max_episodes=None,
    ):
        self.root_dir = root_dir
        self.split = "val" if split in ["val", "valid", "eval", "test"] else split
        self.split_ratio = split_ratio
        self.num_frames = num_frames
        self.min_num_frames = min_num_frames
        self.step_size = step_size
        self.img_size = img_size
        self.random_start = random_start if self.split == "train" else False
        self.max_episodes = max_episodes
        self.seed = seed
        self.verbose = verbose
        self.partition_num_frames = partition_num_frames
        self.partition_step_size = partition_step_size
        self.eval_min_num_frames = eval_min_num_frames
        self.eval_max_episodes = eval_max_episodes
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.resize = resize
        if self.resize:
            self.resizer = transforms.Resize(
                    self.img_size,
                    interpolation=transforms.InterpolationMode.BILINEAR
                )

        self.episodes = self.fetch_episodes(seed=seed)

        print_(f"LanguageTable-Real dataset initialized with {len(self.episodes)} {self.split} episodes.")

    @staticmethod
    def _min_required_frames(min_num_frames, num_frames, step_size):
        out = None
        if min_num_frames is not None and min_num_frames > 0:
            out = min_num_frames
        if num_frames is not None:
            need = num_frames * step_size
            out = need if out is None else max(out, need)
        return out

    def fetch_episodes(self, seed=42):
        metadata_path = os.path.join(self.root_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"metadata.json not found in {self.root_dir}")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        all_episode_fnames = sorted(metadata.keys())

        part_nf = self.partition_num_frames if self.partition_num_frames is not None else self.num_frames
        part_ss = self.partition_step_size if self.partition_step_size is not None else self.step_size
        partition_min = self._min_required_frames(self.min_num_frames, part_nf, part_ss)

        if self.partition_num_frames is None and self.num_frames is not None and self.verbose:
            print_("WARNING [LanguageTable-Real]: `partition_num_frames` not set — the "
                   "train/val partition will depend on the requested num_frames "
                   f"(={self.num_frames}). Pin `partition_num_frames` (and `partition_step_size`) "
                   "to the values used at training time for a stable, leak-free split.")

        if partition_min is None:
            partition_fnames = list(all_episode_fnames)
        else:
            partition_fnames = [
                fname for fname in all_episode_fnames
                if metadata[fname]["num_frames"] >= partition_min
            ]

        if self.max_episodes is not None:
            partition_fnames = partition_fnames[:self.max_episodes]

        torch.manual_seed(seed)
        perm = torch.randperm(len(partition_fnames)).tolist()
        cutoff = int(self.split_ratio * len(partition_fnames))
        if self.split == "train":
            split_fnames = [partition_fnames[i] for i in perm[:cutoff]]
        else:
            split_fnames = [partition_fnames[i] for i in perm[cutoff:]]

        requested_floor = self.min_num_frames
        if self.split != "train" and self.eval_min_num_frames is not None:
            requested_floor = self.eval_min_num_frames if requested_floor is None \
                else max(requested_floor, self.eval_min_num_frames)
        requested_min = self._min_required_frames(requested_floor, self.num_frames, self.step_size)

        if requested_min is None:
            valid_fnames = split_fnames
        else:
            valid_fnames = [
                fname for fname in split_fnames
                if metadata[fname]["num_frames"] >= requested_min
            ]

        if self.split != "train" and self.eval_max_episodes is not None:
            valid_fnames = valid_fnames[:self.eval_max_episodes]

        if self.verbose:
            extra = "" if partition_min is None else f" (partition floor {partition_min})"
            if requested_min is None:
                print(f"{self.split} episodes: {len(valid_fnames)} / {len(split_fnames)} in partition{extra}")
            else:
                print(f"{self.split} episodes with >= {requested_min} frames: "
                      f"{len(valid_fnames)} / {len(split_fnames)} in partition{extra}")

        episode_paths = [os.path.join(self.root_dir, fname) for fname in valid_fnames]

        if self.verbose:
            print(f"Using {len(episode_paths)} {self.split} episodes")

        return episode_paths

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        episode_path = self.episodes[idx]
        data = torch.load(episode_path)

        images = data["images"]
        instruction = data["instruction"]

        current_num_frames = images.shape[0]
        if self.num_frames is not None:
            if current_num_frames < (self.num_frames * self.step_size) :
                raise ValueError(f"{self.num_frames * self.step_size = } are required, " +
                                f"but only {current_num_frames = } are available for episode {idx}...")

            if self.random_start:
                max_start = max(current_num_frames - (self.num_frames * self.step_size), 0)
                start_frame_idx = random.randint(0, max_start)
            else:
                start_frame_idx = 0

            frame_indices = range(
                    start_frame_idx,
                    start_frame_idx + self.num_frames * self.step_size,
                    self.step_size
                )
        else:
            frame_indices = slice(None)

        images = images[frame_indices] / 255
        if self.resize:
            images = self.resizer(images)

        sample = {
            "imgs": images,
            "caption": instruction
        }

        return images, sample

    def collate_fn(self, data):
        caption_tokens = self.tokenizer([d[1]['caption'] for d in data], padding=True, return_tensors="pt")

        d0 = torch.stack([d[0] for d in data], dim=0)
        d1 = {
            "imgs": d0,
            "caption": [d[1]['caption'] for d in data],
            "caption_tokens": caption_tokens
        }

        return d0, d1
