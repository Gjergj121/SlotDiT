import os
import csv
import json
import random
import re
import unicodedata
from collections import Counter
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from lib.logger import print_

try:
    from langdetect import detect as _lang_detect
    from langdetect import DetectorFactory as _LangDetectorFactory
    from langdetect.lang_detect_exception import LangDetectException as _LangDetectException
    _LangDetectorFactory.seed = 0
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False

_CAPTION_ARTIFACT_PREFIXES = (
    "alook at the sequence",
    "look at the sequence of images",
)

_LANGDETECT_MIN_CHARS = 15

_DRAWER_OPEN_TOKENS = {
    "open", "opened", "opening", "opens", "opeen", "oppen", "opne", "opend",
}
_DRAWER_CLOSE_TOKENS = {
    "close", "closed", "closing", "closes", "clse", "cloze", "clóse", "clos",
}
_DRAWER_NOUN_TOKENS = {
    "drawer", "drawers", "draw", "drower", "drawr",
}

_DRAWER_TEMPLATE_OPEN = "open the drawer"
_DRAWER_TEMPLATE_CLOSE = "close the drawer"

_CANONICALIZE_TYPO_FIXES = (
    (re.compile(r"\bdrowers\b"), "drawers"),
    (re.compile(r"\bdrower\b"), "drawer"),
    (re.compile(r"\bdrawr\b"), "drawer"),
    (re.compile(r"\bopeen\b"), "open"),
    (re.compile(r"\boppen\b"), "open"),
    (re.compile(r"\bopne\b"), "open"),
    (re.compile(r"\bopend\b"), "opened"),
    (re.compile(r"\bclse\b"), "close"),
    (re.compile(r"\bcloze\b"), "close"),
    (re.compile(r"\bclos\b"), "close"),
)

class BridgeV2(Dataset):
    IMG_SIZES = [(224, 224), (256, 256), (128, 128)]

    def __init__(
        self,
        datapath,
        split,
        num_frames=None,
        step_size=1,
        img_size=None,
        random_start=False,
        text_tokenizer=None,
        include_actions=False,
        clean_captions=True,
        clean_captions_log_path=None,
        canonicalize_captions=False,
        canonicalize_mode="normalize",
        llm_canonicalization_path=None,
        drop_unknown_action=True,
        unknown_action_label="unknown action",
        eval_min_num_frames=None,
        **kwargs,
    ):
        assert split in ["train", "val", "valid", "test", "eval"], f"Unknown split {split}"
        split = "val" if split in ["val", "valid", "eval", "test"] else split

        self.root = os.path.join(datapath, split)
        assert os.path.isdir(self.root), f"Data directory not found: {self.root}"

        self.split = split
        self.num_frames = num_frames
        self.step_size = step_size
        self.random_start = random_start if split == "train" else False
        self.include_actions = include_actions

        self.eval_min_num_frames = eval_min_num_frames if self.split != "train" else None

        if img_size is None:
            self.img_size = self.IMG_SIZES[0]
        elif isinstance(img_size, int):
            self.img_size = (img_size, img_size)
        else:
            self.img_size = tuple(img_size)

        self.tokenizer = None
        if text_tokenizer is not None:
            from transformers import T5Tokenizer
            self.tokenizer = T5Tokenizer.from_pretrained(text_tokenizer)

        self.transform = transforms.Compose([
            transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

        self.episodes = sorted([
            d for d in os.listdir(self.root)
            if d.startswith("episode") and os.path.isdir(os.path.join(self.root, d))
        ])
        self.num_episodes = len(self.episodes)

        print_(f"Loading BridgeV2 {self.split} labels...")
        with ThreadPoolExecutor() as executor:
            self.labels = list(tqdm(
                executor.map(self._load_label, self.episodes),
                total=self.num_episodes,
            ))

        if clean_captions:
            log_path = clean_captions_log_path
            if log_path is None:
                log_path = os.path.join(self.root, f"caption_filter_log_{self.split}.csv")
            self._filter_noisy_captions(log_path)

        self.canonicalize_captions = canonicalize_captions
        self.canonicalize_mode = canonicalize_mode

        self.llm_canonicalization_path = llm_canonicalization_path

        self.raw_labels = list(self.labels)
        if canonicalize_captions:
            self._canonicalize_all_captions(mode=canonicalize_mode, datapath=datapath)

        self.drop_unknown_action = drop_unknown_action
        self.unknown_action_label = unknown_action_label
        if drop_unknown_action:
            self._drop_unknown_actions(unknown_action_label)

        if self.num_frames is not None:
            self._filter_short_episodes()

        if self.eval_min_num_frames is not None:
            self._filter_eval_short_episodes()

        print_(f"Loaded BridgeV2 {self.split} set:")
        print_(f"  --> datapath     : {datapath}")
        print_(f"  --> split        : {self.split}")
        print_(f"  --> num_episodes : {self.num_episodes}")
        print_(f"  --> num_frames   : {self.num_frames}")
        print_(f"  --> step_size    : {self.step_size}")
        print_(f"  --> img_size     : {self.img_size}")
        print_(f"  --> random_start : {self.random_start}")
        print_(f"  --> eval_min_num_frames : {self.eval_min_num_frames}")

    def __len__(self):
        return self.num_episodes

    def __getitem__(self, idx):
        episode_dir = self.episodes[idx]
        color_dir = os.path.join(self.root, episode_dir, "color")
        assert os.path.isdir(color_dir), f"Color dir missing for {episode_dir}"

        frame_files = sorted(os.listdir(color_dir))
        total_frames = len(frame_files)

        if self.num_frames is not None:
            required = self.num_frames * self.step_size
            if total_frames < required:
                raise ValueError(
                    f"Episode {episode_dir} has {total_frames} frames but "
                    f"{required} are needed (num_frames={self.num_frames}, step_size={self.step_size})"
                )
            if self.random_start:
                max_start = total_frames - required
                start = random.randint(0, max_start)
            else:
                start = 0
            frame_indices = range(start, start + required, self.step_size)
        else:
            start = 0
            frame_indices = range(0, total_frames, self.step_size)

        imgs = []
        for fi in frame_indices:
            img_path = os.path.join(color_dir, frame_files[fi])
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
            imgs.append(img)
        imgs = torch.stack(imgs)

        caption = self.labels[idx]

        sample = {
            "imgs": imgs,
            "caption": caption,
            "episode": episode_dir,
            "start_frame_idx": start,
        }
        return sample

    def collate_fn(self, data):
        caption_tokens = None
        if self.tokenizer is not None:
            caption_tokens = self.tokenizer(
                [d["caption"] for d in data],
                padding=True,
                return_tensors="pt",
            )

        imgs = torch.stack([d["imgs"] for d in data], dim=0)

        batch = {
            "imgs": imgs,
            "caption": [d["caption"] for d in data],
            "caption_tokens": caption_tokens,
        }
        return imgs, batch

    @staticmethod
    def _caption_drop_reason(caption):
        c = caption.strip().strip('"').strip("'").strip()
        if len(c) < 3:
            return "too_short"
        if not any(ch.isalpha() for ch in c):
            return "no_letters"
        ascii_ratio = sum(1 for ch in c if ord(ch) < 128) / len(c)
        if ascii_ratio < 0.8:
            return "non_ascii"
        c_lower = c.lower()
        if c_lower.startswith(_CAPTION_ARTIFACT_PREFIXES):
            return "artifact"
        if _LANGDETECT_AVAILABLE and len(c) >= _LANGDETECT_MIN_CHARS:
            try:
                if _lang_detect(c) != "en":
                    return "non_english"
            except _LangDetectException:
                return "langdetect_failed"
        return None

    @staticmethod
    def _normalize_caption_for_matching(caption):
        text = caption.strip().lower()
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @classmethod
    def _canonicalize_caption(cls, caption):
        text = caption.strip().strip('"').strip("'").strip()
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[\.\!\?\,;:]+$", "", text).strip()
        for pattern, replacement in _CANONICALIZE_TYPO_FIXES:
            text = pattern.sub(replacement, text)
        return text

    @classmethod
    def _canonicalize_caption_to_template(cls, caption):
        norm = cls._canonicalize_caption(caption)
        if not norm:
            return norm

        match_text = cls._normalize_caption_for_matching(norm)
        tokens = match_text.split()
        token_set = set(tokens)
        has_drawer = any(tok in _DRAWER_NOUN_TOKENS for tok in tokens)
        if not has_drawer:
            return norm

        has_open = any(tok in _DRAWER_OPEN_TOKENS for tok in tokens)
        has_close = any(tok in _DRAWER_CLOSE_TOKENS for tok in tokens)
        if not has_open and "pull" in token_set and ("out" in token_set or "open" in token_set):
            has_open = True
        if not has_close and "push" in token_set and (
            "in" in token_set or "back" in token_set or "close" in token_set
        ):
            has_close = True

        if has_open and not has_close:
            return _DRAWER_TEMPLATE_OPEN
        if has_close and not has_open:
            return _DRAWER_TEMPLATE_CLOSE
        return norm

    def _canonicalize_all_captions(self, mode, datapath=None):
        assert mode in ("normalize", "template", "llm"), \
            f"Unknown canonicalize_mode: {mode}"
        before_unique = len(set(self.labels))
        if mode == "template":
            self.labels = [self._canonicalize_caption_to_template(c) for c in self.labels]
        elif mode == "llm":
            mapping_path = self.llm_canonicalization_path
            if mapping_path is None:
                assert datapath is not None, \
                    "datapath required to default-resolve llm canonicalization path"
                mapping_path = os.path.join(datapath, "caption_canonicalization.json")
            assert os.path.isfile(mapping_path), (
                f"LLM canonicalization cache not found at {mapping_path}. "
                "Set llm_canonicalization_path to the JSON produced during "
                "BridgeV2 caption canonicalization."
            )
            with open(mapping_path) as f:
                mapping = json.load(f)
            print_(f"  --> loaded LLM caption mapping ({len(mapping)} entries) from {mapping_path}")
            missing = 0
            new_labels = []
            for raw in self.labels:
                if raw in mapping:
                    new_labels.append(mapping[raw])
                else:
                    missing += 1
                    new_labels.append(self._canonicalize_caption(raw))
            self.labels = new_labels
            if missing:
                print_(f"  --> {missing}/{len(self.labels)} captions missing from LLM cache "
                       f"(fell back to light normalization)")
        else:
            self.labels = [self._canonicalize_caption(c) for c in self.labels]
        after_unique = len(set(self.labels))
        print_(f"BridgeV2 caption canonicalization ({self.split}): "
               f"mode={mode}, unique {before_unique} -> {after_unique}")

    def _drop_unknown_actions(self, unknown_action_label):
        before = len(self.episodes)

        kept_eps, kept_labels, kept_raw = [], [], []
        dropped_unknown = 0
        for ep, label, raw_label in zip(self.episodes, self.labels, self.raw_labels):
            if label == unknown_action_label:
                dropped_unknown += 1
                continue
            kept_eps.append(ep)
            kept_labels.append(label)
            kept_raw.append(raw_label)

        self.episodes = kept_eps
        self.labels = kept_labels
        self.raw_labels = kept_raw
        self.num_episodes = len(self.episodes)

        print_(f"BridgeV2 unknown-action filter ({self.split}): "
               f"kept {self.num_episodes}/{before}, "
               f"dropped {dropped_unknown} ({unknown_action_label!r})")

    def _filter_short_episodes(self):
        required = self.num_frames * self.step_size
        before = len(self.episodes)

        def _frame_count(episode_dir):
            color_dir = os.path.join(self.root, episode_dir, "color")
            if not os.path.isdir(color_dir):
                return 0
            return len(os.listdir(color_dir))

        with ThreadPoolExecutor() as executor:
            counts = list(executor.map(_frame_count, self.episodes))

        kept = [(ep, lab, raw) for ep, lab, raw, n in zip(
            self.episodes, self.labels, self.raw_labels, counts) if n >= required]
        if kept:
            self.episodes, self.labels, self.raw_labels = map(list, zip(*kept))
        else:
            self.episodes, self.labels, self.raw_labels = [], [], []
        self.num_episodes = len(self.episodes)

        dropped = before - self.num_episodes
        if dropped:
            print_(f"BridgeV2 short-episode filter ({self.split}): "
                   f"kept {self.num_episodes}/{before}, "
                   f"dropped {dropped} (< {required} frames needed for "
                   f"num_frames={self.num_frames}, step_size={self.step_size})")

    def _filter_eval_short_episodes(self):
        required = self.eval_min_num_frames
        before = len(self.episodes)

        def _frame_count(episode_dir):
            color_dir = os.path.join(self.root, episode_dir, "color")
            if not os.path.isdir(color_dir):
                return 0
            return len(os.listdir(color_dir))

        with ThreadPoolExecutor() as executor:
            counts = list(executor.map(_frame_count, self.episodes))

        kept = [(ep, lab, raw) for ep, lab, raw, n in zip(
            self.episodes, self.labels, self.raw_labels, counts) if n >= required]
        if kept:
            self.episodes, self.labels, self.raw_labels = map(list, zip(*kept))
        else:
            self.episodes, self.labels, self.raw_labels = [], [], []
        self.num_episodes = len(self.episodes)

        dropped = before - self.num_episodes
        print_(f"BridgeV2 eval-min-frames filter ({self.split}): "
               f"kept {self.num_episodes}/{before}, "
               f"dropped {dropped} (< {required} frames "
               f"required by eval_min_num_frames)")

    def _filter_noisy_captions(self, log_path):
        before = len(self.episodes)
        reasons = [self._caption_drop_reason(caption) for caption in self.labels]

        dropped = [
            (ep, lab, reason)
            for ep, lab, reason in zip(self.episodes, self.labels, reasons)
            if reason is not None
        ]
        self.episodes = [ep for ep, r in zip(self.episodes, reasons) if r is None]
        self.labels = [lab for lab, r in zip(self.labels, reasons) if r is None]
        self.num_episodes = len(self.episodes)

        reason_counts = Counter(r for r in reasons if r is not None)
        print_(f"BridgeV2 caption filter ({self.split}): "
               f"kept {self.num_episodes}/{before}, dropped {before - self.num_episodes}")
        for reason, count in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
            print_(f"  --> {reason:20s}: {count}")
        if not _LANGDETECT_AVAILABLE:
            print_("  --> (langdetect not installed; non_english filter skipped)")

        try:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "reason", "caption"])
                for ep, lab, reason in dropped:
                    writer.writerow([ep, reason, lab])
            print_(f"  --> dropped-caption log: {log_path}")
        except OSError as e:
            print_(f"  --> could not write caption filter log to {log_path}: {e}")

    def _load_label(self, episode_dir):
        path = os.path.join(self.root, episode_dir, "task_description.txt")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"task_description.txt not found in {episode_dir}")
        with open(path, "r") as f:
            return f.read().strip()


class BridgeV2_precomputed_slots(BridgeV2):
    def __init__(
        self,
        datapath,
        split,
        num_frames=None,
        step_size=1,
        img_size=None,
        random_start=False,
        precomputed_slots_path=None,
        t5_embeddings_path=None,
        load_imgs=False,
        **kwargs,
    ):
        if "canonicalize_captions" not in kwargs:
            kwargs["canonicalize_captions"] = True
        if "canonicalize_mode" not in kwargs:
            kwargs["canonicalize_mode"] = "llm"

        super().__init__(
            datapath=datapath,
            split=split,
            num_frames=num_frames,
            step_size=step_size,
            img_size=img_size,
            random_start=random_start,
            text_tokenizer=None,
            **kwargs,
        )

        self.slots_root = precomputed_slots_path or os.path.join(datapath, "precomputed_slots")
        slots_split_dir = os.path.join(self.slots_root, self.split)
        assert os.path.isdir(slots_split_dir), (
            f"Precomputed slots dir not found: {slots_split_dir}."
        )
        self.slots_split_dir = slots_split_dir

        kept_eps, kept_labels = [], []
        missing = 0
        for ep, label in zip(self.episodes, self.labels):
            if os.path.isfile(os.path.join(slots_split_dir, f"{ep}.pt")):
                kept_eps.append(ep)
                kept_labels.append(label)
            else:
                missing += 1
        if missing:
            print_(f"BridgeV2_precomputed_slots: dropped {missing} episodes with no slots file")
        self.episodes = kept_eps
        self.labels = kept_labels
        self.num_episodes = len(self.episodes)

        self.t5_path = t5_embeddings_path or os.path.join(datapath, "t5_embeddings.pt")
        assert os.path.isfile(self.t5_path), (
            f"T5 embedding cache not found at {self.t5_path}."
        )
        cache = torch.load(self.t5_path, map_location="cpu", weights_only=False)
        self.t5_embeddings = cache["embeddings"]
        self.t5_attention_mask = cache["attention_mask"]
        self.caption_to_idx = cache["caption_to_idx"]
        self.t5_max_length = cache["max_length"]
        self.t5_model_name = cache["model_name"]
        print_(f"  --> Loaded T5 cache: {self.t5_embeddings.shape}, "
               f"hidden_dim={self.t5_embeddings.shape[-1]}, model={self.t5_model_name}")

        self.load_imgs = load_imgs

        unknown = [c for c in self.labels if c not in self.caption_to_idx]
        assert not unknown, (
            f"{len(unknown)} canonical labels are missing from the T5 cache "
            f"(sample: {sorted(set(unknown))[:3]}). The canonicalization JSON "
            f"and the T5 cache are out of sync. Regenerate the cache from "
            f"{self.llm_canonicalization_path} and update t5_embeddings_path."
        )

        print_(f"  --> num_episodes after slot/T5 alignment: {self.num_episodes}")

    def __len__(self):
        return self.num_episodes

    def _frame_indices(self, total_frames):
        if self.num_frames is None:
            return list(range(0, total_frames, self.step_size)), 0
        required = self.num_frames * self.step_size
        if total_frames < required:
            raise ValueError(
                f"Episode has {total_frames} frames but {required} needed "
                f"(num_frames={self.num_frames}, step_size={self.step_size})"
            )
        if self.random_start:
            start = random.randint(0, total_frames - required)
        else:
            start = 0
        return list(range(start, start + required, self.step_size)), start

    def _load_imgs_if_requested(self, episode_dir, frame_indices):
        if not self.load_imgs:
            return None
        color_dir = os.path.join(episode_dir, "color")
        files = sorted(os.listdir(color_dir))
        imgs = []
        for fi in frame_indices:
            img = Image.open(os.path.join(color_dir, files[fi])).convert("RGB")
            imgs.append(self.transform(img))
        return torch.stack(imgs)

    def __getitem__(self, idx):
        episode = self.episodes[idx]
        episode_dir = os.path.join(self.root, episode)
        slots_path = os.path.join(self.slots_split_dir, f"{episode}.pt")

        slots = torch.load(slots_path, map_location="cpu", weights_only=True)
        total_frames = slots.shape[0]
        frame_indices, start_frame = self._frame_indices(total_frames)
        slots = slots[frame_indices].float()

        caption = self.labels[idx]
        idx_in_cache = self.caption_to_idx[caption]
        t5_embedding = self.t5_embeddings[idx_in_cache].float()
        t5_mask = self.t5_attention_mask[idx_in_cache]

        imgs = self._load_imgs_if_requested(episode_dir, frame_indices)

        sample = {
            "slots": slots,
            "caption": caption,
            "t5_embedding": t5_embedding,
            "t5_attention_mask": t5_mask,
            "episode": episode,
            "start_frame_idx": start_frame,
        }
        if imgs is not None:
            sample["imgs"] = imgs

        return slots, sample

    def collate_fn(self, data):
        slots = torch.stack([d[0] for d in data], dim=0)
        batch = {
            "slots": slots,
            "caption": [d[1]["caption"] for d in data],
            "t5_embedding": torch.stack([d[1]["t5_embedding"] for d in data], dim=0),
            "t5_attention_mask": torch.stack([d[1]["t5_attention_mask"] for d in data], dim=0),
            "episode": [d[1]["episode"] for d in data],
            "start_frame_idx": [d[1]["start_frame_idx"] for d in data],
        }
        if "imgs" in data[0][1]:
            batch["imgs"] = torch.stack([d[1]["imgs"] for d in data], dim=0)
        return slots, batch


class BridgeV2_precomputed_latents(BridgeV2):
    def __init__(
        self,
        datapath,
        split,
        num_frames=None,
        step_size=1,
        img_size=None,
        random_start=False,
        precomputed_latents_path=None,
        t5_embeddings_path=None,
        load_imgs=False,
        **kwargs,
    ):
        if "canonicalize_captions" not in kwargs:
            kwargs["canonicalize_captions"] = True
        if "canonicalize_mode" not in kwargs:
            kwargs["canonicalize_mode"] = "llm"

        super().__init__(
            datapath=datapath,
            split=split,
            num_frames=num_frames,
            step_size=step_size,
            img_size=img_size,
            random_start=random_start,
            text_tokenizer=None,
            **kwargs,
        )

        assert precomputed_latents_path is not None, (
            "precomputed_latents_path is required."
        )
        self.latents_root = precomputed_latents_path
        latents_split_dir = os.path.join(self.latents_root, self.split)
        assert os.path.isdir(latents_split_dir), (
            f"Precomputed latents dir not found: {latents_split_dir}."
        )
        self.latents_split_dir = latents_split_dir

        manifest_path = os.path.join(self.latents_root, "manifest.json")
        self.manifest = {}
        if os.path.isfile(manifest_path):
            with open(manifest_path) as f:
                self.manifest = json.load(f)
            enc = self.manifest.get("encoder", "<unknown>")
            print_(f"  --> Loaded latents manifest: encoder={enc}, "
                   f"per_frame={self.manifest.get('per_frame', True)}, "
                   f"img_size={self.manifest.get('img_size')}")

        self.temporal_downsampling_factor = int(
            self.manifest.get("temporal_downsampling_factor", 1)
        )

        self.num_frames_frames = self.num_frames

        if self.num_frames is not None and self.temporal_downsampling_factor > 1:
            num_frames_tokens = (self.num_frames - 1) // self.temporal_downsampling_factor + 1
            print_(f"  --> Converting num_frames from {self.num_frames} (frames) to "
                   f"{num_frames_tokens} (tokens); temporal_downsampling_factor="
                   f"{self.temporal_downsampling_factor}")
            self.num_frames = num_frames_tokens

        kept_eps, kept_labels = [], []
        missing = 0
        for ep, label in zip(self.episodes, self.labels):
            if os.path.isfile(os.path.join(latents_split_dir, f"{ep}.pt")):
                kept_eps.append(ep)
                kept_labels.append(label)
            else:
                missing += 1
        if missing:
            print_(f"BridgeV2_precomputed_latents: dropped {missing} episodes "
                   f"with no latents file")
        self.episodes = kept_eps
        self.labels = kept_labels
        self.num_episodes = len(self.episodes)

        self.t5_path = t5_embeddings_path or os.path.join(datapath, "t5_embeddings.pt")
        assert os.path.isfile(self.t5_path), (
            f"T5 embedding cache not found at {self.t5_path}."
        )
        cache = torch.load(self.t5_path, map_location="cpu", weights_only=False)
        self.t5_embeddings = cache["embeddings"]
        self.t5_attention_mask = cache["attention_mask"]
        self.caption_to_idx = cache["caption_to_idx"]
        self.t5_max_length = cache["max_length"]
        self.t5_model_name = cache["model_name"]
        print_(f"  --> Loaded T5 cache: {self.t5_embeddings.shape}, "
               f"hidden_dim={self.t5_embeddings.shape[-1]}, model={self.t5_model_name}")

        self.load_imgs = load_imgs

        unknown = [c for c in self.labels if c not in self.caption_to_idx]
        assert not unknown, (
            f"{len(unknown)} canonical labels are missing from the T5 cache "
            f"(sample: {sorted(set(unknown))[:3]}). Regenerate the cache from "
            f"{self.llm_canonicalization_path}."
        )

        print_(f"  --> num_episodes after latents/T5 alignment: {self.num_episodes}")

    def __len__(self):
        return self.num_episodes

    def _frame_indices(self, total_frames):
        if self.num_frames is None:
            return list(range(0, total_frames, self.step_size)), 0
        required = self.num_frames * self.step_size
        if total_frames < required:
            raise ValueError(
                f"Latent has {total_frames} entries but {required} needed "
                f"(num_frames={self.num_frames}, step_size={self.step_size})"
            )
        if self.random_start:
            start = random.randint(0, total_frames - required)
        else:
            start = 0
        return list(range(start, start + required, self.step_size)), start

    def _load_imgs_if_requested(self, episode_dir, frame_indices):
        if not self.load_imgs:
            return None
        color_dir = os.path.join(episode_dir, "color")
        files = sorted(os.listdir(color_dir))
        imgs = []
        for fi in frame_indices:
            img = Image.open(os.path.join(color_dir, files[fi])).convert("RGB")
            imgs.append(self.transform(img))
        return torch.stack(imgs)

    def __getitem__(self, idx):
        episode = self.episodes[idx]
        episode_dir = os.path.join(self.root, episode)
        latents_path = os.path.join(self.latents_split_dir, f"{episode}.pt")

        latents = torch.load(latents_path, map_location="cpu", weights_only=True)
        total = latents.shape[0]
        frame_indices, start_frame = self._frame_indices(total)
        latents = latents[frame_indices].float()

        caption = self.labels[idx]
        idx_in_cache = self.caption_to_idx[caption]
        t5_embedding = self.t5_embeddings[idx_in_cache].float()
        t5_mask = self.t5_attention_mask[idx_in_cache]

        imgs = self._load_imgs_if_requested(episode_dir, frame_indices) \
            if self.manifest.get("per_frame", True) else None

        sample = {
            "latents": latents,
            "caption": caption,
            "t5_embedding": t5_embedding,
            "t5_attention_mask": t5_mask,
            "episode": episode,
            "start_frame_idx": start_frame,
        }
        if imgs is not None:
            sample["imgs"] = imgs

        return latents, sample

    def collate_fn(self, data):
        latents = torch.stack([d[0] for d in data], dim=0)
        batch = {
            "latents": latents,
            "caption": [d[1]["caption"] for d in data],
            "t5_embedding": torch.stack([d[1]["t5_embedding"] for d in data], dim=0),
            "t5_attention_mask": torch.stack([d[1]["t5_attention_mask"] for d in data], dim=0),
            "episode": [d[1]["episode"] for d in data],
            "start_frame_idx": [d[1]["start_frame_idx"] for d in data],
        }
        if "imgs" in data[0][1]:
            batch["imgs"] = torch.stack([d[1]["imgs"] for d in data], dim=0)
        return latents, batch
