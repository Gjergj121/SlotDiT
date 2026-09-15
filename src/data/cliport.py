import os
import random
import imageio
import torch
from torchvision import transforms
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from lib.logger import print_
import numpy as np
import re
from PIL import Image
import pickle

from transformers import T5Tokenizer
class CLIPort:
    IMG_SIZES = [
            (640, 720),
            (320, 360),
            (160, 180),
            (80, 90),
            (112, 112),
            (168, 168)
        ]
    VARIANTS = ["PackingShapes", "PutBlocksIntoBowl"]

    POS_LOW  = np.array([0.25, -0.5, 0.0],  dtype=np.float32)
    POS_HIGH = np.array([0.75,  0.5, 0.28], dtype=np.float32)
    POS_CENTER    = (POS_HIGH + POS_LOW) / 2.0
    POS_HALFRANGE = (POS_HIGH - POS_LOW) / 2.0

    def __init__(self, datapath, split, num_frames=None, step_size=1, img_size=None,
                 random_start=False, get_depth=False, get_segm=False, **kwargs):
        assert split in ["train", "val", "valid", "test", "eval"], f"Unknown split {split}..."
        split = "val" if split in ["val", "valid", "eval", "test"] else split

        self.datapath = os.path.join(f"{datapath}", f"{split}")
        self.split = split
        self.num_frames = num_frames
        self.step_size = step_size
        self.img_size = img_size if img_size is not None else self.IMG_SIZES[0]
        self.random_start = random_start if split == "train" else False
        self.get_depth = get_depth
        self.get_segm = get_segm
        self.get_actions = kwargs.get('get_actions', False)
        self.get_first_and_last_frame = kwargs.get('get_first_and_last_frame', False)
        self.normalize_actions = kwargs.get('normalize_actions', False)
        self.position_only_actions = kwargs.get('position_only_actions', False)
        self.adaptable_frame_sampling = kwargs.get('adaptable_frame_sampling', False) if split == "train" else False

        print_(f"Split {self.split}, Random start = {self.random_start}")
        print_("Getting depth images!") if self.get_depth else print_("Not getting depth images!")
        print_("Getting segmentation images!") if self.get_segm else print_("Not getting segmentation images!")
        print_("Getting action labels!") if self.get_actions else print_("Not getting action labels!")
        if self.get_first_and_last_frame:
            print_("Getting only first and last frames of each episode!")
        if self.normalize_actions:
            print_("  → Normalizing action positions to [-1, 1]")
        if self.position_only_actions:
            print_("  → Using position-only actions (6-dim: pick_xyz + place_xyz)")
        if self.adaptable_frame_sampling:
            print_("  → Using adaptable frame sampling (random start + uniform intermediate + final)")

        self.resizer = transforms.Resize(
                self.img_size,
                interpolation=transforms.InterpolationMode.BILINEAR
            )
        self.mask_resizer = transforms.Resize(
                self.img_size,
                interpolation=transforms.InterpolationMode.NEAREST
            )

        self.segm_transform = transforms.Compose([
                transforms.Resize(
                        self.img_size,
                        interpolation=transforms.InterpolationMode.NEAREST
                    ),
                transforms.ToTensor()
            ])

        self.episodes = self.fetch_episodes()
        self.num_episodes = len(self.episodes)
        print_(f"Loaded CLIPort {self.split} set, containing {self.num_episodes} episodes")
        print_(f"Step size set to {self.step_size}")
        with ThreadPoolExecutor() as executor:
            self.labels = list(tqdm(
                    executor.map(self.load_label, self.episodes),
                    total=self.num_episodes
                ))

        if self.get_actions:
            self.actions_dir = os.path.join(
                self.datapath,
                f"put-block-in-bowl-seen-colors-{split}",
                "action"
            )
            if not os.path.exists(self.actions_dir):
                raise FileNotFoundError(f"Actions directory not found: {self.actions_dir} where {self.datapath} is used for episodes.")
            print_(f"Loading actions from: {self.actions_dir}")

        SLOTS_PATH = kwargs.get('slots_path', None)
        print_(f"Using slots from path: {SLOTS_PATH}")

        split_path = "valid" if split == "val" or split == "test" else "train"
        if SLOTS_PATH is not None:
            self.slots_dir = os.path.join(SLOTS_PATH, split_path)
            self.slots_files = [f for f in os.listdir(self.slots_dir) if f.endswith('.pt')]

            self.slots_files.sort(key=lambda x: int(re.search(r'video(\d+)\.pt', x).group(1)))
        else:
            self.slots_dir = None

        return

    def __len__(self):
        return self.num_episodes

    def __getitem__(self, idx):
        cur_episode = self.episodes[idx]

        color_frames, depth_frames, segmentation_frames, start_frame_idx, slots, actions = self.load_episode(cur_episode, idx)

        caption = self.labels[idx]

        sample = {
            "imgs": color_frames,
            "depth": depth_frames,
            "segm": segmentation_frames,
            "caption": caption,
            "episode": cur_episode,
            "start_frame_idx": start_frame_idx,
            "slots": slots,
            "actions": actions
        }
        return color_frames, sample

    def fetch_episodes(self):
        ignore_episodes = []

        all_episodes = [f for f in os.listdir(self.datapath) if f.startswith("episode") and f not in ignore_episodes]
        all_episodes = sorted(all_episodes, key=lambda x: int(x.split("episode")[-1]))

        return all_episodes

    def load_label(self, episode_dir):
        task_caption_file = os.path.join(self.datapath, episode_dir, 'task_description.txt')
        if not os.path.exists(task_caption_file):
            raise FileNotFoundError(f"Task description file not found: {task_caption_file}")
        try:
            with open(task_caption_file, 'r') as f:
                label = f.read().strip()
        except Exception as e:
            raise IOError(f"Error reading: {task_caption_file = }, {str(e)}")
        return label

    def load_episode(self, episode, idx):
        color_dir = os.path.join(self.datapath, episode, 'color')
        depth_dir = os.path.join(self.datapath, episode, 'depth')
        segm_dir = os.path.join(self.datapath, episode, 'segm')
        assert os.path.exists(color_dir), f"RBG-Img dir. does not exist for {episode}"
        if self.get_depth:
            assert os.path.exists(depth_dir), f"Depth-Map dir. does not exist for {episode}"
        if self.get_segm:
            assert os.path.exists(segm_dir), f"Segm-Map dir. does not exist for {episode}"

        frame_files = sorted(os.listdir(color_dir))
        num_frames = len(frame_files)

        if self.get_first_and_last_frame:
            frame_indices = [0, num_frames - 1]
            start_frame_idx = 0
        elif self.adaptable_frame_sampling and self.num_frames is not None and self.num_frames >= 2:
            final_frame_idx = num_frames - 1
            max_start = max(final_frame_idx - (self.num_frames - 1), 0)
            start_frame_idx = random.randint(0, max_start)
            if self.num_frames == 2:
                frame_indices = [start_frame_idx, final_frame_idx]
            else:
                intermediate = sorted(random.sample(
                    range(start_frame_idx + 1, final_frame_idx), self.num_frames - 2
                ))
                frame_indices = [start_frame_idx] + intermediate + [final_frame_idx]
        elif self.num_frames is not None:
            if num_frames < (self.num_frames * self.step_size) :
                raise ValueError(f"{self.num_frames * self.step_size = } are required, " +
                                f"but only {num_frames = } are available for {episode = }...")

            if self.random_start:
                max_start = max(num_frames - (self.num_frames * self.step_size), 0)
                start_frame_idx = random.randint(0, max_start)
            else:
                start_frame_idx = 0

            frame_indices = range(
                    start_frame_idx,
                    start_frame_idx + self.num_frames * self.step_size,
                    self.step_size
                )
        else:
            start_frame_idx = 0
            frame_indices = range(
                    start_frame_idx,
                    num_frames,
                    self.step_size
                )

        slots = None
        if self.slots_dir is not None:
            slots_path = os.path.join(self.slots_dir, self.slots_files[idx])
            slots = torch.load(slots_path, weights_only=True)
            slots = slots[frame_indices]

        color_frames, depth_frames, segm_frames = [], [], []
        for idx in frame_indices:
            frame_file = frame_files[idx]
            frame_num = frame_file.split("_")[0]
            color_frame = self._load_img(os.path.join(color_dir, f"{frame_num}_color.png"))
            color_frames.append(color_frame)

            if self.get_depth:
                depth_frame = self._load_img(os.path.join(depth_dir, f"{frame_num}_depth.png"))
                depth_frames.append(depth_frame)
            if self.get_segm:
                segm_frame = self._load_segm(os.path.join(segm_dir, f"{frame_num}_segm.png"))

                segm_frames.append(segm_frame)

        color_frames = torch.stack(color_frames).permute(0, 3, 1, 2) / 255

        color_frames = self.resizer(color_frames)

        if self.get_depth:
            depth_frames = torch.stack(depth_frames).unsqueeze(1) / 5
            depth_frames = self.resizer(depth_frames)
        if self.get_segm:
            segm_frames = torch.stack(segm_frames)

        actions = None
        if self.get_actions:
            actions = self._load_actions(episode)

        return color_frames, depth_frames, segm_frames, start_frame_idx, slots, actions

    def collate_fn(self, data):
        actions_list = [d[1]['actions'] for d in data]
        if actions_list[0] is not None:
            actions_batch = torch.stack(actions_list, dim=0)
        else:
            actions_batch = None

        d0 = torch.stack([d[0] for d in data], dim=0)
        d1 = {
            "imgs": d0,
            "depth": torch.stack([torch.as_tensor(d[1]['depth']) for d in data], dim=0),
            "segm": torch.stack([torch.as_tensor(d[1]['segm']) for d in data], dim=0),
            "caption": [d[1]['caption'] for d in data],
            "episode": [d[1]['episode'] for d in data],
            "start_frame_idx": [d[1]['start_frame_idx'] for d in data],
            "slots": torch.stack([d[1]['slots'] for d in data], dim=0) if data[0][1]['slots'] is not None else None,
            "actions": actions_batch
        }

        return d0, d1

    @staticmethod
    def _load_img(p):
        return torch.tensor(imageio.imread(p))

    def _load_segm(self, p):
        with open(p, "rb") as f:
            img = Image.open(f).convert("L")
        return self.segm_transform(img)

    def _normalize_pos(self, pos: torch.Tensor) -> torch.Tensor:
        center = torch.from_numpy(self.POS_CENTER)
        half   = torch.from_numpy(self.POS_HALFRANGE)
        return (pos - center) / half

    @staticmethod
    def denormalize_pos(pos_norm: np.ndarray,
                        center: np.ndarray = None,
                        half_range: np.ndarray = None) -> np.ndarray:
        if center is None:
            center = np.array([0.50, 0.00, 0.14], dtype=np.float32)
        if half_range is None:
            half_range = np.array([0.25, 0.50, 0.14], dtype=np.float32)
        return pos_norm * half_range + center

    def _load_actions(self, episode):
        import glob

        episode_num = int(episode.replace("episode", ""))

        action_num = episode_num - 1

        action_pattern = os.path.join(self.actions_dir, f"{action_num:06d}-*.pkl")

        matching_files = glob.glob(action_pattern)

        if len(matching_files) == 0:
            raise FileNotFoundError(
                f"Action file not found with pattern {action_pattern} for {episode}"
            )
        if len(matching_files) > 1:
            raise ValueError(
                f"Multiple action files found for pattern {action_pattern}: {matching_files}"
            )

        action_path = matching_files[0]

        try:
            with open(action_path, 'rb') as f:
                actions = pickle.load(f)

            action_dict = actions[0]

            pose0_pos = torch.from_numpy(action_dict['pose0'][0]).float()
            pose0_ori = torch.from_numpy(action_dict['pose0'][1]).float()

            pose1_pos = torch.from_numpy(action_dict['pose1'][0]).float()
            pose1_ori = torch.from_numpy(action_dict['pose1'][1]).float()

            if self.normalize_actions:
                pose0_pos = self._normalize_pos(pose0_pos)
                pose1_pos = self._normalize_pos(pose1_pos)

            if self.position_only_actions:
                action_tensor = torch.cat([pose0_pos, pose1_pos])
            else:
                action_tensor = torch.cat([pose0_pos, pose0_ori, pose1_pos, pose1_ori])

            return action_tensor
        except Exception as e:
            raise IOError(f"Error loading action file {action_path}: {str(e)}")
