import piqa
import torch
from lib.fvd import get_fvd_logits, frechet_distance
from models import InceptionI3d

import torch.nn.functional as F


class Metric:
    def __init__(self):
        self.results = None
        self.reset()

    def reset(self):
        raise NotImplementedError("Base class does not implement 'reset' functionality")

    def accumulate(self):
        raise NotImplementedError("Base class does not implement 'accumulate' functionality")

    def aggregate(self):
        raise NotImplementedError("Base class does not implement 'aggregate' functionality")

    def _shape_check(self, tensor, name="Preds"):
        if len(tensor.shape) not in [3, 4, 5]:
            raise ValueError(f"{name} has shape {tensor.shape}, but it must have one of the folling shapes\n"
                             " - (B, F, C, H, W) for frame or heatmap prediction.\n"
                             " - (B, F, D) or (B, F, N_joints, N_coords) for pose skeleton prediction")


class MSE(Metric):
    LOWER_BETTER = True

    def __init__(self):
        super().__init__()
        self.values = []

    def reset(self):
        self.values = []

    def accumulate(self, preds, targets, **kwargs):
        self._shape_check(tensor=preds, name="Preds")
        self._shape_check(tensor=targets, name="Targets")

        B, F, C, H, W = preds.shape
        preds, targets = preds.view(B * F, C, H, W), targets.view(B * F, C, H, W)
        cur_mse = (preds.float() - targets.float()).pow(2).mean(dim=(-1, -2, -3))
        cur_mse = cur_mse.view(B, F)
        self.values.append(cur_mse)
        return cur_mse.mean()

    def aggregate(self):
        all_values = torch.cat(self.values, dim=0)
        mean_values = all_values.mean()
        frame_values = all_values.mean(dim=0)
        return float(mean_values), frame_values

class PSNR(Metric):
    LOWER_BETTER = False

    def __init__(self):
        super().__init__()
        self.values = []

    def reset(self):
        self.values = []

    def accumulate(self, preds, targets, **kwargs):
        self._shape_check(tensor=preds, name="Preds")
        self._shape_check(tensor=targets, name="Targets")

        B, F, C, H, W = preds.shape
        preds, targets = preds.view(B * F, C, H, W), targets.view(B * F, C, H, W)
        cur_psnr = piqa.psnr.psnr(preds, targets)
        cur_psnr = cur_psnr.view(B, F)
        self.values.append(cur_psnr)
        return cur_psnr.mean()

    def aggregate(self):
        all_values = torch.cat(self.values, dim=0)
        mean_values = all_values.mean()
        frame_values = all_values.mean(dim=0)
        return float(mean_values), frame_values

class SSIM(Metric):
    LOWER_BETTER = False

    def __init__(self, window_size=11, sigma=1.5, n_channels=3):
        self.ssim = piqa.ssim.SSIM(
                window_size=window_size,
                sigma=sigma,
                n_channels=n_channels,
                reduction=None
            )
        super().__init__()
        self.values = []

    def reset(self):
        self.values = []

    def accumulate(self, preds, targets, **kwargs):
        self._shape_check(tensor=preds, name="Preds")
        self._shape_check(tensor=targets, name="Targets")
        if self.ssim.kernel.device != preds.device:
            self.ssim = self.ssim.to(preds.device)

        B, F, C, H, W = preds.shape
        preds, targets = preds.view(B * F, C, H, W), targets.view(B * F, C, H, W)
        cur_ssim = self.ssim(preds, targets)
        cur_ssim = cur_ssim.view(B, F)
        self.values.append(cur_ssim)
        return cur_ssim.mean()

    def aggregate(self):
        all_values = torch.cat(self.values, dim=0)
        mean_values = all_values.mean()
        frame_values = all_values.mean(dim=0)
        return float(mean_values), frame_values

class LPIPS(Metric):
    LOWER_BETTER = True

    def __init__(self, network="alex", pretrained=True, reduction=None):
        self.lpips = piqa.lpips.LPIPS(
                network=network,

                reduction=reduction
            )
        super().__init__()
        self.values = []

    def reset(self):
        self.values = []

    def accumulate(self, preds, targets, **kwargs):
        self._shape_check(tensor=preds, name="Preds")
        self._shape_check(tensor=targets, name="Targets")
        if not hasattr(self.lpips, "device"):
            self.lpips = self.lpips.to(preds.device)
            self.lpips.device = preds.device

        B, F, C, H, W = preds.shape
        preds, targets = preds.view(B * F, C, H, W), targets.view(B * F, C, H, W)
        cur_lpips = self.lpips(preds, targets)
        cur_lpips = cur_lpips.view(B, F)
        self.values.append(cur_lpips)
        return cur_lpips.mean()

    def aggregate(self):
        all_values = torch.cat(self.values, dim=0)
        mean_values = all_values.mean()
        frame_values = all_values.mean(dim=0)
        return float(mean_values), frame_values

class FVD(Metric):
    LOWER_BETTER = True

    def __init__(self, checkpoint):
        self.i3d = InceptionI3d(400, in_channels=3)
        filepath = checkpoint
        self.i3d.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))
        self.i3d.eval()

        self.fake_embeddings_stack = []
        self.real_embeddings_stack = []

        super().__init__()
        self.values = []

    def reset(self):
        self.values = []
        self.fake_embeddings_stack = []
        self.real_embeddings_stack = []

    def preprocess_for_i3d(self, videos):
        B, L, C, H, W = videos.shape

        videos_reshaped = videos.view(B * L, C, H, W)
        videos_resized = F.interpolate(videos_reshaped, size=(224, 224), mode='bilinear', align_corners=False)
        videos_resized = videos_resized.view(B, L, C, 224, 224)

        target_frames = 16
        if L < target_frames:
            pad_frames = target_frames - L
            last_frame = videos_resized[:, -1:, :, :, :].repeat(1, pad_frames, 1, 1, 1)
            videos_processed = torch.cat([videos_resized, last_frame], dim=1)
        elif L > target_frames:
            indices = torch.linspace(0, L - 1, target_frames).long()
            videos_processed = videos_resized[:, indices, :, :, :]
        else:
            videos_processed = videos_resized

        return videos_processed

    @torch.no_grad()
    def accumulate(self, preds, targets, **kwargs):
        if not hasattr(self.i3d, '_device_set'):
            self.i3d = self.i3d.to(preds.device)
            self.i3d._device_set = True

        fake = self.preprocess_for_i3d(preds)
        real = self.preprocess_for_i3d(targets)

        fake_np = fake.permute(0, 1, 3, 4, 2).cpu().numpy()
        fake_np = (fake_np * 255).astype('uint8')

        real_np = real.permute(0, 1, 3, 4, 2).cpu().numpy()
        real_np = (real_np * 255).astype('uint8')

        self.fake_embeddings_stack.append(get_fvd_logits(fake_np, i3d=self.i3d, device=preds.device))
        self.real_embeddings_stack.append(get_fvd_logits(real_np, i3d=self.i3d, device=preds.device))

        return

    def aggregate(self):
        if len(self.fake_embeddings_stack) == 0:
            return 0.0, torch.tensor([])

        fake_embeddings = torch.cat(self.fake_embeddings_stack, dim=0)
        real_embeddings = torch.cat(self.real_embeddings_stack, dim=0)
        fvd = frechet_distance(fake_embeddings.clone(), real_embeddings)

        return fvd.item(), torch.tensor([])
