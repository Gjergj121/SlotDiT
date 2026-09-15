import torch
import math
import torch.nn.functional as F
from lib.logger import print_

def preprocess_single(video, resolution, sequence_length=16):
    video = video.permute(0, 3, 1, 2).float() / 255.
    t, c, h, w = video.shape

    F_curr = video.shape[0]
    if F_curr < sequence_length:
        pad = sequence_length - F_curr
        video = F.pad(video, (0, 0, 0, 0, 0, 0, 0, pad))
    else:
        video = video[:sequence_length]

    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)

    video = F.interpolate(video, size=target_size, mode='bilinear', align_corners=False)

    t, c, h, w = video.shape
    w_start = (w - resolution) // 2
    h_start = (h - resolution) // 2
    video = video[:, :, h_start:h_start + resolution, w_start:w_start + resolution]

    video = video.permute(1, 0, 2, 3).contiguous()

    video -= 0.5

    return video

def preprocess(videos, target_resolution=224):
    b, t, h, w, c = videos.shape
    videos = torch.from_numpy(videos)
    videos = torch.stack([preprocess_single(video, target_resolution) for video in videos])
    return videos * 2

def get_fvd_logits(videos, i3d, device, target_resolution=224):
    videos = preprocess(videos, target_resolution=target_resolution)
    embeddings = get_logits(i3d, videos, device)
    return embeddings


def _symmetric_matrix_square_root(mat, eps=1e-10):
    u, s, v = torch.svd(mat)
    si = torch.where(s < eps, s, torch.sqrt(s))
    return torch.matmul(torch.matmul(u, torch.diag(si)), v.t())

def trace_sqrt_product(sigma, sigma_v):
    sqrt_sigma = _symmetric_matrix_square_root(sigma)
    sqrt_a_sigmav_a = torch.matmul(sqrt_sigma, torch.matmul(sigma_v, sqrt_sigma))
    return torch.trace(_symmetric_matrix_square_root(sqrt_a_sigmav_a))

def cov(m, rowvar=False):
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()

    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()
    return fact * m.matmul(mt).squeeze()

def frechet_distance(x1, x2):
    x1 = x1.flatten(start_dim=1)
    x2 = x2.flatten(start_dim=1)
    m, m_w = x1.mean(dim=0), x2.mean(dim=0)
    sigma, sigma_w = cov(x1, rowvar=False), cov(x2, rowvar=False)

    sqrt_trace_component = trace_sqrt_product(sigma, sigma_w)
    trace = torch.trace(sigma + sigma_w) - 2.0 * sqrt_trace_component

    mean = torch.sum((m - m_w) ** 2)
    fd = trace + mean
    return fd


def get_logits(i3d, videos, device):
    with torch.no_grad():
        logits = []
        for i in range(0, videos.shape[0], 16):
            batch = videos[i:i + 16].to(device)
            logits.append(i3d(batch))

        logits = torch.cat(logits, dim=0)
        return logits.cpu()
