"""Bench: how much does moving the *movable* generation work to the GPU help?

Text rasterization (PIL FreeType) is stuck on the CPU. Everything after it —
backgrounds, compositing, noise/illumination, blur, resize — is batchable torch and
can run on the GPU. This measures, per CPU core:
  - full CPU sample() rate (current pipeline)
  - raster-only rate (the part that MUST stay on CPU)
and on the GPU:
  - batched augment rate (the movable part)
so we can estimate the realistic ceiling of a CPU-raster + GPU-augment split.

    python3 bench_gpu_gen.py
"""

import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from gen_data import render_coverage, gradient_field, degrade

H, W = 32, 320          # representative native strip
B = 256                 # batch
REPS = 8


def time_it(fn, iters):
    fn()  # warm
    t0 = time.time()
    for _ in range(iters):
        fn()
    return time.time() - t0


def cpu_full(rng):
    cov, fill = render_coverage(rng, W, H)
    bg = gradient_field(rng, H, W)
    ink = np.random.rand(3).astype(np.float32)
    img = cov[..., None] * ink + (1 - cov[..., None]) * bg
    degrade(img, rng, H)


def cpu_raster(rng):
    render_coverage(rng, W, H)


def gpu_augment(cov, device):
    """Movable part for a whole batch on the GPU: bg + composite + noise + blur + resize."""
    b = cov.shape[0]
    ink = torch.rand(b, 3, 1, 1, device=device)
    c0 = torch.rand(b, 3, 1, 1, device=device)
    c1 = torch.rand(b, 3, 1, 1, device=device)
    t = torch.rand(b, 1, 1, W, device=device)
    bg = t * c1 + (1 - t) * c0
    img = cov * ink + (1 - cov) * bg
    img = img + torch.randn_like(img) * 0.02
    k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=device)
    k = (k / k.sum()).view(1, 1, 1, 5).expand(3, 1, 1, 5)
    img = F.conv2d(img, k, padding=(0, 2), groups=3)
    img = F.interpolate(img, size=(48, W), mode="bilinear", align_corners=False)
    return img


def main():
    rng = random.Random(0)
    rng.seed(0)
    rate_full = REPS * B / time_it(lambda: [cpu_full(rng) for _ in range(B)], REPS)
    rate_raster = REPS * B / time_it(lambda: [cpu_raster(rng) for _ in range(B)], REPS)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cov = torch.rand(B, 1, H, W, device=device)
    if device == "cuda":
        def gpu_run():
            gpu_augment(cov, device); torch.cuda.synchronize()
    else:
        gpu_run = lambda: gpu_augment(cov, device)
    rate_gpu = REPS * B / time_it(gpu_run, REPS)

    print(f"device={device}  batch={B}")
    print(f"CPU full sample()/core : {rate_full:8.0f} strips/s")
    print(f"CPU raster-only /core  : {rate_raster:8.0f} strips/s  ({rate_raster/rate_full:.2f}x full)")
    print(f"GPU batched augment    : {rate_gpu:8.0f} strips/s")


if __name__ == "__main__":
    main()
