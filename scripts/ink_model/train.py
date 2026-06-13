"""Train the ink U-Net on infinite on-the-fly synthetic strips.

python train.py --steps 20000 --out ckpt/
Resume: --resume ckpt/ink-latest.pt
"""

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from gen_data import sample
from model import InkUNet, param_count


class SyntheticStrips(IterableDataset):
    """Infinite stream; fixed width per worker batch so default collation works."""

    def __init__(self, width: int = 320, seed: int = 0):
        self.width = width
        self.seed = seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker = info.id if info else 0
        rng = random.Random(self.seed + worker * 7919 + os.getpid())
        while True:
            img, cov = sample(rng, width=self.width)
            yield (
                torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))),
                torch.from_numpy(cov[None]),
            )


def validation_batch(n: int = 32, width: int = 320, seed: int = 1234):
    rng = random.Random(seed)
    imgs, covs = [], []
    for _ in range(n):
        img, cov = sample(rng, width=width)
        imgs.append(torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))))
        covs.append(torch.from_numpy(cov[None]))
    return torch.stack(imgs), torch.stack(covs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="ckpt")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--val-every", type=int, default=1000)
    args = ap.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = InkUNet().to(device)
    print(f"device={device} params={param_count(model):,}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_step = 0
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start_step = state["step"]
        print(f"resumed from {args.resume} at step {start_step}")

    loader = DataLoader(
        SyntheticStrips(width=args.width),
        batch_size=args.batch,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None,
    )
    val_img, val_cov = validation_batch(width=args.width)
    val_img, val_cov = val_img.to(device), val_cov.to(device)

    os.makedirs(args.out, exist_ok=True)
    model.train()
    t0 = time.time()
    running = 0.0
    for step, (img, cov) in enumerate(loader, start=start_step + 1):
        if step > args.steps:
            break
        img, cov = img.to(device), cov.to(device)
        logits = model(img)
        loss = F.binary_cross_entropy_with_logits(logits, cov)
        opt.zero_grad()
        loss.backward()
        opt.step()
        running += loss.item()

        if step % args.log_every == 0:
            rate = args.log_every * args.batch / (time.time() - t0)
            print(
                f"step {step:>6}  loss {running / args.log_every:.4f}  {rate:.0f} strips/s",
                flush=True,
            )
            running, t0 = 0.0, time.time()
        if step % args.val_every == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                val_loss = F.binary_cross_entropy_with_logits(model(val_img), val_cov).item()
            model.train()
            print(f"step {step:>6}  VAL loss {val_loss:.4f}", flush=True)
            state = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step}
            torch.save(state, os.path.join(args.out, "ink-latest.pt"))
    print("done")


if __name__ == "__main__":
    main()
