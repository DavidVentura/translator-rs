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

import gpu_degrade
from gen_data import sample, stream
from model import InkUNet, param_count

# Module level so spawn workers (which re-import this module but never run main) share via
# /tmp files instead of the default file_descriptor strategy, whose shm/semaphore leak piles
# prefetched batches into RAM-backed /dev/shm and OOMs mid-run.
torch.multiprocessing.set_sharing_strategy("file_system")


class SyntheticStrips(IterableDataset):
    """Infinite stream of pre-stacked batches.

    Each worker stacks a whole batch and yields it as one tensor pair, so the
    DataLoader ships one transfer per batch instead of `batch` small arrays. That
    collapses the per-strip multiprocessing IPC that otherwise caps throughput well
    below what the workers can generate. Use with `DataLoader(batch_size=None)`.
    """

    def __init__(self, batch: int, width: int = 320, seed: int = 0, reuse: int = 1,
                 apply_degrade: bool = True):
        self.batch = batch
        self.width = width
        self.seed = seed
        self.reuse = reuse
        self.apply_degrade = apply_degrade

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker = info.id if info else 0
        rng = random.Random(self.seed + worker * 7919 + os.getpid())
        # apply_degrade=False (GPU path): degrade + legibility run batched on the GPU.
        gen = stream(rng, self.width, self.reuse, apply_degrade=self.apply_degrade)
        while True:
            imgs, covs, bolds, nhs = [], [], [], []
            for _ in range(self.batch):
                img, cov, bold, native_h = next(gen)
                imgs.append(torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))))
                covs.append(torch.from_numpy(cov[None]))
                bolds.append(torch.from_numpy(bold[None]))
                nhs.append(native_h)
            yield (torch.stack(imgs), torch.stack(covs), torch.stack(bolds),
                   torch.tensor(nhs, dtype=torch.float32))


def validation_batch(n: int = 32, width: int = 320, seed: int = 1234):
    rng = random.Random(seed)
    imgs, covs, bolds = [], [], []
    for _ in range(n):
        img, cov, bold = sample(rng, width=width)
        imgs.append(torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))))
        covs.append(torch.from_numpy(cov[None]))
        bolds.append(torch.from_numpy(bold[None]))
    return torch.stack(imgs), torch.stack(covs), torch.stack(bolds)


# Confident-margin band: targets at/above CONF_BOLD_T are clearly bold, at/below CONF_NORM_T
# clearly normal; the rest is left to pure regression. MARGIN_LOGIT = logit(0.75): the margin
# pushes confident-bold logits above +M (prob >= 0.75) and confident-normal below -M (prob <=
# 0.25), so the two clusters are repelled out of the [0.25,0.75] band. On the logistic targets
# (gen_data) the confident bands hold most of the mass; on a linear ramp they'd hold only the
# already-separated extremes, which is why this term is paired with the logistic calibration.
MARGIN_CONF_BOLD_T = 0.75
MARGIN_CONF_NORM_T = 0.25
MARGIN_LOGIT = 1.0986

# Asymmetric matte Tversky: penalise misses (FN, β) above spills (FP, α). The matting cost is
# asymmetric — a missed-ink pixel leaves residual text the inpainter can't fill, an over-marked
# pixel is benign edge spill the inpainter absorbs — so the matte should be biased toward recall.
MATTE_TVERSKY_ALPHA = 0.3
MATTE_TVERSKY_BETA = 0.7


def ink_losses(logits, cov, bold, bold_weight=1.5, strip_w=None, bold_asym=1.5,
               bold_asym_t0=0.15, bold_margin=0.0, matte_tversky=0.0):
    """Matte BCE over the whole strip + asymmetric bold L1 *masked to ink*. `bold` is the
    continuous per-pixel stroke-width target in [0,1] (gen_data `_font_stroke_ratio` through
    the logistic `_target_q`), not a binary class: the head regresses how thick the ink is so
    the runtime can threshold boldness wherever it wants.

    The bold term is base L1 (keeps calibration everywhere) plus an extra penalty for
    *overshoot* (predicting bolder than truth) concentrated on *thin* ink (low target):
    plain L1 is lenient near 0, so thin text drifts up and gets falsely emboldened, which
    a binary BCE used to slam to 0. `bold_asym` scales that overshoot penalty and
    `bold_asym_t0` is where the thinness gate fades to 0 (so genuine bold, target >= t0, is
    pure L1 — undisturbed). Undershoot (missing a bold) is never extra-penalised; it's the
    benign side of the asymmetric preference. Masking to ink keeps the empty background out.

    `bold_margin` weights a logit-space confident-margin hinge: it lifts clearly-bold scores
    toward the rail (L1 alone leaves them compressed below the saturated target) and pulls the
    clearly-normal tail down, while leaving the ambiguous middle to regression. 0 disables it.
    `strip_w` (B,) is the per-strip legibility weight (illegible strips contribute zero);
    None = all ones. Returns (total, matte, bold, margin) for logging."""
    matte_logit, bold_logit = logits[:, :1], logits[:, 1:]
    if strip_w is None:
        strip_w = torch.ones(logits.shape[0], device=logits.device)
    w = strip_w.view(-1, 1, 1, 1)
    bce_m = F.binary_cross_entropy_with_logits(matte_logit, cov, reduction="none")
    loss_matte = (bce_m * w).sum() / w.expand_as(bce_m).sum().clamp_min(1.0)
    p_m = torch.sigmoid(matte_logit)
    tp = (p_m * cov * w).sum()
    fp = (p_m * (1.0 - cov) * w).sum()
    fn = ((1.0 - p_m) * cov * w).sum()
    loss_tversky = 1.0 - (tp + 1.0) / (tp + MATTE_TVERSKY_ALPHA * fp + MATTE_TVERSKY_BETA * fn + 1.0)
    ink = (cov > 0.5).float() * w
    err = torch.sigmoid(bold_logit) - bold
    thin_gate = torch.clamp(1.0 - bold / bold_asym_t0, min=0.0)
    pen = err.abs() + bold_asym * torch.relu(err) * thin_gate
    loss_bold = (pen * ink).sum() / ink.sum().clamp_min(1.0)

    conf_bold = (bold >= MARGIN_CONF_BOLD_T).float()
    conf_norm = (bold <= MARGIN_CONF_NORM_T).float()
    hinge = (conf_bold * torch.relu(MARGIN_LOGIT - bold_logit)
             + conf_norm * torch.relu(bold_logit + MARGIN_LOGIT))
    loss_margin = (hinge * ink).sum() / ink.sum().clamp_min(1.0)

    total = (loss_matte + matte_tversky * loss_tversky
             + bold_weight * loss_bold + bold_margin * loss_margin)
    return total, loss_matte, loss_bold, loss_margin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--prefetch", type=int, default=4,
                    help="DataLoader prefetch_factor; lower on low-RAM boxes (each in-flight batch is ~150MB)")
    ap.add_argument("--out", default="ckpt")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--no-compile", action="store_true", help="disable torch.compile (GPU)")
    ap.add_argument("--base", type=int, default=16, help="base channel width (capacity)")
    ap.add_argument("--levels", type=int, default=2, help="U-Net depth (3 = bigger RF)")
    ap.add_argument("--reuse", type=int, default=1, help="composites per rasterized strip")
    ap.add_argument("--val-batch", type=int, default=32, help="validation set size")
    ap.add_argument("--bold-weight", type=float, default=1.5, help="weight on the bold L1 term")
    ap.add_argument("--bold-asym", type=float, default=1.5,
                    help="extra penalty on emboldening thin ink (overshoot on low-target); 0 = plain L1")
    ap.add_argument("--bold-asym-t0", type=float, default=0.15,
                    help="target above which the thin-overshoot penalty fades to 0 (bold left as pure L1)")
    ap.add_argument("--bold-margin", type=float, default=0.0,
                    help="weight on the confident-margin hinge (lifts clearly-bold scores to the rail); 0 = off")
    ap.add_argument("--matte-tversky", type=float, default=0.0,
                    help="weight on the recall-biased matte Tversky term (FN>FP); 0 = plain BCE")
    ap.add_argument("--bold-from", type=int, default=1, help="decoder stage feeding the bold head (1=full,2=½,3=¼)")
    ap.add_argument("--detach-bold", action="store_true", help="stop bold gradient into the trunk (matte-priority)")
    ap.add_argument("--gpu-degrade", action="store_true",
                    help="EXPERIMENTAL: batched degrade on the GPU (faster but degrade@48 ≠ @native; regresses quality)")
    ap.add_argument("--bold-head", choices=["dilated", "1x1"], default="dilated",
                    help="bold head: dilated 3×3 (default) or cheap 1×1")
    args = ap.parse_args()

    # spawn (not fork) workers: plain fork inherits the parent's cv2/fontconfig/freetype +
    # CUDA state, and with a broad font set the workers deterministically hit a poisoned
    # native lock and deadlock a few hundred steps in (single-process never does). spawn
    # gives each worker a clean interpreter — bulletproof, at the cost of re-importing torch
    # per worker (~2GB), so size --workers to RAM (needs a ~32GB box for ~10 workers).
    mp_ctx = "spawn" if args.workers > 0 else None

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = InkUNet(base=args.base, levels=args.levels, bold_from=args.bold_from,
                    detach_bold=args.detach_bold, bold_head=args.bold_head).to(device)
    print(f"device={device} base={args.base} levels={args.levels} bold_from={args.bold_from} "
          f"detach_bold={args.detach_bold} bold_head={args.bold_head} params={param_count(model):,}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_step = 0
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start_step = state["step"]
        print(f"resumed from {args.resume} at step {start_step}")

    # AMP (fp16) and kernel fusion only pay off on the GPU, where this small model is
    # launch-bound: fusing conv/BN/ReLU and halving memory traffic is the lever. `model`
    # stays uncompiled so checkpoints save plain InkUNet weights (no `_orig_mod.` prefix)
    # and stay loadable by eval_real.py / export_onnx.py.
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    train_model = torch.compile(model) if use_amp and not args.no_compile else model
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.steps, last_epoch=start_step - 1
    )

    loader = DataLoader(
        SyntheticStrips(batch=args.batch, width=args.width, reuse=args.reuse,
                        apply_degrade=not args.gpu_degrade),
        batch_size=None,  # the dataset yields whole batches (one IPC transfer each)
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        prefetch_factor=args.prefetch if args.workers > 0 else None,
        # No pin_memory: this job is dataloader-bound (the GPU waits on the CPU), so pinned
        # host buffers buy nothing and just add non-reclaimable RAM pressure on a 15GB box.
        pin_memory=False,
        multiprocessing_context=mp_ctx,
    )
    val_img, val_cov, val_bold = validation_batch(n=args.val_batch, width=args.width)
    val_img, val_cov, val_bold = val_img.to(device), val_cov.to(device), val_bold.to(device)

    # GPU degrade state: a device RNG (augmentation randomness) + the cached DCT matrix.
    degrade_gen = torch.Generator(device=device).manual_seed(0)
    dct = gpu_degrade._dct8(device)

    os.makedirs(args.out, exist_ok=True)
    model.train()
    t0 = time.time()
    running = 0.0
    for step, (img, cov, bold, native_h) in enumerate(loader, start=start_step + 1):
        if step > args.steps:
            break
        img = img.to(device, non_blocking=device == "cuda")
        cov = cov.to(device, non_blocking=device == "cuda")
        bold = bold.to(device, non_blocking=device == "cuda")
        native_h = native_h.to(device, non_blocking=device == "cuda")
        # GPU degrade path (experimental): degrade + legibility batched on the GPU, illegible
        # strips become zero-weight. Default path: strips arrive already CPU-degraded + legible.
        strip_w = None
        if args.gpu_degrade:
            with torch.no_grad():
                img = gpu_degrade.degrade_batch(img, native_h, degrade_gen, dct)
                strip_w = gpu_degrade.legible_mask(img, cov, native_h).float()
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            logits = train_model(img)
            loss, _, _, _ = ink_losses(logits, cov, bold, args.bold_weight, strip_w=strip_w,
                                       bold_asym=args.bold_asym, bold_asym_t0=args.bold_asym_t0,
                                       bold_margin=args.bold_margin, matte_tversky=args.matte_tversky)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
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
            with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                vl, vm, vb, vg = ink_losses(train_model(val_img), val_cov, val_bold, args.bold_weight,
                                            bold_asym=args.bold_asym, bold_asym_t0=args.bold_asym_t0,
                                            bold_margin=args.bold_margin, matte_tversky=args.matte_tversky)
            model.train()
            print(
                f"step {step:>6}  VAL loss {vl.item():.4f} (matte {vm.item():.4f} bold {vb.item():.4f} margin {vg.item():.4f})",
                flush=True,
            )
            state = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                     "base": args.base, "levels": args.levels, "bold_from": args.bold_from,
                     "bold_head": args.bold_head}
            torch.save(state, os.path.join(args.out, "ink-latest.pt"))
    print("done")


if __name__ == "__main__":
    main()
