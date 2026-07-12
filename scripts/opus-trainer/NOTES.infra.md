---
name: reference_distillation_infra
description: "Compute + paths for the OPUS-MT->slimt distillation pipeline (Phase 1 CPU box, Phase 2/3 vast.ai GPU)"
metadata: 
  node_type: memory
  type: reference
  originSessionId: cfcd278d-eac4-4fc8-ba04-c981e1ad3340
---

Infra for the [[project_opus_mt_distillation_teachers]] pipeline (scripts in `~/git/translator-rs/scripts/opus-trainer/`):

**Phase 1 (data prep, CPU-only):** host `bigserver` = `david@192.168.2.10`. Old box (Debian, python3.9, no GPU), but native venv works fine — no docker needed for Phase 1. venv at `/fast_storage/david/opus-trainer/venv` (opustools, sentencepiece, fasttext-wheel, **numpy<2** — fasttext-wheel breaks on numpy 2.x). langid model `lid.176.ftz` in same dir. Scratch/workdir on `/nvme2/prom/opus-trainer` (`/nvme2` is root-owned with 434GB free; `/nvme2/prom` is the david-writable subdir). `/fast_storage` has ~84GB free as fallback.

ACTIVE (2026-07-11, Phase 3 full-send): vast **44548433** (box A, RTX 4090, nvidia/cuda:11.8.0-devel) — **TRAINING en→tl** (marian built @ /root/marian-dev/build/marian 755MB; full 10M train.tsv + FLORES-dev valid.entl.tsv in /root/opus; train_entl.log, model_entl/; monitor bjxaqfedn). GPU smoke-train already validated marian accepts the whitespace guided-alignments + trains ~125k w/s. gotchas: (a) build `make marian_train` NOT `make marian` (latter=libmarian.a); (b) valid-sets must be a single 3-col TSV when train is TSV (empty align col ok); (c) box A SSH is flaky — use rsync --partial for big files, nohup for jobs, ServerAliveInterval. tl→en train.tsv + both lex.s2t ready on bigserver /nvme2/prom/train-full/.
en→tl confirmed HEALTHY (Cost 5.0→1.52 by Up.2000, 140k w/s) → launched **box B2 = vast 44552855** (RTX 4090, cuda:11.8-devel, Australia; ssh 144.6.107.170:17850) for tl→en: marian building (monitor bto3et1ws), tlen train.tsv (10M) + valid.tlen.tsv staged in /root/opus. (First box B 44552153 was stuck in "created" → destroyed; use reliability>0.98 offers.) **Destroy when done:** `vastai destroy instance 44552855`. BOTH DIRECTIONS TRAINING (2026-07-12 ~23:20): en→tl on box A 44548433 (model_entl/, first valid ce 4.36 new-best, converging), tl→en on box B2 44552855 (model_tlen/, launched with 2-col valid.tlen2.tsv). Both: base-memory+guided-align, 10M, joint vocab.entl.spm, GPU, early-stopping. Monitors: en→tl completion bp02q7puc, tl→en bvvycd0ka. When each fires "Training finished" → pull model.npz → browsermt quantize+shortlist on bigserver → benchmark vs 59.4/58.5 → index_v5. Destroy each box after its model is pulled. key `opus-staging/pilot_key`. **Destroy when done:** `opus-staging/vastenv/bin/vastai destroy instance 44548433`. Alignments (fast_align 10M both dirs) running on bigserver `/nvme2/prom/train-full/{entl,tlen}/` (monitor b86nfow4y). Staggered plan: box A en→tl first, then a box B for tl→en.

Phase 2 COMPLETE (2026-07-11) — both Phase-2 boxes destroyed. KD data staged in `/home/david/AndroidStudioProjects/opus-staging/`: `kd.en2tl.tl.gz` + `kd.en.zst` (en→tl), `kd.tl2en.en.gz` + `kd.tl.zst` (tl→en), `vocab.entl.model`. Throwaway `pilot_key` there for future vast boxes (vast `chorus` key is passphrase-locked + agent empty; `vastenv/bin/vastai` is the CLI). vast env pins for the pytorch image: transformers==4.44.2, ctranslate2==4.4.0.
Lessons: (1) SSH via attached throwaway key, retry a few times for propagation. (2) distill_data.py must truncate source to 512 tokens (opus-mt max positions) or CT2 crashes mid-run. (3) completion monitors: watch a marker that stays the LAST log line, or tail scrolls past it.

vast env gotchas: pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime image needs `transformers==4.44.2` + `ctranslate2==4.4.0` (CT2 4.8 converter passes a `dtype` kwarg only transformers 5.x accepts, which needs torch>=2.4; the pin avoids it).

**Phase 2/3 (GPU, forward-translation + student training):** rent on **vast.ai — credits are preloaded**. The old CPU box can't run modern marian/CUDA (old glibc), so use a CUDA docker image on the rental. GOTCHA: heed the vast.ai MOTD about **disabling tmux** — if left on it eats all stdout/stderr. Run jobs with tmux disabled (or nohup + logfile), checkpoint to cloud, auto-terminate.
