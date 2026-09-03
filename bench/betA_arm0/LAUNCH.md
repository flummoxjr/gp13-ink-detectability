# Bet A arm 0 — launch notes (2026-09-03)

Build: `python build.py` (assembles `pod_betA_arm0.sh` from `parts/` + the p2a_v3 machinery + the
vendored generator/configs + `data_manifest.json`; runs bash -n, py_compile, JSON parse, DRY walk).
Offline tests (`scratchpad/betA_test.py`): F1 sweep, sparse-plan geometry, config generation vs a fake
layout, finalize gate paths, and a real HF sync of pherc0814-46527 — all passed 2026-09-03.

## 1. Smoke launch (first; ~40–70 min, ~$0.5–0.9)

```
python trackD/bench/tools/launch_pod.py --name betA0smoke --gist <pinned raw url> --script pod_betA_arm0.sh \
  --out C:/Users/benbl/Desktop/Vsuvious/experiments/betA0smoke --deadline-hours 3 --no-status-min 20 \
  --disk 120 --min-vcpu 12 --min-ram 48 --fetch-files results.json,bundle.tgz,ckpts_keep.tgz \
  --gpus "NVIDIA GeForce RTX 5090/COMMUNITY,NVIDIA GeForce RTX 5090/SECURE,NVIDIA GeForce RTX 4090/COMMUNITY"
```
Env defaults in the script: `SMOKE_ONLY=1` → seed 42 only, 2,000 iterations, save every 1,000; the
2,000-step checkpoint and the released checkpoint are scored on the five native crops. Expected status
sequence: provision (~2 min) → ckpt → trainer_check (0814 labels, 30 synthetic iterations: the first real
test of the trainer contract) → labels_fetch (~32k files; 5–15 min) → sv_fetch (16.5k chunks, 29 GB;
15–40 min at 32 threads) → pool (15 volumes; 5–15 min) → config_gen → native_fetch → ctl (released ckpt:
native ≥ 0.95 / reverse ≤ 0.80 or the run dies) → train_s42 (2,000 steps; 5–10 min) → eval_s42 → ref →
finalize (bundle.tgz, ckpts_keep.tgz) → ALL DONE.

## 2. What to check before the full run

- `results/results.json → training.42.it_per_s_last` (khj1222: 7–8 it/s on a 5090; below ~4 it/s the
  dataloader is starved → lower `dataloader_workers`, raise vCPU).
- `eval.s42_2000` vs `eval.ref_released` on the five crops: the 2k-step model should already beat the
  floor on some crops; the released in-scroll model should read ~0.999 AUC on w035.
- `ctl` block: native forward ≥ 0.95, reverse ≤ 0.80 (harness certified).
- `sv_plan.json` counts vs the manifest; `labels_*.json` plane counts; pooled-volume gates in status.

## 3. Full run (~8–11.5 h, ~$5.5–8)

Same command with `--deadline-hours 14 --name betA0` and the gist re-pinned after any script change;
set `SMOKE_ONLY=0` by prefixing the boot (`--pre "export SMOKE_ONLY=0;"`). The prereg gate
(`PREREG_BET_A_DRAFT.md` §4) is evaluated by `finalize.py` only when both seeds complete.
