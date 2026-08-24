"""Streaming ink screen for PHerc1203's 2.4um band (pod-side).

Loads ink_3d_dino_guided once, iterates non-overlapping 256^3 tiles across a
z-range shard of the band (skipping tiles outside the scroll mask), and stores
only reduced artifacts:
  - per-tile stats (max prob, frac>0.5, frac>0.8, mask fill) -> stats.jsonl
  - 4x-downsampled probability (uint8) per tile -> prob_L2 zarr (local)
Usage:
  python screen_band.py <z0> <z1> <out_dir>              # explicit z-range
  python screen_band.py worker <id> <n_workers> <out_dir> # interleaved z-tiles
"""
import json
import os
import sys

import numpy as np
import torch
import zarr
import fsspec

VOL_URL = "s3://vesuvius-challenge-open-data/PHerc1203/volumes/20260319130212-2.403um-0.2m-77keV-masked.zarr"
CKPT = "/workspace/models/ink3d/ckpt_78k_fullsup.pth"
TILE = 256
SHAPE = (15137, 26493, 26493)

sys.path.insert(0, "/workspace/villa/vesuvius/src")
from vesuvius.models.run.inference import (_normalize_train_py_model_config,
                                           _select_train_py_state_dict)
from vesuvius.models.build.build_network_from_config import NetworkFromConfig


def load_ink3d(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mc = _normalize_train_py_model_config(ck)

    class Mgr:
        model_config = mc
        targets = mc.get("targets", {})
        train_patch_size = mc.get("train_patch_size", mc.get("patch_size", (256, 256, 256)))
        train_batch_size = mc.get("train_batch_size", mc.get("batch_size", 1))
        in_channels = mc.get("in_channels", 1)
        autoconfigure = mc.get("autoconfigure", False)
        enable_deep_supervision = bool(mc.get("enable_deep_supervision", False))
        model_name = mc.get("model_name", "Model")
        spacing = [1, 1, 1]

    model = NetworkFromConfig(Mgr())
    sd, src = _select_train_py_state_dict(ck)
    prefixes = ("module.", "_orig_mod.")

    def strip(k):
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if k.startswith(p):
                    k = k[len(p):]
                    changed = True
        return k

    model.load_state_dict({strip(k): v for k, v in sd.items()}, strict=True)
    print("loaded state dict from", src, flush=True)
    return model


def normalize(x):
    """percentile min-max, per model card / config."""
    lo, hi = np.percentile(x[x > 0], [1, 99]) if (x > 0).any() else (0, 1)
    if hi <= lo:
        return None
    y = np.clip((x.astype(np.float32) - lo) / (hi - lo), 0, 1)
    return y


def main():
    if sys.argv[1] == "worker":
        wid, nw = int(sys.argv[2]), int(sys.argv[3])
        out_dir = sys.argv[4]
        z_tiles = [i for i in range(SHAPE[0] // TILE) if i % nw == wid]
        z0, z1 = 0, SHAPE[0]  # bookkeeping only
    else:
        z0, z1 = int(sys.argv[1]), int(sys.argv[2])
        out_dir = sys.argv[3]
        z_tiles = list(range(z0 // TILE, min(z1, SHAPE[0]) // TILE))
        wid, nw = 0, 1
    os.makedirs(out_dir, exist_ok=True)

    model = load_ink3d(CKPT).cuda().eval().half()

    fs = fsspec.filesystem("s3", anon=True)
    store = zarr.open(fs.get_mapper(VOL_URL.replace("s3://", "")), mode="r")["0"]

    # full L4 mask (2.6GB u8 -> bool) for tile skipping; 256^3 tile = 16^3 L4 block
    l4 = zarr.open(fs.get_mapper(VOL_URL.replace("s3://", "")), mode="r")["4"]
    m4 = np.asarray(l4[:]) > 0
    print("L4 mask loaded", m4.shape, flush=True)

    stats_path = os.path.join(out_dir, f"stats_{z0}_{z1}.jsonl")
    done = set()
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            done = {tuple(json.loads(l)["tile"]) for l in f if l.strip()}
    fstats = open(stats_path, "a")

    probL2_dir = os.path.join(out_dir, f"probL2_{z0}_{z1}")
    os.makedirs(probL2_dir, exist_ok=True)

    n_y = SHAPE[1] // TILE
    n_x = SHAPE[2] // TILE
    total, run, skip = 0, 0, 0
    import time
    t0 = time.time()

    # build the run-list first (mask-gated at L4), then prefetch tile data
    jobs = []
    for iz in z_tiles:
        for iy in range(n_y):
            for ix in range(n_x):
                total += 1
                key = (iz * TILE, iy * TILE, ix * TILE)
                if key in done:
                    continue
                b = 16
                fill = m4[iz * b:(iz + 1) * b, iy * b:(iy + 1) * b, ix * b:(ix + 1) * b].mean()
                if fill < 0.05:
                    skip += 1
                    fstats.write(json.dumps({"tile": key, "fill": round(float(fill), 3), "skipped": True}) + "\n")
                    continue
                jobs.append((key, float(fill)))
    fstats.flush()
    print(f"run-list: {len(jobs)} tiles ({skip} mask-skipped of {total})", flush=True)

    def fetch(job):
        key, fill = job
        a = np.asarray(store[key[0]:key[0] + TILE, key[1]:key[1] + TILE, key[2]:key[2] + TILE])
        return key, fill, a

    # bounded prefetch: at most 8 tiles in flight so RAM stays flat
    from concurrent.futures import ThreadPoolExecutor
    from collections import deque
    ex = ThreadPoolExecutor(6)
    inflight = deque()
    ji = 0

    def results():
        nonlocal ji
        while ji < len(jobs) or inflight:
            while ji < len(jobs) and len(inflight) < 8:
                inflight.append(ex.submit(fetch, jobs[ji]))
                ji += 1
            yield inflight.popleft().result()

    with torch.no_grad():
        for key, fill, a in results():
                    norm = normalize(a)
                    if norm is None:
                        skip += 1
                        continue
                    x = torch.from_numpy(norm).half().cuda()[None, None]
                    logits = model(x)
                    if isinstance(logits, dict):
                        logits = next(iter(logits.values()))
                    if isinstance(logits, (list, tuple)):
                        logits = logits[0]
                    prob = torch.sigmoid(logits.float())[0, 0]
                    pmax = float(prob.max())
                    f05 = float((prob > 0.5).float().mean())
                    f08 = float((prob > 0.8).float().mean())
                    pL2 = torch.nn.functional.avg_pool3d(prob[None, None], 4)[0, 0]
                    np.save(os.path.join(probL2_dir, f"{key[0]}_{key[1]}_{key[2]}.npy"),
                            (pL2.cpu().numpy() * 255).astype(np.uint8))
                    fstats.write(json.dumps({"tile": key, "fill": round(float(fill), 3),
                                             "pmax": round(pmax, 3), "f05": round(f05, 5),
                                             "f08": round(f08, 5)}) + "\n")
                    fstats.flush()
                    run += 1
                    if run % 50 == 0:
                        el = time.time() - t0
                        print(f"[{el:.0f}s] run={run} skip={skip} total={total} "
                              f"rate={run / el:.2f} tiles/s", flush=True)
    print(f"DONE shard {z0}:{z1} run={run} skip={skip}", flush=True)


if __name__ == "__main__":
    main()
