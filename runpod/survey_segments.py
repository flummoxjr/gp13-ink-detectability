"""Corpus survey: render + ink_9um infer every published GP-scroll segment.

Pod-side. Usage: python survey_segments.py <shard_id> <n_shards>
Per segment: fetch tifxyz -> render 21-slice SV -> infer seed42 fwd+rev ->
compact stats + tripwire + fwd/rev map correlation -> 4x-downsampled prediction
PNG/npy kept, big intermediates deleted. Appends JSONL so partial work survives.

Heavy analysis (full 4-test battery) runs later on the laptop over the saved maps.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

CATALOG = "/workspace/segment_catalog.json"
OUT = "/workspace/survey"
CKPT = "/workspace/ckpts/ink_9um/hybrid_3d2d-seed42/step-075000.pth"
BUCKET_S3 = "s3://vesuvius-challenge-open-data"
BLANK_P99 = 195  # control blank-papyrus p99 (DN) — tripwire value gate

os.makedirs(OUT, exist_ok=True)
shard, nshards = int(sys.argv[1]), int(sys.argv[2])
cat = json.load(open(CATALOG))
mine = [c for i, c in enumerate(cat) if i % nshards == shard]
done_path = os.path.join(OUT, f"survey_{shard}.jsonl")
done = set()
if os.path.exists(done_path):
    for line in open(done_path):
        try:
            done.add(json.loads(line)["name"])
        except Exception:
            pass
print(f"shard {shard}/{nshards}: {len(mine)} segments, {len(done)} already done", flush=True)


def run(cmd, timeout=3600):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)


def stats(pred):
    """Compact battery-relevant stats on a prediction map."""
    m = pred > 0
    if m.sum() < 1000:
        return None
    v = pred[m]
    out = {
        "canvas": list(pred.shape), "nonzero_frac": round(float(m.mean()), 4),
        "max": int(pred.max()), "p50": float(np.percentile(v, 50)),
        "p99": float(np.percentile(v, 99)),
        "frac_gt_half": round(float((pred > 0.5 * max(pred.max(), 1)).mean()), 4),
        "frac_gt_blankp99": round(float((pred > BLANK_P99).mean()), 6),
    }
    # tripwire: components above the control blank p99 with letter-scale area/width
    try:
        from scipy import ndimage
        lab, n = ndimage.label(pred > BLANK_P99)
        trips = []
        if n:
            sizes = ndimage.sum_labels(np.ones_like(lab, np.float32), lab, range(1, n + 1))
            big = [int(i) + 1 for i in np.argsort(sizes)[::-1][:12] if sizes[int(i)] >= 1e4]
            for cid in big:
                sel = lab == cid
                dist = ndimage.distance_transform_edt(sel)
                width = float(dist.max() * 2)
                if width >= 30:
                    ys, xs = np.nonzero(sel)
                    trips.append({"area": int(sel.sum()), "width": round(width, 1),
                                  "centroid": [int(ys.mean()), int(xs.mean())]})
        out["tripwire_hits"] = trips
        out["n_hot_components"] = int(n)
    except Exception as e:
        out["tripwire_error"] = str(e)[:100]
    return out


import tifffile

for c in mine:
    if c["name"] in done:
        continue
    t0 = time.time()
    name, scroll = c["name"], c["scroll"]
    rec = {"name": name, "scroll": scroll, "seg_dir": c["seg_dir"]}
    mesh = f"/workspace/data/{name}.tifxyz"
    sv = f"/workspace/data/{name}.sv.zarr"
    try:
        os.makedirs(mesh, exist_ok=True)
        import s3fs
        fs = s3fs.S3FileSystem(anon=True)
        for f in ("x.tif", "y.tif", "z.tif", "meta.json"):
            src = c["tifxyz"].replace(f"{BUCKET_S3}/", "") + f
            try:
                fs.get(f"vesuvius-challenge-open-data/{c['tifxyz']}{f}", f"{mesh}/{f}")
            except Exception:
                if f != "meta.json":
                    raise
        r = run(f'cd /workspace/villa/vesuvius && PATH=$HOME/.local/bin:$PATH uv run --no-sync '
                f'--extra models python /workspace/scripts/render_tifxyz_sv.py {mesh} '
                f'"{BUCKET_S3}/{scroll}/volumes/{c["volume"]}" {sv} --num-slices 21', timeout=2400)
        if not os.path.exists(sv):
            rec["error"] = "render_failed: " + (r.stderr or r.stdout)[-300:]
            raise RuntimeError("render")
        preds = {}
        for direction in ("forward", "reverse"):
            tif = f"/workspace/survey/{name}_{direction}.tif"
            dflag = "--direction forward" if direction == "forward" else "--direction reverse"
            rr = run(f'cd /workspace/villa/vesuvius && PATH=$HOME/.local/bin:$PATH uv run --no-sync '
                     f'--extra models python -m vesuvius.ink_detection.inference.infer {sv} {CKPT} '
                     f'{tif} {dflag} --batch-size 16 --num-workers 8 --gpus 0', timeout=2400)
            if os.path.exists(tif):
                preds[direction] = tifffile.imread(tif).astype(np.float32)
                np.save(f"/workspace/survey/{name}_{direction}_ds4.npy",
                        preds[direction][::4, ::4].astype(np.uint8))
                os.remove(tif)
            else:
                rec[f"infer_{direction}_error"] = (rr.stderr or rr.stdout)[-200:]
        if "forward" in preds:
            rec["forward"] = stats(preds["forward"])
        if "reverse" in preds:
            rec["reverse"] = stats(preds["reverse"])
        if len(preds) == 2:
            a, b = preds["forward"], preds["reverse"]
            h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
            a, b = a[:h, :w], b[:h, :w]
            mm = (a > 0) & (b > 0)
            if mm.sum() > 1000:
                rec["fwd_rev_r"] = round(float(np.corrcoef(a[mm], b[mm])[0, 1]), 4)
    except Exception as e:
        rec.setdefault("error", str(e)[:300])
    finally:
        subprocess.run(f"rm -rf {sv} {mesh}", shell=True)
    rec["secs"] = round(time.time() - t0, 1)
    with open(done_path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    trips = (rec.get("forward") or {}).get("tripwire_hits") or []
    print(f"{name}: {rec.get('secs')}s fwd_rev_r={rec.get('fwd_rev_r')} "
          f"TRIPWIRE={len(trips)} {'ERR ' + rec['error'][:60] if 'error' in rec else ''}", flush=True)

print(f"SHARD {shard} COMPLETE", flush=True)
