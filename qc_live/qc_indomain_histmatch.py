"""QC test 5: in-domain positive control + histogram-matching salvage probe.

A) Run the screen pipeline (same ckpt, same normalize) on Paris4 CT tiles at
   locations where the RELEASED Paris4 ink3d pred is (i) active and (ii) silent.
   If our output matches the released pred (sparse, quiet where quiet), the
   pipeline+model are healthy in-domain -> 1203 blanket firing is domain shift.
B) Histogram-match a 1203 tile's nonzero voxels to the Paris4 nonzero CT
   distribution, re-run inference: does blanket firing collapse?
"""
import json
import sys

import numpy as np

STUBS = r"C:\Users\benbl\AppData\Local\Temp\claude\C--Users-benbl-Desktop-Vsuvious\b3441997-0118-49b2-8364-cbdf28fc6397\scratchpad\stubs"
VILLA = r"C:\Users\benbl\Desktop\Vsuvious\villa\vesuvius\src"
sys.path.insert(0, STUBS)
sys.path.insert(0, VILLA)

import torch  # noqa: E402
import zarr  # noqa: E402
import fsspec  # noqa: E402

sys.path.insert(0, r"C:\Users\benbl\Desktop\Vsuvious\trackD\qc_live")
from qc_norm_equiv import load_ink3d, norm_screen, stats, CKPT  # noqa: E402

P4_VOL = "vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr"
P4_PRED = ("vesuvius-challenge-open-data/PHercParis4/representations/predictions/ink-3d/"
           "20260411134726-ink3d-20260428123845-v3-78k-fullsup.zarr")
V1203 = "vesuvius-challenge-open-data/PHerc1203/volumes/20260319130212-2.403um-0.2m-77keV-masked.zarr"
OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\qc_live\qc_indomain_histmatch_result.json"
TILE = 256

fs = fsspec.filesystem("s3", anon=True)
p4ct = zarr.open(fs.get_mapper(P4_VOL), mode="r")["0"]
p4pred = zarr.open(fs.get_mapper(P4_PRED), mode="r")
try:
    p4pred = p4pred["0"]
except (KeyError, TypeError):
    pass
ct1203 = zarr.open(fs.get_mapper(V1203), mode="r")["0"]

# reproduce qc_paris4.py sample coords (same rng stream) to reuse its findings
rng = np.random.default_rng(0)
sh = p4pred.shape
coords = []
for _ in range(8):
    coords.append([int(rng.integers(s // 4, 3 * s // 4)) for s in sh])
# block idx 2 had f05=0.0817 (active), idx 0 had f05=0.0 pmax=0.016 (silent)
active_c, silent_c = coords[2], coords[0]
print("active block", active_c, "silent block", silent_c, flush=True)

model = load_ink3d(CKPT).cuda().eval().half()
res = {}


def run_model(nrm):
    with torch.no_grad():
        x = torch.from_numpy(nrm).half().cuda()[None, None]
        logits = model(x)
        if isinstance(logits, dict):
            logits = next(iter(logits.values()))
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        prob = torch.sigmoid(logits.float())[0, 0].cpu().numpy()
        del x, logits
        torch.cuda.empty_cache()
    return prob


p4_nonzero_pool = []
for name, c in (("p4_active", active_c), ("p4_silent", silent_c)):
    # center a 256^3 tile on the sampled 64^3 block
    o = [max(0, cc - 96) for cc in c]
    ct = np.asarray(p4ct[o[0]:o[0] + TILE, o[1]:o[1] + TILE, o[2]:o[2] + TILE])
    pred = np.asarray(p4pred[o[0]:o[0] + TILE, o[1]:o[1] + TILE, o[2]:o[2] + TILE]).astype(np.float32) / 255.0
    row = {"origin_zyx": o, "ct_nonzero_frac": round(float((ct > 0).mean()), 4),
           "ct_nonzero_mean": round(float(ct[ct > 0].mean()), 2) if (ct > 0).any() else None,
           "released_pred": stats(pred)}
    nrm = norm_screen(ct)
    if nrm is None:
        row["screen"] = "degenerate"
    else:
        prob = run_model(nrm)
        row["screen"] = stats(prob)
        row["corr_vs_released"] = round(float(np.corrcoef(prob.ravel()[::8], pred.ravel()[::8])[0, 1]), 4)
    if (ct > 0).any():
        p4_nonzero_pool.append(ct[ct > 0][::8])
    res[name] = row
    print(name, json.dumps(row), flush=True)

# --- B: histogram matching on a 1203 tile ---
t = (7424, 7168, 11776)
ct = np.asarray(ct1203[t[0]:t[0] + TILE, t[1]:t[1] + TILE, t[2]:t[2] + TILE])
p4pool = np.concatenate(p4_nonzero_pool)
res["intensity_shift"] = {
    "p4_nonzero_mean_std": [round(float(p4pool.mean()), 2), round(float(p4pool.std()), 2)],
    "t1203_nonzero_mean_std": [round(float(ct[ct > 0].mean()), 2), round(float(ct[ct > 0].std()), 2)],
    "p4_pcts": [float(v) for v in np.percentile(p4pool, [1, 25, 50, 75, 99])],
    "t1203_pcts": [float(v) for v in np.percentile(ct[ct > 0], [1, 25, 50, 75, 99])]}
print("intensity", json.dumps(res["intensity_shift"]), flush=True)

# baseline (screen norm) for this tile
prob = run_model(norm_screen(ct))
res["t1203_baseline"] = stats(prob)

# histogram match nonzero voxels of 1203 tile -> p4 pool distribution, keep zeros
nz = ct > 0
src = ct[nz].astype(np.float64)
src_sorted = np.sort(src)
ranks = np.searchsorted(src_sorted, src, side="left") / max(len(src) - 1, 1)
tgt = np.quantile(p4pool.astype(np.float64), np.clip(ranks, 0, 1))
matched = ct.astype(np.float32).copy()
matched[nz] = tgt.astype(np.float32)
prob_hm = run_model(norm_screen(matched))
res["t1203_histmatched"] = stats(prob_hm)
print("baseline", res["t1203_baseline"], flush=True)
print("histmatched", res["t1203_histmatched"], flush=True)

with open(OUT, "w") as f:
    json.dump(res, f, indent=1)
print("WROTE", OUT)
