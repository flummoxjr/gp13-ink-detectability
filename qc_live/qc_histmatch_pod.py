"""Minimal histmatch probe (pod-side). Single 1203 tile: baseline vs
histogram-matched-to-Paris4, plus a Paris4 control tile. Prints JSON lines."""
import json
import sys

import numpy as np

sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/villa/vesuvius/src")
import torch  # noqa: E402
import zarr  # noqa: E402
import fsspec  # noqa: E402
from screen_band import load_ink3d, normalize, CKPT  # noqa: E402

V1203 = "vesuvius-challenge-open-data/PHerc1203/volumes/20260319130212-2.403um-0.2m-77keV-masked.zarr"
P4 = "vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr"
T1203 = (7424, 7168, 11776)   # inside CLI smoke region
TP4 = (21700, 8347, 10941)    # released-pred-active Paris4 region
TILE = 256

fs = fsspec.filesystem("s3", anon=True)
v1203 = zarr.open(fs.get_mapper(V1203), mode="r")["0"]
vp4 = zarr.open(fs.get_mapper(P4), mode="r")["0"]

print("fetching tiles...", flush=True)
ct = np.asarray(v1203[T1203[0]:T1203[0] + TILE, T1203[1]:T1203[1] + TILE, T1203[2]:T1203[2] + TILE])
p4 = np.asarray(vp4[TP4[0]:TP4[0] + TILE, TP4[1]:TP4[1] + TILE, TP4[2]:TP4[2] + TILE])
print("fetched", ct.shape, p4.shape, flush=True)

model = load_ink3d(CKPT).cuda().eval().half()


def run(nrm):
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


def stats(p):
    return {"pmax": round(float(p.max()), 4), "pmean": round(float(p.mean()), 5),
            "f05": round(float((p > 0.5).mean()), 5), "f08": round(float((p > 0.8).mean()), 5)}


def save_l2(p, name):
    t = torch.from_numpy(p)[None, None]
    pl2 = torch.nn.functional.avg_pool3d(t, 4)[0, 0].numpy()
    np.save(f"/workspace/out/{name}.npy", (pl2 * 255).astype(np.uint8))


res = {}
res["p4_control"] = stats(run(normalize(p4)))
print("p4_control", json.dumps(res["p4_control"]), flush=True)

prob = run(normalize(ct))
res["t1203_baseline"] = stats(prob)
save_l2(prob, "qc_hm_base")
print("t1203_baseline", json.dumps(res["t1203_baseline"]), flush=True)

# histogram match: nonzero voxels of 1203 -> nonzero distribution of P4 tile
nz = ct > 0
src = ct[nz].astype(np.float32)
tgt_sorted = np.sort(p4[p4 > 0].astype(np.float32))
tgt_q = np.quantile(tgt_sorted, np.linspace(0, 1, 4096))
src_sorted = np.sort(src)
q = np.searchsorted(src_sorted, src, side="left") / max(len(src) - 1, 1)
matched = ct.astype(np.float32).copy()
matched[nz] = np.interp(q, np.linspace(0, 1, 4096), tgt_q).astype(np.float32)
res["t1203_intensity"] = {
    "src_p": [float(v) for v in np.percentile(src, [1, 50, 99])],
    "tgt_p": [float(v) for v in np.percentile(tgt_sorted, [1, 50, 99])],
    "matched_p": [float(v) for v in np.percentile(matched[nz], [1, 50, 99])]}
print("intensity", json.dumps(res["t1203_intensity"]), flush=True)

prob_hm = run(normalize(matched))
res["t1203_histmatched"] = stats(prob_hm)
save_l2(prob_hm, "qc_hm_matched")
print("t1203_histmatched", json.dumps(res["t1203_histmatched"]), flush=True)

with open("/workspace/out/qc_histmatch_result.json", "w") as f:
    json.dump(res, f, indent=1)
print("DONE", flush=True)
