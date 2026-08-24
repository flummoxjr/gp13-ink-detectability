"""K2c — sheet-separability axis for the Grand-Prize scrolls.

WHY THIS EXISTS
---------------
The K2b index (`k2b_detectability_index.py`) measures *scan quality*: how far a
volume's structure stands above its own noise. It does not measure whether the
papyrus sheets in that volume can be told apart, and those are different axes.
PHerc0813 ranks #1 of 14 on K2b structural SNR, yet the first surfaces ever grown
there show lamella modulation 0.037-0.073 against the control's 0.443.

TWO THINGS THIS SCRIPT FIXES
----------------------------
1. THE STATISTIC. Sheet separability is measured as the median, over 32^3 blocks,
   of the structure-tensor planarity (lam1 - lam2)/(lam1 + lam2) of the local
   image gradient. Laminated papyrus has gradients aligned to one axis (the sheet
   normal) and scores high; granular incrustation scores low however bright it is.
   The statistic is a normalized eigenvalue ratio, so it is orthogonal by
   construction to overall contrast -- i.e. it cannot merely restate K2b's SNR.

2. THE SAMPLING FRAME. K2b's `pick_rois` scores candidates by
   `np.where(fill > 0.98, inten, 0)` and takes the highest -- it samples each
   scroll's BRIGHTEST dense material, which is preferentially mineral incrustation
   rather than papyrus. This script keeps K2b's fill gate and central-z band
   verbatim and changes exactly one thing: candidates are drawn UNIFORMLY AT
   RANDOM instead of by intensity. Running both frames on the same volumes
   measures the bias directly.

Outputs: trackD/out/k2c_separability/<scroll>.json + k2c_summary.json.
Cubes cached to D:\\vesuvius-data\\trackD\\k2c so re-runs are free.
"""
import json
import os
import sys
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy import ndimage as ndi

BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
META_DIR = r"C:\Users\benbl\Desktop\Vsuvious\trackD\meta"
OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\out\k2c_separability"
CACHE = r"D:\vesuvius-data\trackD\k2c"
ROI = 256
N_RANDOM = 24          # ROIs per scroll; the shipped Section 1.8 sampling. The original
                       # n=12 pass is a deterministic prefix of these draws (fixed seed,
                       # deterministic shuffle), so raising n only appends.
MIN_SEP_L3 = 32        # level-3 voxels between ROI centres (= one full ROI at L0)
BLOCK = 32             # structure-tensor block edge, voxels (~0.3 mm)
SIGMA = 1.0            # pre-smoothing before the gradient
FILL_GATE = 0.98       # identical to K2b
SEED = 20260818

SCROLLS = [
    ("PHerc0125", "20250821151825"), ("PHerc0191", "20250821151635"),
    ("PHerc0211", "20250821151803"), ("PHerc0257", "20250821151750"),
    ("PHerc0268", "20251110183117"), ("PHerc0358", "20250821151737"),
    ("PHerc0800", "20250521135224"), ("PHerc0813", "20250821151723"),
    ("PHerc0826", "20250821151701"), ("PHerc1218", "20250521120456"),
    ("PHerc1447", "20250521151220"), ("PHerc1545", "20250821151648"),
    ("PHerc1203", "20250820131727"), ("PHerc0139", "20250728140407"),
]
VOX_UM = {"PHerc0268": 8.640, "PHerc0800": 8.640, "PHerc1218": 8.640, "PHerc1447": 8.640}


def vol_info(sample, vid):
    with open(os.path.join(META_DIR, f"{sample}.json"), encoding="utf-8") as f:
        s = json.load(f)
    v = s["volumes"][vid]
    return v["long_id"], v["properties"]["shape"]


def open_level(sample, long_id, level):
    import zarr, fsspec
    return zarr.open(fsspec.get_mapper(f"{BUCKET}/{sample}/volumes/{long_id}"), mode="r")[str(level)]


def probe_level(sample, long_id):
    """Level 3, or 4 when the central slab would be too big — K2b's rule verbatim."""
    zl = open_level(sample, long_id, 3)
    level = 3
    zlo, zhi = zl.shape[0] // 6, 5 * zl.shape[0] // 6
    if (zhi - zlo) * zl.shape[1] * zl.shape[2] > 8e8:
        zl = open_level(sample, long_id, 4)
        level = 4
        zlo, zhi = zl.shape[0] // 6, 5 * zl.shape[0] // 6
    return zl, level, zlo, zhi


def pick_random_rois(sample, long_id, shape0, n, seed):
    """K2b's frame (central z band, 32-window fill > 0.98) sampled UNIFORMLY.

    Returns (origins, n_candidates, level) so the sampling frame is reportable.
    """
    from scipy.ndimage import uniform_filter
    z3, level, zlo, zhi = probe_level(sample, long_id)
    arr = np.asarray(z3[zlo:zhi])
    fill = uniform_filter((arr > 0).astype(np.float32), size=32, mode="constant")
    ok = np.argwhere(fill > FILL_GATE)
    if len(ok) == 0:
        return [], 0, level
    rng = np.random.default_rng(seed)
    rng.shuffle(ok)
    chosen = []
    for p in ok:
        if all(np.abs(p - c).max() >= MIN_SEP_L3 for c in chosen):
            chosen.append(p)
            if len(chosen) >= n:
                break
    scale = 2 ** level
    out = []
    for z, y, x in chosen:
        out.append((int(np.clip((z + zlo) * scale - ROI // 2, 0, shape0[0] - ROI)),
                    int(np.clip(y * scale - ROI // 2, 0, shape0[1] - ROI)),
                    int(np.clip(x * scale - ROI // 2, 0, shape0[2] - ROI))))
    return out, int(len(ok)), level


def coherence_stats(a, block=BLOCK, sigma=SIGMA):
    """Median/spread of local structure-tensor planarity over in-material blocks."""
    n = a.shape[0]
    cs, normals = [], []
    for z in range(0, n - block + 1, block):
        for y in range(0, n - block + 1, block):
            for x in range(0, n - block + 1, block):
                bl = a[z:z + block, y:y + block, x:x + block]
                if (bl > 0).mean() < 0.98:
                    continue
                v = ndi.gaussian_filter(bl.astype(np.float32), sigma)
                g = np.gradient(v)
                J = np.array([[float((g[i] * g[j]).mean()) for j in range(3)] for i in range(3)])
                w, V = np.linalg.eigh(J)
                cs.append(float((w[2] - w[1]) / max(w[2] + w[1], 1e-9)))
                normals.append(V[:, 2])
    if len(cs) < 8:
        return None
    C = np.array(cs)
    N = np.array(normals)
    wgt = C / C.sum()
    T = np.einsum('i,ij,ik->jk', wgt, N, N)
    tau1 = float(np.linalg.eigvalsh(T)[-1])
    v = a.astype(np.float32)
    return dict(n_blocks=len(cs),
                coh_med=float(np.median(C)), coh_p25=float(np.percentile(C, 25)),
                coh_p75=float(np.percentile(C, 75)),
                align=float((3 * tau1 - 1) / 2),
                mean_dn=float(v.mean()), std_dn=float(v.std()),
                sat255=float((a >= 255).mean()), fill=float((a > 0).mean()))


def run_scroll(sample, vid, n=N_RANDOM):
    long_id, shape0 = vol_info(sample, vid)
    origins, n_cand, level = pick_random_rois(sample, long_id, shape0, n, SEED)
    z0 = open_level(sample, long_id, 0)

    def one(i_o):
        i, o = i_o
        cp = os.path.join(CACHE, f"{sample}_rnd{i:02d}.npy")
        if os.path.exists(cp):
            a = np.load(cp)
        else:
            a = np.asarray(z0[o[0]:o[0] + ROI, o[1]:o[1] + ROI, o[2]:o[2] + ROI])
            np.save(cp, a)
        return o, a

    rows = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for o, a in ex.map(one, list(enumerate(origins))):
            if a.shape != (ROI, ROI, ROI):
                continue
            st = coherence_stats(a)
            if st is None:
                continue
            st["origin"] = list(map(int, o))
            rows.append(st)
    return dict(scroll=sample, voxel_um=VOX_UM.get(sample, 9.362),
                probe_level=level, n_frame_candidates=n_cand,
                frame="uniform-random over K2b fill>0.98 central-z band",
                seed=SEED, block=BLOCK, sigma=SIGMA, n_rois=len(rows), rois=rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(CACHE, exist_ok=True)
    only = sys.argv[1:] or None
    summary = {}
    for sample, vid in SCROLLS:
        if only and sample not in only:
            continue
        p = os.path.join(OUT, f"{sample}.json")
        if os.path.exists(p):
            summary[sample] = json.load(open(p))
            print(f"{sample}: cached", flush=True)
            continue
        # S3/DNS on this host flaps; a transient failure must not cost a scroll.
        last = None
        for attempt in range(4):
            try:
                r = run_scroll(sample, vid)
                json.dump(r, open(p, "w"), indent=1)
                summary[sample] = r
                cm = [x["coh_med"] for x in r["rois"]]
                print(f"{sample}: {len(cm)} random ROIs  coh med={np.median(cm):.3f} "
                      f"range {min(cm):.3f}-{max(cm):.3f}"
                      + (f"  (attempt {attempt + 1})" if attempt else ""), flush=True)
                last = None
                break
            except Exception as e:
                last = e
                print(f"{sample}: attempt {attempt + 1} failed "
                      f"({type(e).__name__}: {e}) — retrying", flush=True)
                time.sleep(20 * (attempt + 1))
        if last is not None:
            print(f"{sample}: FAILED after 4 attempts {type(last).__name__}: {last}", flush=True)
    json.dump(summary, open(os.path.join(OUT, "k2c_summary.json"), "w"), indent=1)
    print("wrote", os.path.join(OUT, "k2c_summary.json"))


if __name__ == "__main__":
    main()
