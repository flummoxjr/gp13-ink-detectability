"""K1b — validate the frozen depth-contrast intensity statistic on held-out segments.

QC overturned K1's kill: ink at native 9.362um has an ANTISYMMETRIC depth profile
(brighter ~z=10-14, darker ~z=4-8), which the z-mean cancels. Frozen statistic
(pre-registered here before touching held-out data):

    D(u,v) = mean(f[z=10..14]) - mean(f[z=4..8])      (calibrated DN)

Positive class: inklabel>0 (max-z). Negatives: supervision-mask, non-ink.
Report AUC per segment on w039, w040, w041, w044 (w035 = discovery segment,
reported for reference only). Also z-std texture AUC as secondary channel.

Outputs: trackD/out/k1b_depth_validation.json
"""
import json
import os

import numpy as np

S3 = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
HF = "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/native9-scrollprizeorg-21slices"
VOL = "9.362um-1.2m-113keV-volume-20250728140407.zarr"
SEGS = {
    "w035": "20260317000000-w035_2026031718",
    "w039": "20260302000000-w039_2026030210",
    "w040": "20250831000000-w040_2025083102",
    "w041": "20260108000000-w041_2026010816",
    "w044": "20260115000000-w044_2026011522",
}
CACHE = r"D:\vesuvius-data\trackD"
OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\out"
WIN = (-0.03, 0.145)


def load_zarr(url, cache_name):
    cache = os.path.join(CACHE, cache_name)
    if os.path.exists(cache):
        return np.load(cache)
    import zarr, fsspec
    print("fetching", url.split("/")[-2] + "/" + url.split("/")[-1])
    arr = np.asarray(zarr.open(fsspec.get_mapper(url), mode="r")["0"][:])
    np.save(cache, arr)
    return arr


def tie_auc(pos, neg, max_n=1_000_000):
    from scipy.stats import rankdata
    rng = np.random.default_rng(0)
    if len(pos) > max_n:
        pos = rng.choice(pos, max_n, replace=False)
    if len(neg) > max_n:
        neg = rng.choice(neg, max_n, replace=False)
    allv = np.concatenate([pos, neg])
    ranks = rankdata(allv)
    n1, n2 = len(pos), len(neg)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


def main():
    os.makedirs(OUT, exist_ok=True)
    results = {}
    for seg, segdir in SEGS.items():
        try:
            surf = load_zarr(f"{S3}/PHerc0139/segments/{segdir}/surface-volumes/{VOL}", f"{seg}_surf.npy")
            ink = load_zarr(f"{HF}/{seg}/{seg}_inklabels.zarr", f"{seg}_ink.npy")
            sup = load_zarr(f"{HF}/{seg}/{seg}_supervision_mask.zarr", f"{seg}_sup.npy")
        except Exception as e:
            print(f"{seg}: FETCH FAILED — {e}")
            results[seg] = {"error": str(e)[:200]}
            continue
        nz = surf.shape[0]
        lo, hi = WIN
        f = surf.astype(np.float32) / 255.0 * (hi - lo) + lo

        ink2d = (ink > 0).max(axis=0)
        sup2d = (sup > 0).max(axis=0)
        nonzero2d = (surf > 0).mean(axis=0) > 0.9
        valid = sup2d & nonzero2d
        pos_n, neg_n = int((ink2d & valid).sum()), int((valid & ~ink2d).sum())
        if pos_n < 500 or neg_n < 500:
            results[seg] = {"skipped": f"too few px pos={pos_n} neg={neg_n}"}
            continue

        # frozen depth-contrast statistic (indices safe for nz>=15; these are 28-slice)
        upper = f[10:15].mean(axis=0)
        lower = f[4:9].mean(axis=0)
        D = upper - lower
        auc_D = tie_auc(D[ink2d & valid], D[valid & ~ink2d])

        zstd = f.std(axis=0)
        auc_T = tie_auc(zstd[ink2d & valid], zstd[valid & ~ink2d])

        results[seg] = {
            "pos_px": pos_n, "neg_px": neg_n, "n_slices": int(nz),
            "auc_depth_contrast": round(auc_D, 4),
            "auc_zstd_texture": round(auc_T, 4),
        }
        print(f"{seg}: depth-contrast AUC={auc_D:.4f}  z-std AUC={auc_T:.4f}  (pos {pos_n}, neg {neg_n})")

    with open(os.path.join(OUT, "k1b_depth_validation.json"), "w") as fh:
        json.dump(results, fh, indent=1)
    held = [v["auc_depth_contrast"] for k, v in results.items()
            if k != "w035" and "auc_depth_contrast" in v]
    if held:
        print(f"\nheld-out mean depth-contrast AUC: {np.mean(held):.4f} over {len(held)} segments")


if __name__ == "__main__":
    main()
