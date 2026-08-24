"""Corpus-level analysis of the GP-scroll segment survey.

Runs the decisive text test (line-ruling periodicity vs a texture-preserving
block-permutation null) on every surveyed segment's downsampled prediction map,
plus intensity/symmetry summaries, and ranks segments by text-likeness.

Calibration anchors (from salvage/ink9um_1203_verdict.md):
  control w035  : ruling 4.68-4.73 mm, z = +25.6..+34.1, fwd/rev r = 0.076
  1203 probe    : z in [-2.7, +1.0], fwd/rev r = 0.51-0.60  (texture)
Maps here are 4x-downsampled: 9.362um*4 = 37.4 um/px (8.64um scrolls: 34.6 um/px).

Usage: python analyze_survey_corpus.py [--maps-dir DIR]
Outputs: out/survey/corpus_analysis.json, corpus_ranking.png
"""
import glob
import json
import os
import sys

import numpy as np

OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\out\survey"
PX_UM = {"PHerc1203": 9.362 * 4, "PHerc1447": 8.64 * 4, "PHerc0800": 8.64 * 4}
BAND_MM = (1.7, 8.4)      # ruling search band (matches the validated battery)
N_PERM = 16               # block permutations per map (texture-preserving null)
BLOCK = 32                # permutation block size, in analysis px (~2.4 mm at ds8)
BLANK_P99 = 195
DS = 2                    # extra downsample before rotation work (maps are ds4 already)
THETA_STEP = 15           # orientation search step, degrees
# At ds8 (74.9 um/px for 9.362 um scrolls) a 4.7 mm ruling is ~63 px — still
# oversampled by ~30x vs the search band's lower edge, so the extra decimation
# costs no ruling sensitivity while cutting rotation cost ~4x.


def orientation_project(img, mask, theta_deg):
    """Mean projection profile along `theta`, over valid pixels only."""
    from scipy.ndimage import rotate
    r_img = rotate(img, theta_deg, order=1, reshape=True, mode="constant", cval=0)
    r_msk = rotate(mask.astype(np.float32), theta_deg, order=1, reshape=True, mode="constant", cval=0)
    num = r_img.sum(axis=1)
    den = r_msk.sum(axis=1)
    ok = den > 0.25 * den.max() if den.max() > 0 else den > 0
    if ok.sum() < 64:
        return None
    prof = np.zeros_like(num)
    prof[ok] = num[ok] / den[ok]
    return prof[ok]


def ruling_score(img, mask, px_um):
    """Best periodic-peak prominence over orientations, in the ruling band."""
    best = (0.0, None, None)
    lo_px = BAND_MM[0] * 1000 / px_um
    hi_px = BAND_MM[1] * 1000 / px_um
    for theta in range(0, 180, THETA_STEP):
        prof = orientation_project(img, mask, theta)
        if prof is None or len(prof) < 4 * lo_px:
            continue
        p = prof - prof.mean()
        w = np.hanning(len(p))
        ps = np.abs(np.fft.rfft(p * w)) ** 2
        freqs = np.fft.rfftfreq(len(p))
        with np.errstate(divide="ignore"):
            periods = np.where(freqs > 0, 1.0 / np.maximum(freqs, 1e-9), np.inf)
        sel = (periods >= lo_px) & (periods <= hi_px)
        if sel.sum() < 3:
            continue
        band = ps[sel]
        # prominence = peak over median of the band (robust, scale-free)
        prom = float(band.max() / max(np.median(band), 1e-12))
        if prom > best[0]:
            best = (prom, theta, float(periods[sel][np.argmax(band)] * px_um / 1000))
    return best


def block_permute(img, mask, rng, block=BLOCK):
    """Shuffle blocks within the mask: preserves stroke texture, destroys long-range order."""
    h, w = img.shape
    out = img.copy()
    coords = [(y, x) for y in range(0, h - block, block) for x in range(0, w - block, block)
              if mask[y:y + block, x:x + block].mean() > 0.8]
    if len(coords) < 8:
        return None
    perm = rng.permutation(len(coords))
    src = [img[y:y + block, x:x + block].copy() for y, x in coords]
    for i, (y, x) in enumerate(coords):
        out[y:y + block, x:x + block] = src[perm[i]]
    return out


def analyze(path, scroll):
    a = np.load(path).astype(np.float32)[::DS, ::DS]
    mask = a > 0
    if mask.mean() < 0.05 or mask.sum() < 5e4 / (DS * DS):
        return None
    # degenerate maps (near-constant response) cannot support a periodicity claim
    v = a[mask]
    if float(v.std()) < 1.0:
        return None
    px_um = PX_UM.get(scroll, 37.4) * DS
    obs_prom, theta, period_mm = ruling_score(a, mask, px_um)
    rng = np.random.default_rng(0)
    nulls = []
    for _ in range(N_PERM):
        pa = block_permute(a, mask, rng)
        if pa is None:
            break
        nulls.append(ruling_score(pa, mask, px_um)[0])
    if len(nulls) < 8:
        return None
    mu, sd = float(np.mean(nulls)), float(np.std(nulls) + 1e-9)
    return {
        "ruling_prominence": round(obs_prom, 2),
        "ruling_z": round((obs_prom - mu) / sd, 2),
        "null_mean": round(mu, 2), "null_sd": round(sd, 2),
        "theta_deg": theta, "period_mm": round(period_mm, 2) if period_mm else None,
        "frac_gt_blankp99": round(float((a > BLANK_P99).mean()), 5),
        "p99": float(np.percentile(v, 99)), "mask_frac": round(float(mask.mean()), 3),
    }


def main():
    maps = sorted(glob.glob(os.path.join(OUT, "maps_shard*", "*_forward_ds4.npy")))
    survey = {}
    p = os.path.join(OUT, "survey_all.json")
    if os.path.exists(p):
        for r in json.load(open(p)):
            survey[r["name"]] = r
    print(f"{len(maps)} forward maps found; {len(survey)} survey records")
    results = []
    for m in maps:
        name = os.path.basename(m).replace("_forward_ds4.npy", "")
        rec = survey.get(name, {})
        scroll = rec.get("scroll", "PHerc1203")
        try:
            r = analyze(m, scroll)
        except Exception as e:
            print(f"{name}: FAILED {str(e)[:70]}")
            continue
        if r is None:
            continue
        r.update({"name": name, "scroll": scroll, "fwd_rev_r": rec.get("fwd_rev_r"),
                  "tripwire": len((rec.get("forward") or {}).get("tripwire_hits") or [])})
        results.append(r)
        print(f"{scroll}/{name}: ruling_z={r['ruling_z']:+.2f} period={r['period_mm']}mm "
              f"fwd_rev_r={r['fwd_rev_r']} hot={r['frac_gt_blankp99']:.5f}", flush=True)

    results.sort(key=lambda x: -x["ruling_z"])
    with open(os.path.join(OUT, "corpus_analysis.json"), "w") as f:
        json.dump({"control_reference": {"ruling_z": [25.6, 34.1], "period_mm": [4.68, 4.73],
                                         "fwd_rev_r": 0.076},
                   "n_segments": len(results), "results": results}, f, indent=1)
    if results:
        zs = [r["ruling_z"] for r in results]
        print(f"\n=== CORPUS RESULT: {len(results)} segments ===")
        print(f"ruling z: min {min(zs):+.2f}, median {np.median(zs):+.2f}, max {max(zs):+.2f}")
        print(f"control reference: +25.6 to +34.1")
        flagged = [r for r in results if r["ruling_z"] >= 5]
        print(f"segments with ruling z >= 5: {len(flagged)}")
        for r in flagged[:10]:
            print(f"  ! {r['scroll']}/{r['name']}: z={r['ruling_z']} period={r['period_mm']}mm")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 6))
        colors = {"PHerc1203": "#3A6B8C", "PHerc1447": "#C8791E", "PHerc0800": "#6B8C3A"}
        for sc in colors:
            v = [r["ruling_z"] for r in results if r["scroll"] == sc]
            if v:
                ax.scatter(range(len(v)), sorted(v, reverse=True), label=f"{sc} (n={len(v)})",
                           color=colors[sc], s=28)
        ax.axhline(25.6, color="#A33A2E", ls="--", lw=1.5, label="w035 control (letters), z=+25.6")
        ax.axhline(5, color="gray", ls=":", lw=1, label="flag threshold z=5")
        ax.set_ylabel("line-ruling z vs texture-preserving null")
        ax.set_xlabel("segment (ranked within scroll)")
        ax.set_title("Text-ruling signal across the published GP segment corpus")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "corpus_ranking.png"), dpi=110)
        print("wrote corpus_ranking.png")


if __name__ == "__main__":
    main()
