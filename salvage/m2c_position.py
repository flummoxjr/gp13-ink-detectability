"""m2c_position.py — where does the fired volume sit relative to the papyrus
sheet mask (surface-pred > 127)?  Signed distance: + = inside sheet (vox into
sheet), - = outside (vox away from sheet).  Ink should sit AT the sheet surface
(|d| <= ~2, biased slightly outside/at interface); crack/gap-filling sits
outside; sheet-core texture sits deep inside.

Null (DISCIPLINE): compare fired voxels' signed-distance distribution to the
distribution over ALL interior (ct>5) voxels of the same tile — i.e., what a
volume-random firing pattern would give — via per-tile subsampling; 200 tile-
level bootstrap draws for the null band.
"""
import json, os
import numpy as np
from scipy import ndimage as ndi

OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\salvage"
CACHE = os.path.join(OUT, "cache")
RNG = np.random.default_rng(7)
inv = json.load(open(os.path.join(OUT, "inventory.json")))["samples"]

bins = np.arange(-15.5, 16.5, 1.0)   # signed distance bins
h_fired = np.zeros(len(bins) - 1)
h_inter = np.zeros(len(bins) - 1)
per_tile = []
sheet_fills = []
null_draws = []

for rec in inv:
    z, y, x = rec["tile"]
    prob8 = np.load(rec["file"])
    ct = np.load(os.path.join(CACHE, f"ct_{z}_{y}_{x}.npy"))
    surf = np.load(os.path.join(CACHE, f"surf_{z}_{y}_{x}.npy"))
    binm = prob8 >= 128
    interior = ct > 5
    if interior.mean() < 0.05:
        continue
    sheet = surf > 127
    sheet_fills.append(float(sheet[interior].mean()) if interior.any() else 0.0)
    if sheet.mean() < 0.01 or (~sheet).mean() < 0.01:
        continue
    d_in = ndi.distance_transform_edt(sheet)        # >0 inside sheet
    d_out = ndi.distance_transform_edt(~sheet)      # >0 outside sheet
    signed = np.where(sheet, d_in, -d_out)

    f = binm & interior
    if f.sum() < 100:
        continue
    sv = signed[f]
    iv = signed[interior]
    h_fired += np.histogram(sv, bins)[0]
    h_inter += np.histogram(iv, bins)[0]
    per_tile.append({
        "tile": [z, y, x],
        "fired_med_signed": float(np.median(sv)),
        "interior_med_signed": float(np.median(iv)),
        "fired_frac_surface_band": float((np.abs(sv) <= 2).mean()),
        "interior_frac_surface_band": float((np.abs(iv) <= 2).mean()),
        "fired_frac_inside": float((sv > 0).mean()),
        "interior_frac_inside": float((iv > 0).mean()),
        "fired_frac_deep_out": float((sv < -3).mean()),
        "interior_frac_deep_out": float((iv < -3).mean()),
    })
    # null draw material: sample len(sv) interior voxels at random, get stats
    k = min(len(sv), len(iv))
    null_draws.append((iv, len(sv)))

obs_band = np.array([t["fired_frac_surface_band"] for t in per_tile])
ref_band = np.array([t["interior_frac_surface_band"] for t in per_tile])
obs_in = np.array([t["fired_frac_inside"] for t in per_tile])
ref_in = np.array([t["interior_frac_inside"] for t in per_tile])

# permutation-style null: per tile, random interior subsets of the fired size
NB = 200
null_band = np.empty(NB)
null_inside = np.empty(NB)
for b in range(NB):
    vals_band, vals_in = [], []
    for iv, nf in null_draws:
        s = iv[RNG.integers(0, len(iv), size=min(nf, len(iv)))]
        vals_band.append((np.abs(s) <= 2).mean())
        vals_in.append((s > 0).mean())
    null_band[b] = np.mean(vals_band)
    null_inside[b] = np.mean(vals_in)

res = {
    "n_tiles_used": len(per_tile),
    "sheet_fill_of_interior_median": float(np.median(sheet_fills)),
    "signed_hist_bins_centers": [float(b) for b in (bins[:-1] + 0.5)],
    "signed_hist_fired": [int(v) for v in h_fired],
    "signed_hist_interior": [int(v) for v in h_inter],
    "fired_frac_surface_band_mean": float(obs_band.mean()),
    "null_frac_surface_band": {"mean": float(null_band.mean()), "sd": float(null_band.std()),
                               "p_emp": float((null_band >= obs_band.mean()).mean())},
    "fired_frac_inside_mean": float(obs_in.mean()),
    "null_frac_inside": {"mean": float(null_inside.mean()), "sd": float(null_inside.std()),
                         "p_emp_greater": float((null_inside >= obs_in.mean()).mean()),
                         "p_emp_less": float((null_inside <= obs_in.mean()).mean())},
    "per_tile": per_tile,
}
json.dump(res, open(os.path.join(OUT, "position_vs_sheet.json"), "w"), indent=1)

hf = h_fired / h_fired.sum()
hi = h_inter / h_inter.sum()
print(f"tiles used: {len(per_tile)}; sheet fill of interior (median): {np.median(sheet_fills):.2f}")
print(f"fired: frac within +-2 of sheet surface = {obs_band.mean():.3f}  "
      f"(volume-random null {null_band.mean():.3f}+-{null_band.std():.3f}, p={res['null_frac_surface_band']['p_emp']:.3f})")
print(f"fired: frac INSIDE sheet = {obs_in.mean():.3f}  "
      f"(null {null_inside.mean():.3f}+-{null_inside.std():.3f})")
print("\nsigned-distance profile (center, fired-share, interior-share):")
for c, a, b2 in zip(bins[:-1] + 0.5, hf, hi):
    if abs(c) <= 8:
        bar = '#' * int(a * 200)
        print(f"{c:+5.1f}  {a:.3f}  {b2:.3f}  {bar}")
mode_bin = float((bins[:-1] + 0.5)[np.argmax(hf)])
print(f"\nfired mode at signed distance {mode_bin:+.1f} (interior mode {float((bins[:-1]+0.5)[np.argmax(hi)]):+.1f})")
