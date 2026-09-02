"""NS4a -- does '0 of 71 pass' depend on the null at all, and is the v2
permutation tile (64 ds4 px) beyond the maps' measured correlation length?

Part 1: gate decomposition from corpus_analysis_v2.json (no recomputation):
        which gates are null-dependent, and what survives if the significance
        gate is deleted entirely (the most extreme possible null correction).
Part 2: measured 2D autocorrelation length of every scored map (the field the
        64-px joint block permutation assumes uncorrelated at tile scale),
        same prep as the screen (valid_mask -> crop -> map*mask).
"""
import sys, os, json, glob, time
import numpy as np
from scipy import ndimage as ndi

TRACKD = r"C:\Users\benbl\Desktop\Vsuvious\trackD"
OUT = os.path.join(TRACKD, "out", "survey")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TRACKD)
sys.path.insert(0, os.path.join(TRACKD, "verify_flag"))
import analyze_survey_corpus_v2 as V2   # the screen's own prep/valid_mask

t0 = time.time()
d = json.load(open(os.path.join(OUT, "corpus_analysis_v2.json")))
res = d["results"]
out = {}

# ---------------- Part 1: gate decomposition ----------------
GATES = ["gate_significance", "gate_cycles", "gate_autocorr", "gate_band_bin",
         "gate_fwd_rev"]
cnt = {g: sum(1 for r in res if r.get(g)) for g in GATES}
n = len(res)
no_sig = [r for r in res if r["gate_cycles"] and r["gate_autocorr"]
          and r["gate_band_bin"] and r["gate_fwd_rev"]]
no_sig_no_fr = [r for r in res if r["gate_cycles"] and r["gate_autocorr"]
                and r["gate_band_bin"]]
sig_passers = [dict(name=r["name"], scroll=r["scroll"], p=r["empirical_p"],
                    z=r["z_corrected"], fwd_rev_r=r["fwd_rev_r"],
                    cycles=r["n_cycles"], bin=r["peak_bin_index"],
                    tile=r["perm_tile_ds4"])
               for r in res if r["gate_significance"]]
out["gate_decomposition"] = dict(
    n_scored=n, gate_pass_counts=cnt,
    null_dependent_gates=["gate_significance"],
    null_free_gates=["gate_cycles", "gate_autocorr", "gate_band_bin", "gate_fwd_rev"],
    pass_all_except_significance=len(no_sig),
    pass_periodicity_gates_ignoring_sig_and_fwdrev=len(no_sig_no_fr),
    min_fwd_rev_r_scored=min(r["fwd_rev_r"] for r in res if r.get("fwd_rev_r") is not None),
    significance_passers=sig_passers)
print("scored %d | pass counts %s" % (n, cnt))
print("pass ALL FOUR null-free gates (sig deleted): %d" % len(no_sig))
print("pass 3 periodicity gates (sig+fwdrev deleted): %d" % len(no_sig_no_fr))
print("min fwd/rev r among scored: %.3f (gate needs < 0.20)"
      % out["gate_decomposition"]["min_fwd_rev_r_scored"])
print("significance passers (raw p<=0.05):")
for s in sig_passers:
    print("   %-45s p=%.5f z=%+.2f fwd_rev_r=%.3f cycles=%.1f bin=%d"
          % (s["name"], s["p"], s["z"], s["fwd_rev_r"], s["cycles"], s["bin"]))

# ---------------- Part 2: map correlation lengths ----------------
def corr_profile(img, msk, maxlag=200):
    """masked 2D autocorrelation of (map - masked mean), radial-ish via x/y axes."""
    x = np.where(msk, img - img[msk].mean(), 0.0).astype(np.float64)
    a = msk.astype(np.float64)
    FY, FX = x.shape[0] * 2, x.shape[1] * 2
    Xf = np.fft.rfft2(x, s=(FY, FX)); Af = np.fft.rfft2(a, s=(FY, FX))
    cxx = np.fft.irfft2(Xf * np.conj(Xf), s=(FY, FX))
    cnn = np.fft.irfft2(Af * np.conj(Af), s=(FY, FX))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = cxx / np.maximum(cnn, 1.0)
    r = r / r[0, 0]
    mx = min(maxlag, x.shape[1] - 1); my = min(maxlag, x.shape[0] - 1)
    return r[0, :mx], r[:my, 0]

def xing(v, thr):
    for k, val in enumerate(v):
        if val < thr: return k
    return len(v)

units, meta = V2.build_units()
rows = []
for u in units:
    img, msk, mf = V2.prep(u["img"], u.get("mask"))
    if img is None:
        continue
    rx, ry = corr_profile(img, msk)
    rows.append(dict(name=u["key"], kind=u["kind"], shape=list(img.shape),
                     L_e_x=xing(rx, 1 / np.e), L_e_y=xing(ry, 1 / np.e),
                     L01_x=xing(rx, 0.1), L01_y=xing(ry, 0.1),
                     r_at_64_x=round(float(rx[64]) if len(rx) > 64 else float("nan"), 3),
                     r_at_64_y=round(float(ry[64]) if len(ry) > 64 else float("nan"), 3)))
    if len(rows) % 20 == 0:
        print("  ...%d maps %.0fs" % (len(rows), time.time() - t0), flush=True)

seg = [r for r in rows if r["kind"] == "segment"]
ctl = [r for r in rows if r["kind"] != "segment"]
def summ(key):
    v = np.array([r[key] for r in seg], float)
    return dict(min=float(np.min(v)), median=float(np.median(v)), max=float(np.max(v)))
out["map_correlation"] = dict(
    n_measured=len(seg),
    L_e_x=summ("L_e_x"), L_e_y=summ("L_e_y"),
    L01_x=summ("L01_x"), L01_y=summ("L01_y"),
    r_at_64_x=summ("r_at_64_x"), r_at_64_y=summ("r_at_64_y"),
    n_seg_L01_over_64=int(sum(1 for r in seg
                              if max(r["L01_x"], r["L01_y"]) > 64)),
    perm_tile=64,
    per_map=rows, control=ctl)
print("segments: L(1/e) x med %.0f max %.0f | L(0.1) x med %.0f max %.0f | "
      "r at lag64 x med %.3f max %.3f"
      % (out["map_correlation"]["L_e_x"]["median"], out["map_correlation"]["L_e_x"]["max"],
         out["map_correlation"]["L01_x"]["median"], out["map_correlation"]["L01_x"]["max"],
         out["map_correlation"]["r_at_64_x"]["median"], out["map_correlation"]["r_at_64_x"]["max"]))
print("segments with L(0.1) > 64 px in either axis: %d of %d"
      % (out["map_correlation"]["n_seg_L01_over_64"], len(seg)))
for r in ctl:
    print("control %-28s shape %s  L(1/e) x/y %d/%d  L(0.1) x/y %d/%d  r@64 x/y %.3f/%.3f"
          % (r["name"], r["shape"], r["L_e_x"], r["L_e_y"], r["L01_x"], r["L01_y"],
             r["r_at_64_x"], r["r_at_64_y"]))

out["elapsed_s"] = round(time.time() - t0, 1)
json.dump(out, open(os.path.join(HERE, "ns4a_corpus_structure.json"), "w"), indent=1)
print("done %.1fs" % out["elapsed_s"])
