"""Bet A input statistics (prereg gate for arm 1): measure the pooled 2.4->9.6 um training volumes
and the five native 9.36 um eval crops with the SAME per-crop 2-D estimator the arm-1 degradation
uses (vesuvius.ink_detection.data.degradation.measure_2d: Hanning radial PSD, residual-based white
floor from the 0.35-0.48 cyc/px band, structural SNR at q = 0.25, bandwidth = max q with PSD >= 2x floor,
DN headroom = p99.5 - p0.5 in-mask). Writes results/input_stats.json and says a summary line per store.

  python measure_inputs.py <aligned9 dir> <native dir> <k2b_index.json> [n_windows=64] [size=128]

Reading: arm 1 can only degrade a crop whose measured SNR/bandwidth exceeds the drawn target. If the
pooled sources already sit at or below the index targets, the noise/blur steps are inactive and arm 1
reduces to the headroom match -- which is reported here BEFORE any arm-1 pod spends on training."""
import glob, json, os, sys
import numpy as np
import zarr
sys.path.insert(0, os.environ["SCRIPTS"])
import curvelib as cl
from vesuvius.ink_detection.data.degradation import measure_2d, sample_inplane_windows


def open_level0(path):
    """Pooled aligned9 volumes and native crops are zarr GROUPS with the full-resolution array at "0"
    (prepare_9um_isotropic_input: group.create_array("0")); a bare array is accepted too."""
    g = zarr.open(path, mode="r")
    if hasattr(g, "shape"):
        return g
    for key in ("0", "s0", "level0"):
        if key in g:
            return g[key]
    keys = sorted(k for k in g.array_keys()) if hasattr(g, "array_keys") else []
    if not keys:
        raise ValueError(f"no array in zarr group {path}")
    return g[keys[0]]


def stats_for(volume, name, n, size, rng):
    wins = sample_inplane_windows(volume, n, size, rng)
    rows = []
    for w in wins:
        q, p, nz = measure_2d(w)
        i25 = int(np.argmin(np.abs(q - 0.25)))
        above = (q > 0.02) & (p / nz >= 2.0)
        dn = w[w > 0].astype(np.float32)
        rows.append(dict(snr_q025=float(p[i25] / nz), bandwidth_cyc_px=float(q[above].max()) if above.any() else 0.0,
                         dn_headroom=float(np.percentile(dn, 99.5) - np.percentile(dn, 0.5)), mean_dn=float(dn.mean())))
    if not rows:
        return dict(name=name, n=0)
    def med_iqr(k):
        v = np.array([r[k] for r in rows]); return [round(float(np.median(v)), 4), round(float(np.percentile(v, 25)), 4), round(float(np.percentile(v, 75)), 4)]
    return dict(name=name, n=len(rows), shape=list(volume.shape), snr_q025_med_iqr=med_iqr("snr_q025"),
                bandwidth_med_iqr=med_iqr("bandwidth_cyc_px"), dn_headroom_med_iqr=med_iqr("dn_headroom"), mean_dn=med_iqr("mean_dn"))


def main():
    aligned, native, index_path = sys.argv[1], sys.argv[2], sys.argv[3]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 64
    size = int(sys.argv[5]) if len(sys.argv) > 5 else 128
    rng = np.random.default_rng(20260903)
    out = dict(estimator="2-D per-window, residual floor (conservative); index targets are 3-D air/residual", window=size, n_per_volume=n, pooled={}, native={})
    for kind, root in (("pooled", aligned), ("native", native)):
        for z in sorted(glob.glob(os.path.join(root, "*.zarr"))):
            name = os.path.basename(z)[:-5]
            try:
                st = stats_for(open_level0(z), name, n, size, rng)
            except Exception as e:  # one bad store must not sink a reporting stage
                st = dict(name=name, n=0, error=f"{type(e).__name__}: {e}"[:200])
            out[kind][name] = st
            if st.get("n"):
                cl.say(f"INPUTSTAT {kind} {name}: snr25 {st['snr_q025_med_iqr'][0]:.1f} [{st['snr_q025_med_iqr'][1]:.1f},{st['snr_q025_med_iqr'][2]:.1f}] "
                       f"bw {st['bandwidth_med_iqr'][0]:.3f} head {st['dn_headroom_med_iqr'][0]:.0f} (n={st['n']})")
            else:
                cl.say(f"INPUTSTAT {kind} {name}: FAILED {st.get('error', 'no windows')}")
    idx = json.load(open(index_path))
    tgt_snr = sorted(r["snr_q025"] for rec in idx.values() for r in rec["rois"])
    tgt_bw = sorted(r["bandwidth_cyc_px"] for rec in idx.values() for r in rec["rois"])
    pooled_snr = [v["snr_q025_med_iqr"][0] for v in out["pooled"].values() if v.get("n")]
    pooled_bw = [v["bandwidth_med_iqr"][0] for v in out["pooled"].values() if v.get("n")]
    out["index_targets"] = dict(n=len(tgt_snr), snr_q025_median=float(np.median(tgt_snr)), bandwidth_median=float(np.median(tgt_bw)))
    out["arm1_active_fraction"] = dict(
        noise=float(np.mean([s > np.median(tgt_snr) for s in pooled_snr])) if pooled_snr else None,
        blur=float(np.mean([b > np.median(tgt_bw) for b in pooled_bw])) if pooled_bw else None)
    json.dump(out, open(os.path.join(cl.RESULTS, "input_stats.json"), "w"), indent=1)
    cl.say(f"INPUTSTAT summary: pooled snr25 median {np.median(pooled_snr) if pooled_snr else float('nan'):.1f} vs index-target median {np.median(tgt_snr):.1f}; "
           f"pooled bw median {np.median(pooled_bw) if pooled_bw else float('nan'):.3f} vs target median {np.median(tgt_bw):.3f}; "
           f"fraction of pooled stores where arm-1 noise / blur is active: {out['arm1_active_fraction']}")


if __name__ == "__main__":
    main()
