"""Scoring for the w059/C0 pod.
  python oi_score.py c0 <our.png> <published.tif>      -> results/c0_<tag>.json (Pearson r on joint support at ds4)
  python oi_score.py ds <name> <map.png> <px_um>        -> ds4/ds16 npys (block means) + stats
  python oi_score.py fwdrev <name> <fwd.png> <rev.png>  -> fwd/rev Pearson r at ds4 on joint support
Maps are uint8 PNGs from optimized_inference (cv2.imwrite); the published reference is a uint8 TIFF."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.environ["SCRIPTS"])
import curvelib as cl


def load_map(path):
    import cv2
    a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if a is None:
        import tifffile
        a = tifffile.imread(path)
    if a.ndim == 3:
        a = a[..., 0]
    return a


def block_mean(a, f):
    H, W = a.shape[0] // f * f, a.shape[1] // f * f
    return a[:H, :W].astype(np.float32).reshape(H // f, f, W // f, f).mean(axis=(1, 3))


def pearson_joint(a4, b4):
    m = (a4 > 0) & (b4 > 0)
    if m.sum() < 1000:
        return float("nan"), int(m.sum())
    return float(np.corrcoef(a4[m], b4[m])[0, 1]), int(m.sum())


def c0(ours, ref, tag):
    a = load_map(ours); b = load_map(ref)
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    a4, b4 = block_mean(a[:h, :w], 4), block_mean(b[:h, :w], 4)
    r, n = pearson_joint(a4, b4)
    res = dict(tag=tag, ours_shape=list(a.shape), ref_shape=list(b.shape), r_ds4_joint=r, n_joint_ds4=n,
               ours_nonzero=float((a > 0).mean()), ref_nonzero=float((b > 0).mean()),
               ours_p99=float(np.percentile(a[a > 0], 99)) if (a > 0).any() else None,
               ref_p99=float(np.percentile(b[b > 0], 99)) if (b > 0).any() else None, gate=0.90, passed=bool(r >= 0.90))
    json.dump(res, open(os.path.join(cl.RESULTS, f"c0_{tag}.json"), "w"), indent=1)
    np.save(os.path.join(cl.OUT, "maps", f"c0_{tag}_ours_ds4.npy"), np.clip(np.rint(a4), 0, 255).astype(np.uint8))
    if not os.path.exists(os.path.join(cl.OUT, "maps", "c0_reference_ds4.npy")):
        np.save(os.path.join(cl.OUT, "maps", "c0_reference_ds4.npy"), np.clip(np.rint(b4), 0, 255).astype(np.uint8))
    cl.say(f"C0 {tag}: r_ds4(joint)={r:.4f} on {n} px; ours nonzero {res['ours_nonzero']:.3f} vs ref {res['ref_nonzero']:.3f}; "
           f"shapes {a.shape} vs {b.shape} -> {'PASS' if res['passed'] else 'below 0.90'}")
    sys.exit(0 if res["passed"] else 21)


def ds(name, path, px_um):
    a = load_map(path)
    a4 = np.clip(np.rint(block_mean(a, 4)), 0, 255).astype(np.uint8)
    a16 = np.clip(np.rint(block_mean(a, 16)), 0, 255).astype(np.uint8)
    np.save(os.path.join(cl.OUT, "maps", f"{name}_ds4.npy"), a4)
    np.save(os.path.join(cl.OUT, "maps", f"{name}_ds16.npy"), a16)
    nz = a > 0
    st = dict(name=name, shape=list(a.shape), px_um=float(px_um), ds16_px_um=float(px_um) * 16, nonzero_frac=float(nz.mean()),
              p50=float(np.percentile(a[nz], 50)) if nz.any() else None, p99=float(np.percentile(a[nz], 99)) if nz.any() else None)
    p = os.path.join(cl.RESULTS, "maps.json")
    d = json.load(open(p)) if os.path.exists(p) else {}
    d[name] = st; json.dump(d, open(p, "w"), indent=1)
    cl.say(f"DS {name}: {a.shape} nonzero {st['nonzero_frac']:.3f} p99 {st['p99']} -> ds4/ds16 npys ({st['ds16_px_um']:.1f} um/px at ds16)")


def fwdrev(name, f, r_):
    a4 = block_mean(load_map(f), 4); b4 = block_mean(load_map(r_), 4)
    h, w = min(a4.shape[0], b4.shape[0]), min(a4.shape[1], b4.shape[1])
    r, n = pearson_joint(a4[:h, :w], b4[:h, :w])
    p = os.path.join(cl.RESULTS, "maps.json")
    d = json.load(open(p)) if os.path.exists(p) else {}
    d[f"{name}_fwdrev"] = dict(r_ds4_joint=r, n_joint=n); json.dump(d, open(p, "w"), indent=1)
    cl.say(f"FWDREV {name}: r={r:.4f} on {n} ds4 px (control 0.094; corpus min 0.22; gate 5 requires < 0.20)")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "c0":
        c0(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "ds":
        ds(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "fwdrev":
        fwdrev(sys.argv[2], sys.argv[3], sys.argv[4])
