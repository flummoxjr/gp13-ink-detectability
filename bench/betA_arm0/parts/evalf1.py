"""Score checkpoints on the five native PHerc0139 crops.
  python evalf1.py <tag> <ckpt.pth> [--reverse]
For each crop: run infer (forward; reverse too if asked), then
  * khj1222 replica: 256-bin histograms of the uint8 prediction over the supervision
    region split by ink; F1 at every threshold t (positive iff score >= t); best F1;
    floor = 2p/(1+p); margin = best - floor;
  * benchmark: tie-corrected pixel AUC (curvelib.hist_auc), pos = ink & sup, neg = sup & ~ink.
Appends to results/eval.json under <tag>."""
import json, os, subprocess, sys
import numpy as np
sys.path.insert(0, os.environ["SCRIPTS"])
import curvelib as cl

NATIVE, PREDS = os.environ["NATIVE"], cl.PREDS
CROPS = ["w035", "w039", "w040", "w041", "w044"]


def infer(zarr_path, ckpt, out_tif, direction):
    if os.path.exists(out_tif) and os.path.getsize(out_tif) > 0:
        return
    cmd = ["uv", "run", "--no-sync", "--extra", "models", "python", "-m", "vesuvius.ink_detection.inference.infer",
           zarr_path, ckpt, out_tif, "--direction", direction, "--batch-size", os.environ.get("BATCH", "16"),
           "--num-workers", os.environ.get("WORKERS", "8"), "--gpus", "0", "--no-compile"]
    r = subprocess.run(cmd, cwd="/workspace/villa/vesuvius", capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out_tif):
        cl.say("INFER FAILED: " + (r.stderr or r.stdout)[-600:].replace("\n", " | "))
        raise RuntimeError(f"infer failed for {out_tif}")


def f1_sweep(pred, ink, sup):
    q = pred.astype(np.int64)
    hp = np.bincount(q[ink & sup], minlength=256).astype(np.float64)
    hn = np.bincount(q[sup & ~ink], minlength=256).astype(np.float64)
    P = hp.sum(); N = hn.sum()
    tp = np.cumsum(hp[::-1])[::-1]          # positives with score >= t
    fp = np.cumsum(hn[::-1])[::-1]
    fn = P - tp
    f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
    t = int(np.argmax(f1))
    p_ink = P / max(P + N, 1)
    floor = 2 * p_ink / (1 + p_ink)
    return dict(best_f1=float(f1[t]), threshold=t, floor=float(floor), margin=float(f1[t] - floor),
                n_pos=int(P), n_neg=int(N))


def main():
    tag, ckpt = sys.argv[1], sys.argv[2]
    do_rev = "--reverse" in sys.argv
    import tifffile
    p = os.path.join(cl.RESULTS, "eval.json")
    res = json.load(open(p)) if os.path.exists(p) else {}
    row = res.get(tag, {"ckpt": ckpt, "crops": {}})
    for w in CROPS:
        z = os.path.join(NATIVE, f"{w}_crop.zarr")
        ink = np.load(os.path.join(NATIVE, f"{w}_ink.npy")); sup = np.load(os.path.join(NATIVE, f"{w}_sup.npy"))
        cell = {}
        for d, suffix in (("forward", ""), ("reverse", "_reverse")):
            if d == "reverse" and not do_rev:
                continue
            tif = os.path.join(PREDS, f"{tag}_{w}{suffix}.tif")
            if d == "forward":
                infer(z, ckpt, tif, "forward")
            else:
                infer(z, ckpt, os.path.join(PREDS, f"{tag}_{w}_r.tif"), "reverse")
                tif = os.path.join(PREDS, f"{tag}_{w}_r.tif")
            pred = tifffile.imread(tif)
            if pred.shape != ink.shape:
                pred = np.clip(np.rint(cl.resample_pred(pred, ink.shape)), 0, 255).astype(np.uint8)
            f1 = f1_sweep(pred, ink, sup)
            q = cl.quantize_map(pred.astype(np.float32))
            auc = cl.hist_auc(cl.masked_hist(q, ink & sup), cl.masked_hist(q, sup & ~ink))
            cell[d] = dict(f1=f1, auc=float(auc))
            cl.say(f"EVAL {tag} {w} {d}: bestF1={f1['best_f1']:.4f}@{f1['threshold']} floor={f1['floor']:.3f} "
                   f"margin={f1['margin']:+.3f} AUC={auc:.4f}")
        row["crops"][w] = cell
    f = [row["crops"][w]["forward"]["f1"] for w in CROPS]
    row["native5_mean_best_f1"] = float(np.mean([x["best_f1"] for x in f]))
    row["native5_mean_floor"] = float(np.mean([x["floor"] for x in f]))
    row["native5_mean_margin"] = float(np.mean([x["margin"] for x in f]))
    row["native5_mean_auc_forward"] = float(np.mean([row["crops"][w]["forward"]["auc"] for w in CROPS]))
    res[tag] = row
    json.dump(res, open(p, "w"), indent=1)
    cl.say(f"EVAL {tag}: native-5 mean bestF1={row['native5_mean_best_f1']:.4f} (floor {row['native5_mean_floor']:.3f}, "
           f"margin {row['native5_mean_margin']:+.3f}); mean AUC fwd={row['native5_mean_auc_forward']:.4f}")


if __name__ == "__main__":
    main()
