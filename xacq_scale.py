"""Cross-acquisition lift vs resolution (re-audit addendum, $0 local).

The harness maps are the published `-ds8.jpg` previews (8x block-downsampled +
JPEG). A pixel-level lift computed at native resolution (the 14.2x figure) is a
different scale. Here L(q) and r are recomputed on each registered pair after
further block-mean downsampling of the ds8 grid by 1, 2, 4, 8 (= native ds8,
ds16, ds32, ds64), with the roll null at the same scale, to measure how lift
depends on scale and whether the trend extrapolates toward the native figure.

  python xacq_scale.py [--limit N]  -> out/xacq/reaudit_scale.json + .md"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xacq_score as X

FACTORS = (1, 2, 4, 8)


def block_mean(a, f):
    if f == 1:
        return a
    H, W = a.shape[0] // f * f, a.shape[1] // f * f
    return a[:H, :W].reshape(H // f, f, W // f, f).mean(axis=(1, 3))


def block_all(m, f):
    if f == 1:
        return m
    H, W = m.shape[0] // f * f, m.shape[1] // f * f
    return m[:H, :W].reshape(H // f, f, W // f, f).all(axis=(1, 3))


def one(sample, seg):
    a, b0 = X.load_pair(sample, seg)
    bs, mbs, reg = X.register(a, b0)
    va = a > X.VALID_T
    rb, rm = np.roll(bs, X.ROLL, (0, 1)), np.roll(mbs, X.ROLL, (0, 1))
    out = dict(sample=sample, seg=seg, shape=list(a.shape), scales={})
    for f in FACTORS:
        af, bf, rbf = block_mean(a, f), block_mean(bs, f), block_mean(rb, f)
        m = block_all(va & mbs, f); mr = block_all(va & rm, f)
        r, L = X.r_and_L(af[m], bf[m]) if m.sum() >= 10 else (float("nan"), {})
        rr, Lr = X.r_and_L(af[mr], rbf[mr]) if mr.sum() >= 10 else (float("nan"), {})
        out["scales"][f"ds{8*f}"] = dict(n=int(m.sum()), r=round(float(r), 4), L=L, n_roll=int(mr.sum()),
                                         r_roll=round(float(rr), 4), L_roll=Lr)
    return out


def med(v):
    v = [x for x in v if x is not None and x == x]
    return round(float(np.median(v)), 3) if v else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
    pairs = X.load_manifest_pairs()
    if a.limit:
        pairs = pairs[:a.limit]
    rows, t0 = [], time.time()
    for i, (s, g) in enumerate(pairs):
        try:
            rows.append(one(s, g))
        except Exception as e:
            rows.append(dict(sample=s, seg=g, error=str(e)[:200]))
        if (i + 1) % 10 == 0:
            print(f"{i+1}/{len(pairs)} {time.time()-t0:.0f}s", flush=True)
    ok = [r for r in rows if "error" not in r]
    strata = {"all": ok, "main": [r for r in ok if r["sample"] != "PHercParis4"], "Paris4": [r for r in ok if r["sample"] == "PHercParis4"]}
    summ = {}
    for name, sub in strata.items():
        summ[name] = dict(n=len(sub))
        for f in FACTORS:
            k = f"ds{8*f}"
            summ[name][k] = dict(r=med([r["scales"][k]["r"] for r in sub]), r_roll=med([r["scales"][k]["r_roll"] for r in sub]),
                                 L99=med([(r["scales"][k]["L"] or {}).get("0.99") for r in sub]),
                                 L99_roll=med([(r["scales"][k]["L_roll"] or {}).get("0.99") for r in sub]),
                                 L98=med([(r["scales"][k]["L"] or {}).get("0.98") for r in sub]),
                                 L95=med([(r["scales"][k]["L"] or {}).get("0.95") for r in sub]),
                                 L90=med([(r["scales"][k]["L"] or {}).get("0.90") for r in sub]),
                                 n_L99_ge5=sum(1 for r in sub if ((r["scales"][k]["L"] or {}).get("0.99") or 0) >= 5),
                                 n_px_med=med([r["scales"][k]["n"] for r in sub]))
    res = dict(version="xacq-scale-1.0", harness=X.HARNESS_VERSION, factors=list(FACTORS), n_pairs=len(rows), n_ok=len(ok),
               summary=summ, pairs=rows, runtime_s=round(time.time() - t0, 1))
    json.dump(res, open(X.OUT / "reaudit_scale.json", "w"), indent=1)
    lines = ["# xacq lift vs resolution (2026-09-03)", "",
             "Harness maps are the published ds8 JPEG previews. Each registered pair is block-mean downsampled further",
             "by 1/2/4/8 (ds8 = as shipped); the roll null is computed at the same scale on the same block-valid support.", "",
             "| stratum | n | scale | r | r roll | L99 | L99 roll | L98 | L95 | L90 | L99>=5 | median px |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in summ.items():
        for f in FACTORS:
            k = f"ds{8*f}"; c = s[k]
            lines.append(f"| {name} | {s['n']} | {k} | {c['r']} | {c['r_roll']} | {c['L99']} | {c['L99_roll']} | {c['L98']} | {c['L95']} | {c['L90']} | {c['n_L99_ge5']} | {c['n_px_med']} |")
    open(X.OUT / "reaudit_scale.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
