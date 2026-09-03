"""Cross-acquisition null RE-AUDIT (queued item, $0 local).

Question: our corpus-wide median L(0.99) = 25.1 vs the 14.2x reported on Discord
(williamshermer, pixel-level). Hypothesis: a MASK convention difference, not a
signal difference. The frozen harness takes quantiles over the JOINT-VALID mask
(value > 2/255 in both maps), which drops the model's exact-zero papyrus
interior; a "sheet footprint" convention keeps those pixels. Here every pair is
re-scored under both conventions, with the null (B rolled by ROLL / rotated
180) built so that B's SHEET MASK is rolled WITH B's calls, i.e. the null's
support is footprint_A & roll(footprint_B) -- the "sheet mask rolled with the
calls" variant the queue asked for. Registration, r_and_L and constants are
imported from the frozen xacq_score.py unchanged.

  python xacq_reaudit.py [--limit N]   -> out/xacq/reaudit_masks.json + .md"""
import argparse, json, os, sys, time
import numpy as np
from scipy import ndimage as ndi
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xacq_score as X

HOLE_PX = 4096      # fill enclosed holes smaller than 64x64 px in the footprint
CLOSE = 11          # closing element (px) for the footprint


def footprint(valid):
    closed = ndi.binary_closing(valid, np.ones((CLOSE, CLOSE), bool))
    holes = ~closed
    lab, n = ndi.label(holes)
    if n:
        sizes = np.bincount(lab.ravel())
        small = (sizes < HOLE_PX)
        small[0] = False
        closed = closed | small[lab]
    return closed


def score_masked(a, bs, m):
    if m.sum() < 10:
        return dict(r=None, L={f"{q:.2f}": None for q in X.Q_LIST}, n=int(m.sum()))
    r, L = X.r_and_L(a[m], bs[m])
    return dict(r=round(float(r), 4), L=L, n=int(m.sum()))


def one(sample, seg):
    a, b0 = X.load_pair(sample, seg)
    bs, mbs, reg = X.register(a, b0)
    va = a > X.VALID_T
    fa, fb = footprint(va), footprint(mbs)
    out = dict(sample=sample, seg=seg, shape=list(a.shape), reg=reg,
               frac_valid_A=round(float(va.mean()), 4), frac_fp_A=round(float(fa.mean()), 4),
               frac_valid_B=round(float(mbs.mean()), 4), frac_fp_B=round(float(fb.mean()), 4))
    # convention 1: frozen joint-valid (reproduces the shipped numbers)
    jv = va & mbs
    out["jointvalid"] = dict(real=score_masked(a, bs, jv))
    rb, rm = np.roll(bs, X.ROLL, (0, 1)), np.roll(mbs, X.ROLL, (0, 1))
    out["jointvalid"]["roll"] = score_masked(a, rb, va & rm)
    out["jointvalid"]["rot180"] = score_masked(a, bs[::-1, ::-1], va & mbs[::-1, ::-1])
    # convention 2: sheet footprint (zeros inside the sheet kept); null rolls B's footprint with B
    fp = fa & fb
    out["footprint"] = dict(real=score_masked(a, bs, fp))
    out["footprint"]["roll"] = score_masked(a, rb, fa & np.roll(fb, X.ROLL, (0, 1)))
    out["footprint"]["rot180"] = score_masked(a, bs[::-1, ::-1], fa & fb[::-1, ::-1])
    # convention 3: footprint support, but only pixels where BOTH are valid count as "calls"
    # (zeros can never be in the top 1%, so this equals convention 2 for L unless zeros dominate)
    out["zero_frac_in_fp_A"] = round(float((~va & fp).mean() / max(fp.mean(), 1e-9)), 4)
    out["zero_frac_in_fp_B"] = round(float((~mbs & fp).mean() / max(fp.mean(), 1e-9)), 4)
    return out


def med(v):
    v = [x for x in v if x is not None and x == x]
    return round(float(np.median(v)), 3) if v else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
    pairs = X.load_manifest_pairs()
    norm = []
    for p in pairs:
        if isinstance(p, dict):
            norm.append((p["sample"], p["seg"]))
        else:
            norm.append((p[0], p[1]))
    if a.limit:
        norm = norm[:a.limit]
    rows, t0 = [], time.time()
    for i, (s, g) in enumerate(norm):
        try:
            rows.append(one(s, g))
        except Exception as e:
            rows.append(dict(sample=s, seg=g, error=str(e)[:200]))
        if (i + 1) % 10 == 0:
            print(f"{i+1}/{len(norm)} {time.time()-t0:.0f}s", flush=True)
    ok = [r for r in rows if "error" not in r]
    def agg(sub, conv, key, q="0.99"):
        return med([(r[conv][key]["L"] or {}).get(q) for r in sub])
    strata = {"all": ok, "main": [r for r in ok if r["sample"] != "PHercParis4"], "Paris4": [r for r in ok if r["sample"] == "PHercParis4"]}
    summ = {}
    for name, sub in strata.items():
        summ[name] = dict(n=len(sub))
        for conv in ("jointvalid", "footprint"):
            summ[name][conv] = dict(
                r_real=med([r[conv]["real"]["r"] for r in sub]), r_roll=med([r[conv]["roll"]["r"] for r in sub]),
                r_rot=med([r[conv]["rot180"]["r"] for r in sub]),
                L99_real=agg(sub, conv, "real"), L99_roll=agg(sub, conv, "roll"), L99_rot=agg(sub, conv, "rot180"),
                L95_real=agg(sub, conv, "real", "0.95"), L90_real=agg(sub, conv, "real", "0.90"),
                n_L99_real_ge5=sum(1 for r in sub if ((r[conv]["real"]["L"] or {}).get("0.99") or 0) >= 5),
                n_survive_roll=sum(1 for r in sub if X.survives(r[conv]["real"]["r"], (r[conv]["real"]["L"] or {}).get("0.99"),
                                                                r[conv]["roll"]["r"], (r[conv]["roll"]["L"] or {}).get("0.99"))),
                n_survive_rot=sum(1 for r in sub if X.survives(r[conv]["real"]["r"], (r[conv]["real"]["L"] or {}).get("0.99"),
                                                               r[conv]["rot180"]["r"], (r[conv]["rot180"]["L"] or {}).get("0.99"))))
        summ[name]["zero_frac_in_fp_A"] = med([r["zero_frac_in_fp_A"] for r in sub])
        summ[name]["zero_frac_in_fp_B"] = med([r["zero_frac_in_fp_B"] for r in sub])
    res = dict(version="xacq-reaudit-1.0", harness=X.HARNESS_VERSION, footprint=dict(close_px=CLOSE, hole_px=HOLE_PX),
               n_pairs=len(rows), n_ok=len(ok), summary=summ, pairs=rows, runtime_s=round(time.time() - t0, 1))
    os.makedirs(X.OUT, exist_ok=True)
    json.dump(res, open(X.OUT / "reaudit_masks.json", "w"), indent=1)
    lines = ["# xacq null re-audit: mask convention (2026-09-03)", "",
             "Frozen harness = joint-valid mask (value > 2/255 in both maps). Footprint = 11-px closing of each",
             "map's valid region with enclosed holes < 64x64 px filled; the null rolls B's footprint WITH B's calls.", "",
             "| stratum | n | conv | r real | r roll | r rot | L99 real | L99 roll | L99 rot | L95 | L90 | L99>=5 | survive roll | survive rot | zero-frac A/B in footprint |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in summ.items():
        for conv in ("jointvalid", "footprint"):
            c = s[conv]
            lines.append(f"| {name} | {s['n']} | {conv} | {c['r_real']} | {c['r_roll']} | {c['r_rot']} | {c['L99_real']} | {c['L99_roll']} | {c['L99_rot']} | "
                         f"{c['L95_real']} | {c['L90_real']} | {c['n_L99_real_ge5']} | {c['n_survive_roll']} | {c['n_survive_rot']} | {s['zero_frac_in_fp_A']}/{s['zero_frac_in_fp_B']} |")
    open(X.OUT / "reaudit_masks.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
