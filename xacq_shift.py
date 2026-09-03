"""Cross-acquisition lift vs registration residual (re-audit addendum, $0 local).

At ds8 the harness phase-registers B onto A to ~2 px (16 native px). A pixel-level
lift computed at NATIVE resolution without sub-ds8 registration would carry that
residual; this measures how fast L(0.99) decays when B is deliberately mis-shifted
by 0..4 ds8 px (0..32 native px) after registration. Roll null unchanged.

  python xacq_shift.py [--limit N]  -> out/xacq/reaudit_shift.json + .md"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xacq_score as X

SHIFTS = ((0, 0), (1, 0), (1, 1), (2, 0), (2, 2), (3, 3), (4, 4))


def one(sample, seg):
    a, b0 = X.load_pair(sample, seg)
    bs, mbs, reg = X.register(a, b0)
    va = a > X.VALID_T
    out = dict(sample=sample, seg=seg, reg=reg, shifts={})
    for dy, dx in SHIFTS:
        b2 = X.int_shift(bs, (dy, dx)) if (dy or dx) else bs
        m2 = (X.int_shift(mbs.astype(np.float32), (dy, dx)) > 0.5) if (dy or dx) else mbs
        m = va & m2
        r, L = X.r_and_L(a[m], b2[m]) if m.sum() >= 10 else (float("nan"), {})
        out["shifts"][f"{dy},{dx}"] = dict(n=int(m.sum()), r=round(float(r), 4), L=L)
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
        for dy, dx in SHIFTS:
            k = f"{dy},{dx}"
            summ[name][k] = dict(r=med([r["shifts"][k]["r"] for r in sub]),
                                 L99=med([(r["shifts"][k]["L"] or {}).get("0.99") for r in sub]),
                                 L98=med([(r["shifts"][k]["L"] or {}).get("0.98") for r in sub]),
                                 L95=med([(r["shifts"][k]["L"] or {}).get("0.95") for r in sub]),
                                 n_L99_ge5=sum(1 for r in sub if ((r["shifts"][k]["L"] or {}).get("0.99") or 0) >= 5))
    res = dict(version="xacq-shift-1.0", harness=X.HARNESS_VERSION, shifts_ds8_px=[list(s) for s in SHIFTS], n_pairs=len(rows), n_ok=len(ok),
               summary=summ, pairs=rows, runtime_s=round(time.time() - t0, 1))
    json.dump(res, open(X.OUT / "reaudit_shift.json", "w"), indent=1)
    lines = ["# xacq lift vs registration residual (2026-09-03)", "",
             "After the harness's phase registration, B is deliberately mis-shifted by (dy,dx) ds8 px (x8 = native px).", "",
             "| stratum | n | shift ds8 px (native) | r | L99 | L98 | L95 | L99>=5 |", "|---|---|---|---|---|---|---|---|"]
    for name, s in summ.items():
        for dy, dx in SHIFTS:
            k = f"{dy},{dx}"; c = s[k]
            lines.append(f"| {name} | {s['n']} | ({dy},{dx}) ({8*dy},{8*dx}) | {c['r']} | {c['L99']} | {c['L98']} | {c['L95']} | {c['n_L99_ge5']} |")
    open(X.OUT / "reaudit_shift.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
