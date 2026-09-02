"""NS4b -- re-null the corpus screen's borderline cases at a block size at or
above the measured map correlation length.

Uses the screen's OWN code path (analyze_survey_corpus_v2: prep, score,
joint_permute) -- nothing reimplemented -- with a fresh, independent seed
stream.  For each target map:
  * observed prominence via V2.score (validated: reproduces the stored values),
  * null at tile 64 (the v2 choice; independent 50-draw consistency check),
  * null at tile 128 (>= measured L(0.1) of the passers' maps, still below the
    claimed ruling periods of 6.9-8.0 mm = 199-230 ds4 px, and below the
    control's 4.678 mm = 125 px period only barely -- discussed in writeup).

Run:  python ns4b_renull.py <key> [nperm]
"""
import sys, os, json, time
import numpy as np

TRACKD = r"C:\Users\benbl\Desktop\Vsuvious\trackD"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TRACKD)
sys.path.insert(0, os.path.join(TRACKD, "verify_flag"))
import analyze_survey_corpus_v2 as V2

KEY = sys.argv[1]
NPERM = int(sys.argv[2]) if len(sys.argv) > 2 else 50
TILES = [int(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["64", "128"])]

units, meta = V2.build_units(only={KEY})
u = [x for x in units if x["key"] == KEY][0]
img, msk, mf = V2.prep(u["img"], u.get("mask"))
print("%s  shape %s  px %.3f um" % (KEY, img.shape, u["px"]), flush=True)

t0 = time.time()
bf, bg, df, dg = V2.score(img, msk, u["px"], want_detail=True)
obs = bf["prom"]
print("obs prominence %.3f  theta %.1f  period %.3f mm   [%.1fs/score]"
      % (obs, bf["theta"], bf["period_mm"], time.time() - t0), flush=True)

res = {"key": KEY, "shape": list(img.shape), "px_um": u["px"], "n_perm": NPERM,
       "obs_prominence": round(obs, 3), "theta_deg": bf["theta"],
       "period_mm": round(bf["period_mm"], 3), "tiles": {}}
outpath = os.path.join(HERE, "ns4b_%s.json" % KEY.replace("/", "_"))

for tile in TILES:
    nulls = []
    nb = V2.n_tiles(msk, tile)
    print("tile %d: %d perm blocks" % (tile, nb), flush=True)
    if nb < 8:
        res["tiles"][str(tile)] = {"error": "fewer than 8 blocks"}
        continue
    for draw in range(NPERM):
        seed = 20260823 + 1000 * draw + tile
        rng = np.random.default_rng(seed)
        pi, pm = V2.joint_permute(img, msk, rng, tile)
        if pi is None:
            continue
        pbf, _, _, _ = V2.score(pi, pm, u["px"])
        nulls.append(pbf["prom"])
        if draw % 10 == 9:
            print("  tile %d draw %d/%d  %.0fs" % (tile, draw + 1, NPERM,
                                                   time.time() - t0), flush=True)
    nl = np.array(nulls)
    n_ge = int((nl >= obs).sum())
    p = (1 + n_ge) / (len(nl) + 1)
    sd = float(nl.std(ddof=1)); mu = float(nl.mean())
    res["tiles"][str(tile)] = dict(
        n_blocks=nb, n_perm=len(nl), null_mean=round(mu, 3), null_sd=round(sd, 3),
        null_max=round(float(nl.max()), 3), z=round((obs - mu) / sd, 3),
        empirical_p=round(p, 5), n_null_ge_obs=n_ge)
    print("tile %3d: null %7.2f +/- %6.2f  max %7.2f  z %+6.2f  p %.4f (%d/%d >= obs)"
          % (tile, mu, sd, nl.max(), (obs - mu) / sd, p, n_ge, len(nl)), flush=True)
    json.dump(res, open(outpath, "w"), indent=1)

res["elapsed_s"] = round(time.time() - t0, 1)
json.dump(res, open(outpath, "w"), indent=1)
print("wrote %s  [%.1f min]" % (outpath, res["elapsed_s"] / 60))
