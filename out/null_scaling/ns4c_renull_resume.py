"""NS4c -- resumable tile-128 re-null of the v2 significance passers.

Same code path and seed scheme as ns4b (the screen's OWN prep/score/
joint_permute from analyze_survey_corpus_v2, seeds 20260823+1000*draw+tile),
but runs a draw RANGE per invocation and appends to a state file, so each
foreground call stays short.  Asserts the observed score matches the stored
corpus_analysis_v2.json value before any null draw is counted.

Run:  python ns4c_renull_resume.py <key> <draw_start> <draw_end> [tile=128]
"""
import sys, os, json, time
import numpy as np

TRACKD = r"C:\Users\benbl\Desktop\Vsuvious\trackD"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TRACKD)
sys.path.insert(0, os.path.join(TRACKD, "verify_flag"))
import analyze_survey_corpus_v2 as V2

KEY = sys.argv[1]
D0, D1 = int(sys.argv[2]), int(sys.argv[3])
TILE = int(sys.argv[4]) if len(sys.argv) > 4 else 128

state_path = os.path.join(HERE, "ns4c_%s_t%d.json" % (KEY.replace("/", "_"), TILE))
state = json.load(open(state_path)) if os.path.exists(state_path) else None

units, meta = V2.build_units(only={KEY})
u = [x for x in units if x["key"] == KEY][0]
img, msk, mf = V2.prep(u["img"], u.get("mask"))

t0 = time.time()
bf, bg, df, dg = V2.score(img, msk, u["px"], want_detail=True)
obs = bf["prom"]

# hard premise check: our score must equal the stored screen value
stored = json.load(open(os.path.join(TRACKD, "out", "survey", "corpus_analysis_v2.json")))
sr = [r for r in stored["results"] + stored.get("control", []) if r["name"] == KEY][0]
assert abs(obs - sr["obs_prominence"]) < 5e-3, (obs, sr["obs_prominence"])
assert abs(bf["theta"] - sr["theta_deg"]) < 1e-6, (bf["theta"], sr["theta_deg"])
print("%s obs %.3f == stored %.3f  theta %.1f  period %.3f mm  [%.1fs/score]"
      % (KEY, obs, sr["obs_prominence"], bf["theta"], bf["period_mm"], time.time() - t0),
      flush=True)

if state is None:
    state = {"key": KEY, "tile": TILE, "shape": list(img.shape), "px_um": u["px"],
             "obs_prominence": round(obs, 3), "theta_deg": bf["theta"],
             "period_mm": round(bf["period_mm"], 3),
             "stored_v2": {"p_200perm_tile64": sr["empirical_p"], "z": sr["z_corrected"],
                           "null_sd_tile64": sr["null_sd"], "null_mean_tile64": sr["null_mean"]},
             "n_blocks": V2.n_tiles(msk, TILE), "draws": {}}

for draw in range(D0, D1):
    k = str(draw)
    if k in state["draws"]:
        continue
    seed = 20260823 + 1000 * draw + TILE
    rng = np.random.default_rng(seed)
    pi, pm = V2.joint_permute(img, msk, rng, TILE)
    if pi is None:
        state["draws"][k] = None
        continue
    pbf, _, _, _ = V2.score(pi, pm, u["px"])
    state["draws"][k] = round(pbf["prom"], 4)
    json.dump(state, open(state_path, "w"), indent=1)
    print("  draw %d -> %.3f   [%.0fs]" % (draw, pbf["prom"], time.time() - t0), flush=True)

nl = np.array([v for v in state["draws"].values() if v is not None])
if len(nl) >= 8:
    n_ge = int((nl >= obs).sum())
    p = (1 + n_ge) / (len(nl) + 1)
    mu, sd = float(nl.mean()), float(nl.std(ddof=1))
    state["summary"] = dict(n_perm=len(nl), null_mean=round(mu, 3), null_sd=round(sd, 3),
                            null_max=round(float(nl.max()), 3),
                            z=round((obs - mu) / sd, 3), empirical_p=round(p, 5),
                            n_null_ge_obs=n_ge)
    print("tile %d after %d perms: null %.2f +/- %.2f max %.2f  z %+.2f  p %.4f (%d >= obs)"
          % (TILE, len(nl), mu, sd, nl.max(), (obs - mu) / sd, p, n_ge), flush=True)
json.dump(state, open(state_path, "w"), indent=1)
print("state -> %s  [%.1f min]" % (state_path, (time.time() - t0) / 60))
