"""NS1 -- premise verification for the null-scaling methods result.

Reproduces, from the on-disk Frag1 arrays and the committed estimator (m1_lib),
the two claims that the whole item rests on:
  (a) sd*sqrt(n) is NOT a transferable constant  (slabs of Frag1: 57,650..146,740;
      plates: 59,173 Frag2 .. 139,407 Frag1),
  (b) the 40-translation null is heavy-tailed (one Frag1 draw at -12,833 against
      an IQR of [-992,+1334]).
If either fails to reproduce, the item is KILLED.
"""
import sys, os, json, time
import numpy as np
MET = r"C:\Users\benbl\AppData\Local\Temp\claude\C--Users-benbl-Desktop-Vsuvious\35d67aba-4ea6-4b13-a3c0-2f3fc87bbe13\scratchpad\metrology"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MET)
import m1_lib as G
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

t0 = time.time()
T = 32; H, W = 8181, 6330
L = (np.array(Image.open(os.path.join(MET, "f1_inklabels.png"))) > 0)
M = (np.array(Image.open(os.path.join(MET, "f1_mask.png"))) > 0)
P = np.load(os.path.join(MET, "f1_P.npy"))
stored = json.load(open(os.path.join(MET, "g1_results.json")))
sf = {f["fragment"]: f for f in stored["fragments"]}
out = {"checks": []}

def chk(name, ok, detail):
    out["checks"].append({"name": name, "ok": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + name + " :: " + detail, flush=True)

# ---- (1) whole-plate 40-draw null, committed seed ----
ny, nx = H // T, W // T
Mv = G.tileview(M, ny, nx, T); Pv = G.tileview(P, ny, nx, T)
obs, n_tiles, _, _ = G.excess(Pv, L, Mv, ny, nx, T)
sh = G.shifts_for(H, W)
nulls = np.array([G.excess(Pv, G.roll(L, dy, dx), Mv, ny, nx, T)[0] for dy, dx in sh])
sd = float(nulls.std(ddof=1)); mn = float(nulls.mean())
stored_nulls = np.array(sf["Frag1"]["null_values"])
chk("frag1_obs_and_tiles", abs(obs - 1914.07) < 0.5 and n_tiles == 2560,
    "obs %+.2f (stored +1914.07), tiles %d (stored 2560)" % (obs, n_tiles))
chk("frag1_null_draws_reproduce",
    np.allclose(np.round(nulls, 2), stored_nulls, atol=0.02),
    "max |draw diff| vs stored = %.3f" % np.abs(np.round(nulls, 2) - stored_nulls).max())
chk("frag1_null_sd_2755", abs(sd - 2755.28) < 1.0,
    "sd %.2f (stored 2755.28), mean %+.2f (stored -39.08)" % (sd, mn))
q25, med, q75 = np.percentile(nulls, [25, 50, 75])
worst = nulls[np.argmax(np.abs(nulls - nulls.mean()))]
sd_wo = float(np.delete(nulls, np.argmax(np.abs(nulls - nulls.mean()))).std(ddof=1))
# excess kurtosis
zc = nulls - nulls.mean()
kurt = float((zc**4).mean() / (zc**2).mean()**2 - 3.0)
var_share = float(zc[np.argmax(np.abs(zc))]**2 / (zc**2).sum())
chk("frag1_null_heavy_tailed", worst < -12000 and var_share > 0.30,
    "worst draw %+.0f, IQR [%+.0f,%+.0f], excess kurtosis %+.2f, "
    "single-draw variance share %.1f%%, sd without it %.1f" %
    (worst, q25, q75, kurt, 100 * var_share, sd_wo))
out["frag1_whole_plate"] = dict(obs=round(float(obs), 2), n_tiles=int(n_tiles),
    null_sd=round(sd, 2), null_mean=round(mn, 2), worst_draw=round(float(worst), 1),
    iqr=[round(float(q25), 1), round(float(q75), 1)], excess_kurtosis=round(kurt, 2),
    worst_draw_variance_share=round(var_share, 4), sd_without_worst=round(sd_wo, 1),
    sd_sqrt_n=round(sd * np.sqrt(n_tiles), 0))
print("whole plate done %.1fs" % (time.time() - t0), flush=True)

# ---- (2) the 7 independent 1024-row slabs, committed seed per slab ----
slabs = []
for a in range(0, H - 1023, 1024):
    hh = 1024; nyy = hh // T
    Mv2 = G.tileview(M[a:a+hh], nyy, nx, T); Pv2 = G.tileview(P[a:a+hh], nyy, nx, T)
    Lb2 = L[a:a+hh]
    oo, nn, _, _ = G.excess(Pv2, Lb2, Mv2, nyy, nx, T)
    s2 = G.shifts_for(hh, W)
    nl2 = np.array([G.excess(Pv2, G.roll(Lb2, dy, dx), Mv2, nyy, nx, T)[0] for dy, dx in s2])
    nl2 = nl2[np.isfinite(nl2)]
    ssd = float(nl2.std(ddof=1))
    slabs.append(dict(rows=[a, a+hh-1], n_tiles=int(nn), obs=round(float(oo), 1),
                      null_sd=round(ssd, 1), sd_sqrt_n=round(ssd * np.sqrt(nn), 0)))
    print("  slab %4d..%4d tiles %4d null_sd %8.1f sd*sqrt(n) %9.0f" %
          (a, a+hh-1, nn, ssd, ssd*np.sqrt(nn)), flush=True)
out["slabs_1024"] = slabs
st = json.load(open(os.path.join(MET, "m1_frag1_diag.json")))["slabs_1024"]
rep = all(abs(s["null_sd"] - t["null_sd"]) < 1.0 and s["n_tiles"] == t["n_tiles"]
          for s, t in zip(slabs, st))
cons = [s["sd_sqrt_n"] for s in slabs]
chk("frag1_slab_constants_reproduce", rep,
    "sd*sqrt(n) over 7 slabs: min %.0f max %.0f (stored 57,650..146,740), ratio %.2fx"
    % (min(cons), max(cons), max(cons)/min(cons)))

# ---- (3) cross-plate constants from the stored raw draws (Frag2/Frag6 volumes
#          are band fetches; their null draws are stored raw in g1_results.json) ----
xp = {}
for fr in ("Frag1", "Frag2", "Frag6"):
    nv = np.array(sf[fr]["null_values"]); n = sf[fr]["n_tiles"]
    s = float(nv.std(ddof=1))
    zc = nv - nv.mean()
    xp[fr] = dict(n_tiles=n, sd_recomputed=round(s, 1),
                  sd_stored=sf[fr]["null_sd"],
                  sd_sqrt_n=round(s * np.sqrt(n), 0),
                  excess_kurtosis=round(float((zc**4).mean()/(zc**2).mean()**2 - 3), 2))
out["plates"] = xp
ok = (abs(xp["Frag1"]["sd_recomputed"] - 2755.3) < 1 and
      abs(xp["Frag2"]["sd_recomputed"] - 934.0) < 1 and
      abs(xp["Frag6"]["sd_recomputed"] - 1839.1) < 1)
chk("plate_constants", ok,
    "sd*sqrt(n): Frag1 %.0f, Frag2 %.0f, Frag6 %.0f -> plate ratio %.2fx" %
    (xp["Frag1"]["sd_sqrt_n"], xp["Frag2"]["sd_sqrt_n"], xp["Frag6"]["sd_sqrt_n"],
     max(x["sd_sqrt_n"] for x in xp.values()) / min(x["sd_sqrt_n"] for x in xp.values())))

out["verdict"] = "PREMISE REPRODUCED" if all(c["ok"] for c in out["checks"]) else "PREMISE FAILED"
out["elapsed_s"] = round(time.time() - t0, 1)
json.dump(out, open(os.path.join(HERE, "ns1_premise.json"), "w"), indent=1)
print(out["verdict"], "%.1fs" % out["elapsed_s"])
