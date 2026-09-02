"""G1 v2 verdict -- walks PREREG_G1_V2.md (git a2e0001) exactly.
Inputs: g1v2_f1/f2/f6.json (measured), g1v2_bulk.json (per-plate BULK)."""
import os, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
F = {t: json.load(open(os.path.join(HERE, "g1v2_%s.json" % t))) for t in ("f1", "f2", "f6")}
B = json.load(open(os.path.join(HERE, "g1v2_bulk.json")))
BULK = {"Frag1": B["Frag1"]["BULK_used"], "Frag2": B["Frag2"]["BULK_used"], "Frag6": B["Frag6"]["BULK_used"]}
MG_PER_VOX = 3.24e-4 * 0.6 * 1000.0     # 3.24 um voxel * 0.6 g/cm3 papyrus -> mg/cm2 per vox-equiv

out = dict(prereg="PREREG_G1_V2.md, git a2e0001, 2026-08-24T10:43:59-05:00")

# ---- 1. VOID gate (Frag1 anchors) ----
void = F["f1"]["VOID_check"]
out["VOID_gate"] = void
assert void["ok"], "VOID: anchors failed to reproduce"

# ---- 2. reproduction of v1 on identical tiles ----
rep = {}
for t in ("f1", "f2", "f6"):
    f = F[t]
    rep[f["fragment"]] = dict(obs=f["obs"], obs_ok=f["reproduction"]["obs_ok"],
                              n_tiles=f["n_tiles"], n_ok=f["reproduction"]["n_ok"],
                              G0_auc=f["G0"]["paired_auc"], G0_ok=f["G0"]["reproduces_v1"],
                              rigid_first40_match_v1=f["rigid_diag"]["first40_match_v1"])
out["v1_reproduction"] = rep
assert all(r["obs_ok"] and r["n_ok"] and r["G0_ok"] and r["rigid_first40_match_v1"] for r in rep.values())

# ---- 3. G1-v2 gate per fragment ----
gates = {}
for t in ("f1", "f2", "f6"):
    f = F[t]; sd = f["block_null"]["boot_sd"]
    gates[f["fragment"]] = dict(n_tiles=f["n_tiles"], tiles_ok=bool(f["n_tiles"] >= 1700),
                                block_sd=sd, sd_ok=bool(sd <= 2300.0),
                                G0_pass=f["G0"]["passes"],
                                admitted=bool(f["n_tiles"] >= 1700 and sd <= 2300.0 and f["G0"]["passes"]))
out["G1v2_gates"] = gates
admitted = [k for k, g in gates.items() if g["admitted"]]
out["admitted_fragments"] = admitted
out["pooling_licensed"] = bool(len(admitted) >= 3)

# ---- 4. detection test (block null, Bonferroni across admitted set) ----
det = {F[t]["fragment"]: F[t]["z_block"] for t in ("f1", "f2", "f6") if F[t]["fragment"] in admitted}
out["detection"] = dict(threshold=4.0, family="Bonferroni across %d admitted fragments "
                        "(one-sided per-test alpha 3.17e-5, family alpha %.1e)" % (len(admitted), 3.17e-5 * len(admitted)),
                        z_block=det, any_over=bool(any(z > 4.0 for z in det.values())))

# ---- 5. per-fragment ceilings, BOTH framings ----
ceil = {}
for t in ("f1", "f2", "f6"):
    f = F[t]; name = f["fragment"]; bulk = BULK[name]
    obs = f["obs"]
    b_sd = f["block_null"]["boot_sd"]           # block framing: zero-centered null
    c_block = abs(obs - 0.0) + 2 * b_sd
    r_m, r_sd = f["rigid_diag"]["mean"], f["rigid_diag"]["sd"]
    c_rigid = abs(obs - r_m) + 2 * r_sd
    ceil[name] = dict(
        BULK_DN_per_voxel=bulk,
        block=dict(null_mean=0.0, null_sd=b_sd, ceiling_DNvox=round(c_block, 1),
                   ceiling_voxequiv=round(c_block / bulk, 4),
                   ceiling_mg_cm2=round(c_block / bulk * MG_PER_VOX, 4)),
        rigid=dict(null_mean=r_m, null_sd=r_sd, ceiling_DNvox=round(c_rigid, 1),
                   ceiling_voxequiv=round(c_rigid / bulk, 4),
                   ceiling_mg_cm2=round(c_rigid / bulk * MG_PER_VOX, 4)),
        block_over_rigid_sd_ratio=round(b_sd / r_sd, 3))
out["per_fragment_ceilings"] = ceil

# ---- 6. pooled inverse-variance (voxel-equivalents primary) ----
def pool(framing):
    d = []; s = []
    for t in ("f1", "f2", "f6"):
        f = F[t]; name = f["fragment"]
        if name not in admitted: continue
        bulk = BULK[name]
        if framing == "block":
            mu, sd = 0.0, f["block_null"]["boot_sd"]
        else:
            mu, sd = f["rigid_diag"]["mean"], f["rigid_diag"]["sd"]
        d.append((f["obs"] - mu) / bulk); s.append(sd / bulk)
    d = np.array(d); s = np.array(s); wi = 1.0 / s**2
    delta = float((d * wi).sum() / wi.sum()); sig = float(1.0 / np.sqrt(wi.sum()))
    c = abs(delta) + 2 * sig
    return dict(pooled_delta_voxequiv=round(delta, 4), pooled_sd_voxequiv=round(sig, 4),
                pooled_z_diagnostic_only=round(delta / sig, 3),
                ceiling_voxequiv=round(c, 4), ceiling_mg_cm2=round(c * MG_PER_VOX, 4))
if out["pooling_licensed"]:
    out["pooled"] = dict(block=pool("block"), rigid=pool("rigid"),
        note="inverse-variance in papyrus-voxel-equivalents (per-plate BULK), "
             "pooled ceiling = |pooled delta| + 2*pooled sd; pooled z is descriptive only, "
             "the detection rule is per-fragment z>4 Bonferroni")

# ---- 7. verdict ----
if out["detection"]["any_over"]:
    out["outcome"] = "DETECTION"
else:
    out["outcome"] = "CEILING" if out["pooling_licensed"] else "PER_FRAGMENT_CEILINGS_ONLY"
out["v1_verdict_reported_alongside"] = (
    "PREREG_G1 v1 (git 502ae65): G1 FAIL via Frag1 40-draw rigid sd 2755.3 > 2300; outcome "
    "NO_BOUND_QUOTABLE; pooling unlicensed at 2 of 3. v2 does not erase it: the 40-draw sd "
    "carries 19.4% self-noise (SE 406) and the converged 400-draw value is 2091.9 -- the FAIL "
    "was a +1.6 SE estimator fluctuation, which is why v2 pre-registered a converged null. "
    "Both verdicts ship together.")
json.dump(out, open(os.path.join(HERE, "G1V2_RESULTS.json"), "w"), indent=1)

print("OUTCOME:", out["outcome"])
print("admitted:", admitted, " pooling:", out["pooling_licensed"])
for n, g in gates.items(): print("  %s  n %5d  block sd %7.1f  admitted %s" % (n, g["n_tiles"], g["block_sd"], g["admitted"]))
print("z_block:", det)
for n, c in ceil.items():
    print("  %s ceiling: block %7.1f DNvox = %.4f vox = %.4f mg/cm2 | rigid %7.1f = %.4f vox | sd ratio %.3f"
          % (n, c["block"]["ceiling_DNvox"], c["block"]["ceiling_voxequiv"], c["block"]["ceiling_mg_cm2"],
             c["rigid"]["ceiling_DNvox"], c["rigid"]["ceiling_voxequiv"], c["block_over_rigid_sd_ratio"]))
if "pooled" in out:
    for fr in ("block", "rigid"):
        p = out["pooled"][fr]
        print("  POOLED %s: delta %.4f sd %.4f -> ceiling %.4f vox-equiv = %.4f mg/cm2"
              % (fr, p["pooled_delta_voxequiv"], p["pooled_sd_voxequiv"], p["ceiling_voxequiv"], p["ceiling_mg_cm2"]))

# ---- 8. robustness: alternative pooling reading (weighted mean of per-fragment ceilings) ----
alt = {}
for fr in ("block", "rigid"):
    cs = []; ws = []
    for t in ("f1", "f2", "f6"):
        f = F[t]; name = f["fragment"]
        if name not in admitted: continue
        bulk = BULK[name]
        sd = f["block_null"]["boot_sd"] if fr == "block" else f["rigid_diag"]["sd"]
        cs.append(ceil[name][fr]["ceiling_voxequiv"]); ws.append((bulk / sd)**2)
    cs = np.array(cs); ws = np.array(ws)
    alt[fr] = round(float((cs * ws).sum() / ws.sum()), 4)
out["pooling_robustness"] = dict(
    alternative_reading_weighted_mean_of_ceilings_voxequiv=alt,
    note="the committed reading is fixed-effect inverse-variance on (obs-null_mean)/BULK with "
         "pooled ceiling |pooled delta|+2*pooled sd; the alternative (inverse-variance-weighted "
         "mean of the per-fragment ceilings) is reported for robustness and is looser")
json.dump(out, open(os.path.join(HERE, "G1V2_RESULTS.json"), "w"), indent=1)
print("alt pooling (weighted mean of ceilings):", alt)
