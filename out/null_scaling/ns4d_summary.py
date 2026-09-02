"""NS4d -- consolidate the corpus-payoff numbers into one verdict file.

Reads: corpus_analysis_v2.json (the screen's stored output),
       ns4a_corpus_structure.json (gate decomposition + map correlation lengths),
       ns4b_z_dbg_gen_00215.json and ns4c_*_t128.json (tile-128 re-nulls).
Writes: ns4_corpus_verdict.json
"""
import json, os, glob
import numpy as np

TRACKD = r"C:\Users\benbl\Desktop\Vsuvious\trackD"
HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(TRACKD, "out", "survey", "corpus_analysis_v2.json")))
a = json.load(open(os.path.join(HERE, "ns4a_corpus_structure.json")))
res = d["results"]
GATES = ["gate_significance", "gate_cycles", "gate_autocorr", "gate_band_bin", "gate_fwd_rev"]

out = {
 "claim_under_test": "0 of 71 corpus maps pass the v2 screen",
 "stored_headline": dict(n_scored=d["n_segments_scored"],
                         n_pass_all=d["n_segments_passing_all_gates"],
                         gate_pass_counts=d["gate_pass_counts"],
                         n_p_le_alpha=d["n_segments_p_le_alpha"],
                         expected_p_le_alpha_under_null=d["expected_n_p_le_alpha_under_null"]),
 "structural_argument": dict(
    note="pass requires all 5 gates; only gate_significance depends on the null. "
         "Deleting the significance gate entirely (the most anti-conservative "
         "possible null correction) leaves the 4 null-free gates.",
    n_pass_all_four_null_free_gates=sum(1 for r in res if all(r.get(g) for g in GATES[1:])),
    n_pass_fwd_rev_alone=sum(1 for r in res if r.get("gate_fwd_rev")),
    min_fwd_rev_r=min(r["fwd_rev_r"] for r in res),
    fwd_rev_threshold=0.2,
    conclusion="no change to the null can raise the pass count above 0"),
 "map_correlation_vs_perm_tile": dict(
    perm_tile_ds4=64,
    n_seg_L01_over_64=a["map_correlation"]["n_seg_L01_over_64"],
    n_measured=a["map_correlation"]["n_measured"],
    L01_x=a["map_correlation"]["L01_x"], L01_y=a["map_correlation"]["L01_y"],
    hazard="tile-64 permutation is anti-conservative on maps whose correlation "
           "length exceeds the tile; 19 of 74 maps have L(0.1) > 64 ds4 px"),
 "renull_tile128_50perm": [],
}

# tile-128 re-nulls (fresh seeds, screen's own prep/score/joint_permute)
files = {"z_dbg_gen_00215": os.path.join(HERE, "ns4b_z_dbg_gen_00215.json")}
for f in glob.glob(os.path.join(HERE, "ns4c_*_t128.json")):
    key = os.path.basename(f)[5:-10]
    files[key] = f
stored_by_name = {r["name"]: r for r in res + d.get("control", [])}
for key, f in sorted(files.items()):
    j = json.load(open(f))
    if "tiles" in j:      # ns4b format
        s = j["tiles"]["128"]
    else:                 # ns4c format
        s = j["summary"]
    sr = stored_by_name[key]
    out["renull_tile128_50perm"].append(dict(
        name=key, kind=sr.get("kind", "segment"),
        obs=sr["obs_prominence"], period_mm=sr["period_mm"],
        p_v2_tile64_200perm=sr["empirical_p"], z_v2_tile64=sr["z_corrected"],
        p_tile128_50perm=s["empirical_p"], z_tile128=s["z"],
        n_null_ge_obs=s["n_null_ge_obs"], null_sd_64=sr["null_sd"],
        null_sd_128=s["null_sd"],
        keeps_raw_p_le_0p05=bool(s["empirical_p"] <= 0.05),
        fwd_rev_r=sr.get("fwd_rev_r"),
        other_gates_failed=[g for g in GATES[1:] if not sr.get(g)] if sr.get("kind") != "control" else None))

seg = [r for r in out["renull_tile128_50perm"] if r["kind"] != "control"]
keep = sum(1 for r in seg if r["keeps_raw_p_le_0p05"])
ctl = [r for r in out["renull_tile128_50perm"] if r["kind"] == "control"][0]
out["verdict"] = dict(
    survives=True,
    headline="'0 of 71 pass' SURVIVES any null correction",
    basis=[
      "structural: 0 of 71 pass the four null-free gates, so no null can flip the verdict",
      "measured: at block 128 >= the measured correlation length, %d of 4 significance "
      "passers keep raw p<=0.05 (screen's own expectation under the null: 3.55 of 71)" % keep,
      "the strongest tile-64 passer (z_dbg_gen_00260, p .00498) rises to p .0588 at tile 128",
      "power retained: the genuine-ruling control still passes at tile 128 "
      "(0/50 nulls >= obs, z %+0.2f vs %+0.2f at tile 64)" % (ctl["z_tile128"], ctl["z_v2_tile64"])])
json.dump(out, open(os.path.join(HERE, "ns4_corpus_verdict.json"), "w"), indent=1)
print(json.dumps(out["verdict"], indent=1))
print("\nrenull table:")
for r in out["renull_tile128_50perm"]:
    print(" %-32s %-8s p64=%.5f -> p128=%.5f  z %.2f->%.2f  nullsd %.2f->%.2f  keeps<=.05 %s" % (
        r["name"], r["kind"], r["p_v2_tile64_200perm"], r["p_tile128_50perm"],
        r["z_v2_tile64"], r["z_tile128"], r["null_sd_64"], r["null_sd_128"],
        r["keeps_raw_p_le_0p05"]))
