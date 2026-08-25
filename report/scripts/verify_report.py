"""Verify report integrity: figure links resolve, headline numbers match primaries."""
import json
import os
import re

R = r"C:\Users\benbl\Desktop\Vsuvious\trackD\report"
T = r"C:\Users\benbl\Desktop\Vsuvious\trackD"
problems, checks = [], []

# --- 1. figure links resolve from each file's own directory
for root, _, files in os.walk(R):
    for fn in files:
        if not fn.endswith(".md"):
            continue
        p = os.path.join(root, fn)
        txt = open(p, encoding="utf-8").read()
        for m in re.finditer(r"\]\(([^)]+\.png)\)", txt):
            target = os.path.normpath(os.path.join(root, m.group(1)))
            if not os.path.exists(target):
                problems.append(f"BROKEN FIG: {os.path.relpath(p, R)} -> {m.group(1)}")
        for m in re.finditer(r"\]\((sections/[^)]+\.md|REPRODUCIBILITY\.md|[A-Z_]+\.md)\)", txt):
            target = os.path.normpath(os.path.join(root, m.group(1)))
            if not os.path.exists(target):
                problems.append(f"BROKEN LINK: {os.path.relpath(p, R)} -> {m.group(1)}")
checks.append("figure/link resolution")

# --- 2. index numbers vs primary JSONs
idx = {}
d = os.path.join(T, "out", "k2b_index")
for fn in os.listdir(d):
    if fn.startswith("PHerc") and fn.endswith(".json"):
        j = json.load(open(os.path.join(d, fn)))
        if "snr_q025_med_iqr" in j:
            idx[fn[:-5]] = j
expect = {"PHerc0813": 159.59, "PHerc0139": 115.54, "PHerc0125": 114.225,
          "PHerc1545": 112.23, "PHerc0211": 106.64, "PHerc0191": 99.565,
          "PHerc0358": 91.82, "PHerc1203": 87.24, "PHerc0826": 72.2,
          "PHerc1218": 24.35, "PHerc0268": 23.6, "PHerc0257": 22.71,
          "PHerc0800": 20.08, "PHerc1447": 8.45}
for k, v in expect.items():
    got = idx.get(k, {}).get("snr_q025_med_iqr", [None])[0]
    if got is None or abs(got - v) > 0.06:
        problems.append(f"INDEX MISMATCH {k}: report {v} vs json {got}")
checks.append(f"index SNR values ({len(expect)} scrolls)")

# tier gap claim: 3.0x between 24.4 and 72.2
gap = expect["PHerc0826"] / expect["PHerc1218"]
if not (2.9 < gap < 3.1):
    problems.append(f"TIER GAP: report says 3.0x, computed {gap:.2f}")
checks.append(f"tier gap = {gap:.2f}x")

# --- 3. w035 control AUCs
s = json.load(open(os.path.join(T, "out", "ink9um_w035_scores.json")))
pairs = [("w035_seed42-075000.tif", 0.9991), ("w035_seed43-075000.tif", 0.9982),
         ("w035_seed42-075000_reverse.tif", 0.5123), ("w035_seed43-075000_reverse.tif", 0.5844)]
for fn, v in pairs:
    got = s.get(fn, {}).get("auc")
    if got is None or abs(got - v) > 0.0002:
        problems.append(f"AUC MISMATCH {fn}: report {v} vs json {got}")
checks.append("w035 control AUCs (4 runs)")

# --- 4. 1203 firing fractions bracket claim (0.097-0.158)
st = json.load(open(os.path.join(T, "out", "ink9um_1203_stats.json")))
fr = [v["frac_gt_half"] for v in st.values()]
if not (abs(min(fr) - 0.0965) < 0.002 and abs(max(fr) - 0.1581) < 0.002):
    problems.append(f"1203 FIRING RANGE: report 0.097-0.158 vs actual {min(fr):.4f}-{max(fr):.4f}")
checks.append(f"1203 firing range {min(fr):.3f}-{max(fr):.3f} over {len(fr)} maps")

# --- 5. K3 stage-2 stats
k3 = json.load(open(os.path.join(T, "out", "k3_s2_stats.json")))
if abs(k3.get("ratio_detrended_sigma", 0) - 0.1259) > 1e-4:
    problems.append(f"K3 SIGMA: report 0.1259 vs {k3.get('ratio_detrended_sigma')}")
if k3.get("low_channel_voxels") != 2:
    problems.append(f"K3 LOW: report 2 vs {k3.get('low_channel_voxels')}")
if k3.get("high_channel_voxels") != 51343:
    problems.append(f"K3 HIGH: report 51,343 vs {k3.get('high_channel_voxels')}")
checks.append("K3 sigma + both channel counts")

# --- 6. K1b held-out values
k1b = json.load(open(os.path.join(T, "out", "k1b_depth_validation.json")))
for seg, v in [("w035", 0.5717), ("w039", 0.5132), ("w040", 0.5012), ("w041", 0.4657)]:
    got = k1b.get(seg, {}).get("auc_depth_contrast")
    if got is None or abs(got - v) > 0.0002:
        problems.append(f"K1b MISMATCH {seg}: report {v} vs {got}")
held = [k1b[s]["auc_depth_contrast"] for s in ("w039", "w040", "w041")]
mean_held = sum(held) / 3
if abs(mean_held - 0.4934) > 0.0005:
    problems.append(f"K1b held-out mean: report 0.4934 vs {mean_held:.4f}")
checks.append(f"K1b 4 segments + held-out mean {mean_held:.4f}")

# --- 7. corpus survey headline numbers (§2.4) vs survey_all.json / corpus_analysis.json
sv = json.load(open(os.path.join(T, "out", "survey", "survey_all.json")))
if len(sv) != 80:
    problems.append(f"CORPUS ROWS: report 80 vs {len(sv)}")
n_err = sum(1 for r in sv if "error" in r or not r.get("forward"))
n_trip = sum(len(r["forward"].get("tripwire_hits") or []) +
             len(r["reverse"].get("tripwire_hits") or []) for r in sv if r.get("forward"))
n_infer = sum(1 for r in sv if r.get("forward")) + sum(1 for r in sv if r.get("reverse"))
if n_err or n_trip or n_infer != 160:
    problems.append(f"CORPUS 80/80 CLAIM: errors {n_err}, tripwire hits {n_trip}, inferences {n_infer}")
counts = {}
for r in sv:
    counts[r["scroll"]] = counts.get(r["scroll"], 0) + 1
if counts != {"PHerc1203": 22, "PHerc1447": 52, "PHerc0800": 6}:
    problems.append(f"CORPUS COMPOSITION: report 22/52/6 vs {counts}")
checks.append(f"corpus survey {len(sv)}/80 rows, {n_infer} inferences, "
              f"{n_err} errors, {n_trip} tripwire hits")

# --- 7b. corpus screen v2 (§2.4) — the protocol the report actually states.
# v1 (corpus_analysis.json) is superseded and is deliberately NOT asserted here:
# its ordering was shown to be substantially permutation noise (ledger row 12).
v2 = json.load(open(os.path.join(T, "out", "survey", "corpus_analysis_v2.json")))
scored = [r for r in v2["results"] if r.get("status") == "scored"]
zs = sorted(r["z_corrected"] for r in scored)
med = zs[len(zs) // 2] if len(zs) % 2 else (zs[len(zs) // 2 - 1] + zs[len(zs) // 2]) / 2
if len(zs) != 71 or abs(zs[0] + 1.20) > 0.005 or abs(zs[-1] - 4.64) > 0.005 or abs(med + 0.09) > 0.005:
    problems.append(f"CORPUS V2 Z: report 71 segs, -1.20..+4.64 median -0.09 vs "
                    f"{len(zs)} segs, {zs[0]:+.2f}..{zs[-1]:+.2f} median {med:+.2f}")
if v2["n_segments_passing_all_gates"] != 0:
    problems.append(f"CORPUS V2 SURVIVORS: report 0 vs {v2['n_segments_passing_all_gates']}")
if v2["n_segments_scored"] != 71 or v2["n_segments_skipped"] != 4:
    problems.append(f"CORPUS V2 COVERAGE: report 71 scored / 4 skipped vs "
                    f"{v2['n_segments_scored']} / {v2['n_segments_skipped']}")
gp = v2["gate_pass_counts"]
for g, want in [("gate_significance", 4), ("gate_cycles", 19), ("gate_autocorr", 23),
                ("gate_band_bin", 23), ("gate_fwd_rev", 0)]:
    if gp.get(g) != want:
        problems.append(f"CORPUS V2 GATE {g}: report {want} vs {gp.get(g)}")
if abs(v2["expected_n_p_le_alpha_under_null"] - 3.55) > 0.01:
    problems.append(f"CORPUS V2 EXPECTED HITS: report 3.55 vs {v2['expected_n_p_le_alpha_under_null']}")
if abs(min(r["holm_p"] for r in scored) - 0.354) > 0.002:
    problems.append(f"CORPUS V2 MIN HOLM: report 0.354 vs {min(r['holm_p'] for r in scored):.3f}")
cs_p = [r["constrained_search"]["empirical_p"] for r in scored
        if r.get("constrained_search") and "empirical_p" in r["constrained_search"]]
if cs_p and abs(min(cs_p) - 0.0597) > 0.002:
    problems.append(f"CORPUS V2 CONSTRAINED BEST p: report 0.0597 vs {min(cs_p):.4f}")
rr = [abs(r["fwd_rev_r"]) for r in scored if r.get("fwd_rev_r") is not None]
if abs(min(rr) - 0.222) > 0.002:
    problems.append(f"CORPUS V2 MIN FWD/REV r: report 0.222 vs {min(rr):.3f}")
if sum(1 for r in scored if r["peak_bin_index"] <= 1) != 48:
    problems.append("CORPUS V2 BAND-EDGE: report 48/71 in the 2 lowest bins vs "
                    f"{sum(1 for r in scored if r['peak_bin_index'] <= 1)}")
ctl = v2["control"][0]
if (abs(ctl["z_corrected"] - 16.259) > 0.01 or abs(ctl["empirical_p"] - 0.00498) > 1e-5
        or abs(ctl["period_mm"] - 4.678) > 0.002 or ctl["n_null_ge_obs"] != 0
        or not ctl["constrained_search"]["passes_all_gates"]):
    problems.append(f"CORPUS V2 CONTROL: report z=+16.26, p=0.00498, 4.678 mm, 0/200, 5/5 gates vs "
                    f"z={ctl['z_corrected']:.2f}, p={ctl['empirical_p']}, {ctl['period_mm']} mm, "
                    f"{ctl['n_null_ge_obs']}/200")
checks.append(f"corpus screen v2: {len(zs)} scored, z {zs[0]:+.2f}..{zs[-1]:+.2f} (median {med:+.2f}), "
              f"0 pass all gates, control z={ctl['z_corrected']:.2f} p={ctl['empirical_p']}")

# --- 7c. the report must NOT still be quoting the superseded v1 corpus numbers
stale = {"+5.94": "v1 corpus max", "25.6…+34.1": "v1 control anchor (never computed by v1)",
         "median −0.27": "v1 corpus median", "69 scorable": "v1 scored count"}
for root, _, files in os.walk(R):
    for fn in files:
        if not fn.endswith(".md"):
            continue
        p = os.path.join(root, fn)
        txt = open(p, encoding="utf-8").read()
        for needle, why in stale.items():
            for line in txt.splitlines():
                if needle not in line:
                    continue
                # allowed only where the line explicitly frames it as superseded
                if any(w in line.lower() for w in
                       ("v1", "first pass", "superseded", "earlier draft", "screen as run",
                        "16 perm", "discard")):
                    continue
                problems.append(f"STALE V1 NUMBER in {os.path.relpath(p, R)}: {needle} ({why})")
checks.append(f"no unmarked v1 corpus numbers ({len(stale)} patterns swept)")

# --- 8. PHerc0813 growth + QC (§2.9)
qc = json.load(open(os.path.join(T, "hunt", "pherc0813_mesh_qc.json")))
good = [r for r in qc if r.get("surface_zero_frac", 1) < 0.2]
tot = sum(r["area_cm2"] for r in qc)
gtot = sum(r["area_cm2"] for r in good)
if len(qc) != 8 or abs(tot - 99.9) > 0.15 or len(good) != 5 or abs(gtot - 68.5) > 0.15:
    problems.append(f"0813: report 8 patches / 99.9 cm2 / 5 on material / 68.5 cm2 vs "
                    f"{len(qc)} / {tot:.1f} / {len(good)} / {gtot:.1f}")
mods = [r["contrast"] / r["surface_mean_DN"] for r in good]
if abs(min(mods) - 0.036) > 0.001 or abs(max(mods) - 0.074) > 0.001:
    problems.append(f"0813 MODULATION: report 0.036-0.074 vs {min(mods):.3f}-{max(mods):.3f}")
ctrl = json.load(open(os.path.join(T, "hunt", "control_profile.json")))["w035_win24"]
cmod = (max(ctrl) - min(ctrl)) / (sum(ctrl) / len(ctrl))
if abs(cmod - 0.443) > 0.002:
    problems.append(f"0813 CONTROL MODULATION: report 0.443 vs {cmod:.3f}")
checks.append(f"PHerc0813: {len(qc)} patches, {tot:.1f} cm2, {len(good)} on material "
              f"({gtot:.1f} cm2), modulation {min(mods):.3f}-{max(mods):.3f} vs control {cmod:.3f}")

# --- 9. separability axis (§1.8) vs out/k2c_separability/
k2c = json.load(open(os.path.join(T, "out", "k2c_separability", "k2c_analysis.json")))
sc = k2c["scrolls"]
if len(sc) != 14:
    problems.append(f"K2C COVERAGE: report 14 scrolls vs {len(sc)}")
expect_sep = {"PHerc0139": 0.744, "PHerc0358": 0.713, "PHerc0813": 0.662, "PHerc0826": 0.640, "PHerc1447": 0.610, "PHerc0800": 0.563, "PHerc1203": 0.555, "PHerc1545": 0.549, "PHerc0211": 0.543, "PHerc1218": 0.526, "PHerc0191": 0.525, "PHerc0125": 0.424, "PHerc0257": 0.385, "PHerc0268": 0.317}
for k, v in expect_sep.items():
    got = sc.get(k, {}).get("sep_med")
    if got is None or abs(got - v) > 0.0006:
        problems.append(f"SEPARABILITY {k}: report {v} vs {got}")
# the calibrator must rank first — that is the axis's whole validation
rank = sorted(sc, key=lambda k: -sc[k]["sep_med"])
if rank[0] != "PHerc0139":
    problems.append(f"SEPARABILITY ANCHOR: report PHerc0139 ranks 1st vs {rank[0]}")
if abs(k2c["sep_vs_snr_spearman"]["rho"] - 0.266) > 0.002:
    problems.append(f"SEP-vs-SNR rho: report +0.266 vs {k2c['sep_vs_snr_spearman']['rho']:+.3f}")
pb = k2c["picker_bias"]
if (pb["n_scrolls_random_higher"] != 14 or abs(pb["median_ratio"] - 3.00) > 0.02
        or pb["mannwhitney_p"] > 1e-20):
    problems.append(f"PICKER BIAS: report 14/14, 3.00x, p<1e-20 vs "
                    f"{pb['n_scrolls_random_higher']}/14, {pb['median_ratio']:.2f}x, p={pb['mannwhitney_p']:.2g}")
if abs(pb["random_med"] - 0.564) > 0.002 or abs(pb["picked_med"] - 0.168) > 0.002:
    problems.append(f"PICKER BIAS MEDIANS: report 0.564 / 0.168 vs "
                    f"{pb['random_med']:.3f} / {pb['picked_med']:.3f}")
flo = json.load(open(os.path.join(T, "out", "k2c_separability", "isotropy_floor.json")))
if abs(flo["air_median"] - 0.105) > 0.001 or len(flo["air"]) != 28:
    problems.append(f"ISOTROPIC FLOOR: report 0.105 over 28 air windows vs "
                    f"{flo['air_median']:.3f} over {len(flo['air'])}")
sen = json.load(open(os.path.join(T, "out", "k2c_separability", "sensitivity.json")))
ref = sen["B32_s1.0"]
import itertools
rhos = []
for k, per in sen.items():
    if k == "B32_s1.0":
        continue
    common = [s_ for s_ in ref if s_ in per]
    if len(common) > 2:
        a_ = [ref[s_] for s_ in common]
        b_ = [per[s_] for s_ in common]
        n_ = len(a_)
        ra = [sorted(a_).index(x) for x in a_]
        rb = [sorted(b_).index(x) for x in b_]
        dsq = sum((x - y) ** 2 for x, y in zip(ra, rb))
        rhos.append(1 - 6 * dsq / (n_ * (n_ ** 2 - 1)))
if rhos and (min(rhos) < 0.97 or len(ref) != 14):
    problems.append(f"SENSITIVITY: report rho +0.978..+0.996 over 14 scrolls vs "
                    f"min {min(rhos):+.3f} over {len(ref)}")
checks.append(f"separability parameter sensitivity: min Spearman rho {min(rhos):+.3f} over {len(ref)} scrolls")

checks.append(f"separability axis: 14 scrolls, calibrator ranks 1st ({sc['PHerc0139']['sep_med']:.3f}), "
              f"rho vs SNR +{k2c['sep_vs_snr_spearman']['rho']:.3f}, picker bias "
              f"{pb['median_ratio']:.2f}x in {pb['n_scrolls_random_higher']}/14, floor {flo['air_median']:.3f}")

# --- 10. PHerc0813 mesh alignment (§2.9.1)
al = json.load(open(os.path.join(T, "out", "k2c_separability", "pherc0813_mesh_alignment.json")))
pu = json.load(open(os.path.join(T, "out", "k2c_separability", "published_mesh_alignment.json")))
if abs(al["median_angle_deg"] - 68.1) > 0.1 or al["n_within_30deg"] != 0 or len(al["meshes"]) != 8:
    problems.append(f"0813 ALIGNMENT: report 8 meshes, 68.1 deg, 0 within 30 vs "
                    f"{len(al['meshes'])}, {al['median_angle_deg']:.1f}, {al['n_within_30deg']}")
if abs(pu["median_angle_deg"] - 13.1) > 0.1 or pu["n_within_30deg"] != 7 or len(pu["meshes"]) != 9:
    problems.append(f"PUBLISHED ALIGNMENT: report 9 meshes, 13.1 deg, 7 within 30 vs "
                    f"{len(pu['meshes'])}, {pu['median_angle_deg']:.1f}, {pu['n_within_30deg']}")
mb = json.load(open(os.path.join(T, "out", "k2c_separability", "mainbuild_nz.json")))["mainbuild_abs_nz"]
import statistics as _stb
if len(mb) != 8 or abs(_stb.median(list(mb.values())) - 0.236) > 0.01 or max(mb.values()) > 0.7:
    problems.append(f"MAINBUILD: report 8 meshes, median 0.236, 0 flat vs "
                    f"{len(mb)}, {_stb.median(list(mb.values())):.3f}, max {max(mb.values()):.3f}")
checks.append(f"current-build re-grow: |n_z| median {_stb.median(list(mb.values())):.3f} (published 0.223) — cause resolved: stale image")

mt = json.load(open(os.path.join(T, "out", "k2c_separability", "modetest_nz.json")))
import statistics as _stt
rv = list(mt["R_random_seed"].values()); ev = list(mt["E_explicit_seed"].values())
if (len(rv) != 8 or len(ev) != 8 or abs(_stt.median(rv) - 0.979) > 0.01
        or abs(_stt.median(ev) - 0.989) > 0.01 or min(rv + ev) <= 0.7):
    problems.append(f"MODETEST: report 8+8, medians 0.979/0.989, 16/16 > 0.7 vs "
                    f"{len(rv)}+{len(ev)}, {_stt.median(rv):.3f}/{_stt.median(ev):.3f}, min {min(rv+ev):.3f}")
checks.append(f"mode A/B: random_seed {_stt.median(rv):.3f} vs explicit {_stt.median(ev):.3f}, "
              f"16/16 flat — seeding exonerated")

cal = json.load(open(os.path.join(T, "out", "k2c_separability", "corpus_alignment_local.json")))
cs = cal["summary"]
if (cs["n_locally_measured"] != 35 or abs(cs["median_local_deg"] - 5.47) > 0.05
        or cs["n_local_ge_45"] != 5):
    problems.append(f"LOCAL ALIGNMENT: report 35 measured / median 5.5 / 5 at >=45 vs "
                    f"{cs['n_locally_measured']} / {cs['median_local_deg']:.1f} / {cs['n_local_ge_45']}")
exon = [r for r in cal["segments"] if r.get("status") == "ok"
        and r["angle_global_deg"] >= 45 and r["angle_local_deg"] < 45]
if len(exon) != 1 or exon[0]["scroll"] != "PHerc1203":
    problems.append(f"LOCAL ALIGNMENT EXONERATION: report 1 (PHerc1203) vs {[(r['scroll']) for r in exon]}")
checks.append(f"local alignment recompute: {cs['n_locally_measured']} measured, median "
              f"{cs['median_local_deg']:.1f} deg, 5 confirmed >=45, 1 exonerated (PHerc1203)")

checks.append(f"PHerc0813 mesh alignment: ours {al['median_angle_deg']:.1f}° ({al['n_within_30deg']}/8 "
              f"within 30°) vs published {pu['median_angle_deg']:.1f}° ({pu['n_within_30deg']}/9)")

# --- 11. §2.9's lamella-modulation claim must now be paired with its §2.9.1 diagnosis
inst = open(os.path.join(R, "sections", "02_instrument.md"), encoding="utf-8").read()
if "0.036–0.074" in inst and "2.9.1" not in inst:
    problems.append("SECTION 2.9: quotes the lamella modulation without the §2.9.1 alignment diagnosis")
if "unresolvable" in inst.lower() and "not" not in inst.lower().split("unresolvable")[0][-80:]:
    pass  # phrasing check only; the substantive assertion is the alignment numbers above
checks.append("§2.9 modulation claim carries its §2.9.1 diagnosis")

# --- 12. corpus-wide mesh alignment audit (§2.7 point 5)
ca = json.load(open(os.path.join(T, "out", "k2c_separability", "corpus_alignment.json")))
csegs = ca["segments"]
meas = [r for r in csegs if r.get("angle_deg") is not None]
outside = [r for r in csegs if r.get("status") == "cube mostly outside scanned volume"]
if len(csegs) != 80 or len(meas) != 56 or len(outside) != 24:
    problems.append(f"CORPUS ALIGN COVERAGE: report 80 rows / 56 measured / 24 outside vs "
                    f"{len(csegs)} / {len(meas)} / {len(outside)}")
if any(r["scroll"] != "PHerc1447" for r in outside):
    problems.append("CORPUS ALIGN: report says all 24 outside-volume rows are PHerc1447")
import statistics as _st
byscroll = {}
for r in meas:
    byscroll.setdefault(r["scroll"], []).append(r["angle_deg"])
for sc, med, w30, n in [("PHerc0800", 3.0, 6, 6), ("PHerc1203", 10.3, 21, 22), ("PHerc1447", 56.6, 9, 28)]:
    a_ = byscroll.get(sc, [])
    if len(a_) != n or abs(_st.median(a_) - med) > 0.1 or sum(1 for x in a_ if x < 30) != w30:
        problems.append(f"CORPUS ALIGN {sc}: report n={n} median={med} within30={w30} vs "
                        f"n={len(a_)} median={_st.median(a_) if a_ else float('nan'):.1f} "
                        f"within30={sum(1 for x in a_ if x < 30)}")
dbg = [r["angle_deg"] for r in meas if "z_dbg_gen" in r["name"]]
cur = [r["angle_deg"] for r in meas if "z_dbg_gen" not in r["name"]]
if abs(_st.median(dbg) - 65.8) > 0.1 or abs(_st.median(cur) - 11.1) > 0.1:
    problems.append(f"CORPUS ALIGN dumps/curated: report 65.8/11.1 vs "
                    f"{_st.median(dbg):.1f}/{_st.median(cur):.1f}")
if sum(1 for r in meas if r["angle_deg"] >= 45) != 19:
    problems.append(f"CORPUS ALIGN >=45: report 19 of 56 vs {sum(1 for r in meas if r['angle_deg'] >= 45)}")
checks.append(f"corpus alignment audit: {len(meas)}/80 measured, median "
              f"{_st.median([r['angle_deg'] for r in meas]):.1f} deg, "
              f"dumps {_st.median(dbg):.1f} vs curated {_st.median(cur):.1f}")


# --- addendum checks (A1-A4) ---
import json as _j
from pathlib import Path as _P
_TD = _P(R).parent
_g = _j.dumps(_j.load(open(_TD/'out'/'g1v2'/'G1V2_RESULTS.json', encoding='utf-8')))
for _needle in ('0.5775', '0.6907', '1.806'):
    if _needle not in _g: problems.append(f"ADDENDUM A1: {_needle} not in G1V2_RESULTS.json")
checks.append("addendum A1: ceiling 0.578/0.691 vox-equiv, max z 1.81 in g1v2 artifact")
_x = open(_TD/'out'/'xacq'/'corpus_summary.json', encoding='utf-8').read()
_c = open(_TD/'out'/'xacq'/'calibration_rows.json', encoding='utf-8').read()
if '25.09' not in _x and '25.1' not in _x: problems.append("ADDENDUM A3: L99 median 25.09 not in corpus_summary.json")
if '0.008' not in (_x + _c): problems.append("ADDENDUM A3: median gain 0.0080 not in xacq artifacts")
checks.append("addendum A3: L99 median ~25 and gain ~0.0080 in xacq artifacts")
_b = _j.load(open(_TD/'out'/'curve_audit'/'expA_baseline.json', encoding='utf-8'))
if abs(_b['auc_forward']-0.6925)>0.0005 or abs(_b['auc_reverse']-0.4477)>0.0005 or _b['gate_passed'] is not False:
    problems.append("ADDENDUM A4: expA_baseline.json does not match 0.6925/0.4477/gate-failed")
checks.append("addendum A4: fragment transfer 0.6925/0.4477, gate failed")
_n = open(_TD/'out'/'null_scaling'/'NULL_SCALING.md', encoding='utf-8').read()
for _needle in ('566', '2755'):
    if _needle not in _n: problems.append(f"ADDENDUM A2: {_needle} not in NULL_SCALING.md")
checks.append("addendum A2: n_eff 566 and sd 2755 in null-scaling writeup")

print("CHECKS RUN:")
for c in checks:
    print("  -", c)
print(f"\nPROBLEMS: {len(problems)}")
for p in problems:
    print("  !", p)
if not problems:
    print("  (none — all verified claims match their primary artifacts)")
