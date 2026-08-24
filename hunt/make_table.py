#!/usr/bin/env python
"""Emit the master comparison table as markdown."""
import json
import os

OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\hunt\out"

dp = {x["key"]: x for x in json.load(open(os.path.join(OUT, "depth_profiles.json")))}
gs = {x["key"]: x for x in json.load(open(os.path.join(OUT, "geom_stats.json")))}
pv = {x["key"]: x for x in json.load(open(os.path.join(OUT, "provenance.json")))}
isx = json.load(open(os.path.join(OUT, "interface_sharpness.json")))
per = json.load(open(os.path.join(OUT, "layer_period.json")))
vmp = os.path.join(OUT, "vertex_material.json")
vm = {x["key"]: x for x in json.load(open(vmp))} if os.path.exists(vmp) else {}

order = ["w035", "w032", "1203_r399", "1203_r460", "1203_r747",
         "1447_r222", "1447_r623", "1447_r914", "0800_r329", "0800_r522"]
LBL = {"w035": "**w035** (CONTROL)", "w032": "w032 (ctrl scroll)"}


def f(v, n=2, dash="—"):
    if v is None:
        return dash
    try:
        return f"{float(v):.{n}f}"
    except Exception:
        return str(v)


ROWS = [
    ("Scroll", lambda k: gs[k]["scroll"]),
    ("Provenance", lambda k: ("seeded from curated wrap, grown on **2 µm surface-prediction volume**"
                              if pv[k]["grown_on_volume"] != "(scroll volume, native)"
                              else "auto-grown in the **native 9 µm-class scroll volume**")),
    ("Growth generations", lambda k: str(pv[k].get("max_gen"))),
    ("ink_9um fwd-vs-rev map r", lambda k: f(dp[k].get("fwd_rev_r"), 3)),
    ("— GRID / SHAPE —", lambda k: ""),
    ("Area (mm²)", lambda k: f(gs[k]["area_mm2"], 0)),
    ("Grid coverage (valid frac)", lambda k: f(gs[k]["valid_frac"], 3)),
    ("Interior holes", lambda k: str(gs[k]["n_interior_holes"])),
    ("Grid edge-length CV", lambda k: f(gs[k]["edge_all_cv"], 4)),
    ("Edges > 2× median (tears)", lambda k: f(gs[k]["edge_row_frac_gt2x"], 5)),
    ("— CURVATURE / NORMALS —", lambda k: ""),
    ("Normal dispersion, median (°)", lambda k: f(gs[k]["normal_dispersion_med_deg"])),
    ("Normal dispersion, p99 (°)", lambda k: f(gs[k]["normal_dispersion_p99_deg"], 1)),
    ("Frac normals >30° apart", lambda k: f(gs[k]["normal_frac_gt30deg"], 5)),
    ("Frac antiparallel neighbours", lambda k: f(gs[k]["normal_frac_antiparallel"], 5)),
    ("Curvature, median (°/mm)", lambda k: f(gs[k]["curv_med_deg_per_mm"], 1)),
    ("Radius of curvature (mm)", lambda k: f(gs[k]["radius_curv_med_mm"], 2)),
    ("— IS THE MESH ON A SHEET? —", lambda k: ""),
    ("Frac mesh w/ no scanned material",
     lambda k: f(vm[k]["frac_no_material"], 3) if k in vm else "—"),
    ("Sheet-centre offset, median (vox)", lambda k: f(dp[k]["tile_centroid_off_med_vox"])),
    ("Sheet-centre offset IQR (vox)", lambda k: f(dp[k]["tile_centroid_off_spread_vox"])),
    ("Per-point centroid within 2 vox", lambda k: f(dp[k]["frac_centroid_within_2vox"], 3)),
    ("— SHEET CONTRAST —", lambda k: ""),
    ("Sheet modulation (max−min)/mean", lambda k: f(dp[k]["tile_modulation_med"], 3)),
    ("Peak / gap intensity ratio",
     lambda k: f(isx[k]["peak_over_gap"], 3) if isx.get(k) else "—"),
    ("Interface sharpness grad/mean (vox^-1)",
     lambda k: f(isx[k]["grad_norm"], 4) if isx.get(k) else "—"),
    ("Per-point contrast (max−min)/255", lambda k: f(dp[k]["profile_contrast_med"], 3)),
    ("Mean DN at offset 0", lambda k: f(dp[k]["intensity_at_0_med"], 0)),
    ("Sheet FWHM (µm)", lambda k: f(dp[k]["sheet_fwhm_med_um"], 0)),
    ("Lamella period (µm)", lambda k: (f(per[k] * (9.362 if gs[k]["scroll"] in ("PHerc0139", "PHerc1203") else 8.640), 0) if k in per else "—")),
    ("Tiles with ≥2 sheets in window", lambda k: f(dp[k]["tile_frac_two_sheets_in_window"], 2)),
]

hdr = ["Statistic"] + [LBL.get(k, k) for k in order]
lines = ["| " + " | ".join(hdr) + " |",
         "|" + "|".join(["---"] * len(hdr)) + "|"]
for name, fn in ROWS:
    lines.append("| " + " | ".join([f"**{name}**"] + [fn(k) for k in order]) + " |")
md = "\n".join(lines)
open(os.path.join(OUT, "master_table.md"), "w", encoding="utf-8").write(md)
print(md)
