"""Recompute every corpus-survey number quoted in report/sections/02_instrument.md.

Reads only primary artifacts:
  trackD/out/survey/survey_all.json        per-segment render+inference stats (80 rows)
  trackD/out/survey/maps_shard*/*.npy      150 retained 4x-downsampled prediction maps
  trackD/out/survey/corpus_analysis.json   per-segment ruling z vs block-permutation null
  trackD/hunt/pherc0813_mesh_qc.json       QC of the 8 new PHerc0813 surfaces
  trackD/hunt/control_profile.json         w035 depth profile under the identical QC code

Writes report/scripts/corpus_summary.json and prints the same numbers.
Usage: python report/scripts/corpus_summary.py
"""
import glob
import json
import os
import re

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SURVEY = os.path.join(ROOT, "out", "survey")
HUNT = os.path.join(ROOT, "hunt")

# Voxel pitch of the volume each scroll's segments were rendered against (um).
PX_UM = {"PHerc1203": 9.362, "PHerc1447": 8.640, "PHerc0800": 8.640}


def strip_ts(name):
    """PHerc1447 republishes segments/raw/<n>/ as segments/<timestamp>-<n>/."""
    return re.sub(r"^[0-9]{14}-", "", name)


def main():
    rows = json.load(open(os.path.join(SURVEY, "survey_all.json")))
    out = {"n_rows": len(rows)}

    # --- completion / error / tripwire accounting -------------------------------
    out["n_errors"] = sum(1 for r in rows if "error" in r or not r.get("forward"))
    out["n_tripwire_segments"] = sum(
        1 for r in rows
        if (r["forward"].get("tripwire_hits") or r["reverse"].get("tripwire_hits"))
    )
    out["n_tripwire_hits"] = sum(
        len(r["forward"].get("tripwire_hits") or []) + len(r["reverse"].get("tripwire_hits") or [])
        for r in rows
    )
    out["gpu_segment_hours"] = round(sum(r.get("secs", 0) for r in rows) / 3600.0, 2)

    # --- per-scroll composition and rendered area ------------------------------
    # Rendered-and-inferred area = canvas px * nonzero fraction * voxel pitch^2.
    # This is the area the model actually saw, not the mesh's meta.json area: it
    # runs ~15-20% above meta.json on well-covered meshes and far below it on
    # sparse ones (by design - a sparse render covered less papyrus).
    per_scroll, uniq_area = {}, {}
    for r in rows:
        f = r["forward"]
        h, w = f["canvas"]
        px_cm = PX_UM[r["scroll"]] * 1e-4
        area = h * w * px_cm * px_cm * f["nonzero_frac"]
        d = per_scroll.setdefault(r["scroll"], {"rows": 0, "area_cm2": 0.0, "names": set()})
        d["rows"] += 1
        d["area_cm2"] += area
        d["names"].add(strip_ts(r["name"]))
        uniq_area.setdefault((r["scroll"], strip_ts(r["name"])), area)
    for sc, d in per_scroll.items():
        sub = [r for r in rows if r["scroll"] == sc]
        d["unique_names"] = len(d.pop("names"))
        d["area_cm2"] = round(d["area_cm2"], 1)
        d["frac_gt_halfmax_fwd"] = [round(min(r["forward"]["frac_gt_half"] for r in sub), 4),
                                    round(max(r["forward"]["frac_gt_half"] for r in sub), 4)]
        d["fwd_rev_r"] = [round(min(r["fwd_rev_r"] for r in sub), 3),
                          round(max(r["fwd_rev_r"] for r in sub), 3)]
        d["nonzero_frac"] = [round(min(r["forward"]["nonzero_frac"] for r in sub), 3),
                             round(max(r["forward"]["nonzero_frac"] for r in sub), 3)]
    out["per_scroll"] = per_scroll
    out["area_cm2_all_rows"] = round(sum(v["area_cm2"] for v in per_scroll.values()), 1)
    out["area_cm2_dedup_by_name"] = round(sum(uniq_area.values()), 1)
    out["n_unique_by_name"] = len(uniq_area)
    out["n_zdbg_rows"] = sum(1 for r in rows if strip_ts(r["name"]).startswith("z_dbg"))
    out["n_zdbg_unique"] = len({strip_ts(r["name"]) for r in rows
                                if strip_ts(r["name"]).startswith("z_dbg")})

    # --- retained maps and analyzer admissibility ------------------------------
    fwd = {os.path.basename(p).replace("_forward_ds4.npy", "")
           for p in glob.glob(os.path.join(SURVEY, "maps_shard*", "*_forward_ds4.npy"))}
    rev = {os.path.basename(p).replace("_reverse_ds4.npy", "")
           for p in glob.glob(os.path.join(SURVEY, "maps_shard*", "*_reverse_ds4.npy"))}
    out["n_maps_retained"] = len(fwd) + len(rev)
    out["n_forward_maps"] = len(fwd)
    out["rows_without_retained_map"] = sorted({r["name"] for r in rows} - fwd)

    ca = json.load(open(os.path.join(SURVEY, "corpus_analysis.json")))
    scored = {r["name"] for r in ca["results"]}
    out["n_scored"] = ca["n_segments"]
    out["maps_not_scorable"] = sorted(fwd - scored)
    out["control_reference"] = ca["control_reference"]

    z = np.array([r["ruling_z"] for r in ca["results"]], float)
    out["ruling_z"] = {
        "n": int(z.size), "min": float(z.min()), "p25": float(np.percentile(z, 25)),
        "median": float(np.median(z)), "p75": float(np.percentile(z, 75)), "max": float(z.max()),
        "n_ge_2": int((z >= 2).sum()), "n_ge_3": int((z >= 3).sum()), "n_ge_5": int((z >= 5).sum()),
    }
    for sc in sorted({r["scroll"] for r in ca["results"]}):
        v = np.array([r["ruling_z"] for r in ca["results"] if r["scroll"] == sc], float)
        out["ruling_z"][sc] = {"n": int(v.size), "min": float(v.min()),
                               "median": float(np.median(v)), "max": float(v.max())}
    out["top_hits"] = [
        {k: r[k] for k in ("scroll", "name", "ruling_z", "period_mm", "theta_deg", "fwd_rev_r")}
        for r in sorted(ca["results"], key=lambda r: -r["ruling_z"])[:6]
    ]
    out["periods_of_z_ge_3_mm"] = sorted(r["period_mm"] for r in ca["results"] if r["ruling_z"] >= 3)

    # --- PHerc0813: the new surfaces -------------------------------------------
    qc = json.load(open(os.path.join(HUNT, "pherc0813_mesh_qc.json")))
    good = [r for r in qc if r.get("surface_zero_frac", 1) < 0.2]
    ctrl = json.load(open(os.path.join(HUNT, "control_profile.json")))["w035_win24"]
    out["pherc0813"] = {
        "n_patches": len(qc),
        "area_cm2_total": round(sum(r["area_cm2"] for r in qc), 1),
        "n_on_material": len(good),
        "area_cm2_on_material": round(sum(r["area_cm2"] for r in good), 1),
        "n_fully_empty": sum(1 for r in qc if r.get("surface_zero_frac", 0) >= 0.99),
        "contrast_DN_on_material": [min(r["contrast"] for r in good),
                                    max(r["contrast"] for r in good)],
        "modulation_on_material": sorted(round(r["contrast"] / r["surface_mean_DN"], 3)
                                         for r in good),
        "peak_offsets_on_material": sorted(r["peak_offset"] for r in good),
        "control_w035_win24_contrast_DN": round(max(ctrl) - min(ctrl), 1),
        "control_w035_win24_modulation": round((max(ctrl) - min(ctrl)) / float(np.mean(ctrl)), 3),
        "control_w035_win24_peak_offset": int(np.arange(-10, 11)[int(np.argmax(ctrl))]),
    }

    dst = os.path.join(os.path.dirname(__file__), "corpus_summary.json")
    json.dump(out, open(dst, "w"), indent=1)
    print(json.dumps(out, indent=1))
    print("\nwrote", dst)


if __name__ == "__main__":
    main()
