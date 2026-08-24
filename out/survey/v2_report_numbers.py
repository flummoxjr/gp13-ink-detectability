"""Emit the PROTOCOL_V2.md tables straight from corpus_analysis_v2.json.

Keeps the write-up free of hand-transcribed numbers. Run after the v2 screen:
    python out/survey/v2_report_numbers.py
"""
import json
import os
import sys

import numpy as np

OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\out\survey"
FLAG = "z_dbg_gen_00166_inp_hr"
CTRL = "w035_CONTROL_strided"


def f(x, spec="{:.2f}", dash="-"):
    return dash if x is None else spec.format(x)


def main(path=None):
    d = json.load(open(path or os.path.join(OUT, "corpus_analysis_v2.json")))
    seg, ctrl = d["results"], {c["name"]: c for c in d["control"]}
    n = len(seg)
    fl = next((r for r in seg if r["name"] == FLAG), None)

    print("### config\n")
    print(json.dumps(d["config"], indent=1))
    print(f"\nscored {n}, skipped {d['n_segments_skipped']}, "
          f"pass all gates {d['n_segments_passing_all_gates']}")
    print(d["multiplicity_note"])

    print("\n### gate cascade\n")
    print("| gate | segments passing | of |")
    print("|---|---|---|")
    for g, c in d["gate_pass_counts"].items():
        print(f"| `{g}` | {c} | {n} |")
    print(f"| **all five** | **{d['n_segments_passing_all_gates']}** | {n} |")
    inner = ("gate_significance", "gate_cycles", "gate_autocorr", "gate_band_bin")
    n_inner = sum(1 for r in seg if all(r[g] for g in inner))
    print(f"| the four map-internal gates only (drop fwd/rev) | **{n_inner}** | {n} |")
    n_per = sum(1 for r in seg if all(r[g] for g in
                ("gate_cycles", "gate_autocorr", "gate_band_bin")))
    print(f"| the three periodicity gates only | {n_per} | {n} |")

    print("\n### corpus distributions\n")
    for k, spec in (("z_corrected", "{:+.2f}"), ("empirical_p", "{:.4f}"),
                    ("period_mm", "{:.2f}"), ("n_cycles", "{:.1f}"),
                    ("fwd_rev_r", "{:.3f}"), ("autocorr_2P", "{:+.3f}")):
        v = np.array([r[k] for r in seg if r.get(k) is not None], float)
        print(f"- {k}: min {spec.format(v.min())}  p25 {spec.format(np.percentile(v,25))}  "
              f"median {spec.format(np.median(v))}  p75 {spec.format(np.percentile(v,75))}  "
              f"max {spec.format(v.max())}")
    print(f"- peak_bin_index == 0 or 1 (band edge): "
          f"{sum(1 for r in seg if r['peak_bin_index'] < 2)}/{n}")
    print(f"- p <= 0.05: {d['n_segments_p_le_alpha']} "
          f"(expected {d['expected_n_p_le_alpha_under_null']})")
    print(f"- min holm_p: {min(r['holm_p'] for r in seg):.4f}")
    print(f"- constrained-search survivors: {d['n_segments_passing_constrained_search']}")
    cs = [r["constrained_search"].get("empirical_p") for r in seg
          if r["constrained_search"].get("empirical_p") is not None]
    if cs:
        print(f"- constrained-search p: min {min(cs):.4f}  median {np.median(cs):.3f}  "
              f"n<=0.05 {sum(1 for x in cs if x <= 0.05)}")

    print("\n### top 12 by empirical p\n")
    print("| # | scroll | segment | v1 z | v2 z | emp p | Holm p | period | cycles | "
          "bin | rho2P | fwd/rev r | gates |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(seg[:12], 1):
        print(f"| {i} | {r['scroll']} | `{r['name']}` | {f(r['v1_ruling_z'],'{:+.2f}')} | "
              f"{f(r['z_corrected'],'{:+.2f}')} | {r['empirical_p']:.4f} | "
              f"{r['holm_p']:.3f} | {f(r['period_mm'])} mm | {r['n_cycles']:.1f} | "
              f"{r['peak_bin_index']}/{r['band_bins']} | {f(r['autocorr_2P'],'{:+.3f}')} | "
              f"{f(r['fwd_rev_r'],'{:.3f}')} | {r['gates_passed']}/5 |")

    print("\n### control + flag, v2\n")
    print("| measurement | " + " | ".join(ctrl) + " | FLAG |")
    print("|---|" + "---|" * (len(ctrl) + 1))
    keys = [("obs_prominence", "{:.2f}"), ("null_mean", "{:.2f}"), ("null_sd", "{:.2f}"),
            ("null_max", "{:.2f}"), ("z_corrected", "{:+.2f}"), ("empirical_p", "{:.4f}"),
            ("n_null_ge_obs", "{:d}"), ("theta_deg", "{:.0f}"), ("period_mm", "{:.3f}"),
            ("profile_len_mm", "{:.1f}"), ("n_cycles", "{:.2f}"),
            ("peak_bin_index", "{:d}"), ("band_bins", "{:d}"),
            ("autocorr_1P", "{:+.3f}"), ("autocorr_2P", "{:+.3f}"),
            ("autocorr_3P", "{:+.3f}"), ("fwd_rev_r", "{:.4f}"),
            ("mask_frac_eroded", "{:.3f}"), ("gates_passed", "{:d}")]
    for k, spec in keys:
        row = [f(c.get(k), spec) for c in ctrl.values()]
        row.append(f(fl.get(k) if fl else None, spec))
        print(f"| {k} | " + " | ".join(row) + " |")
    for name, c in list(ctrl.items()) + ([("FLAG", fl)] if fl else []):
        print(f"\n{name} z_ci95 = {c.get('z_ci95')}  gates: " +
              ", ".join(f"{g.replace('gate_','')}={c.get(g)}" for g in
                        ("gate_significance", "gate_cycles", "gate_autocorr",
                         "gate_band_bin", "gate_fwd_rev")))
        print(f"  constrained_search: {json.dumps(c.get('constrained_search'))}")

    print("\n### skipped\n")
    for s in d["skipped"]:
        print(f"- `{s['name']}` -- {s['status']}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
