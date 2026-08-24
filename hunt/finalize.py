#!/usr/bin/env python
"""Merge the two depth-profile passes, rebuild the table + figure, and splice the
generated blocks into geometry_compare.md."""
import io
import json
import os
import subprocess
import sys

HUNT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\hunt"
OUT = os.path.join(HUNT, "out")
PY = r"C:\Users\benbl\Desktop\Vsuvious\.venv\Scripts\python.exe"

# --- merge pass 2 (local tracking) into pass 1 --------------------------
v1 = {x["key"]: x for x in json.load(open(os.path.join(OUT, "depth_profiles_v1.json")))}
v2 = {x["key"]: x for x in json.load(open(os.path.join(OUT, "depth_profiles.json")))}
merged = []
for k, a in v1.items():
    b = v2.get(k, {})
    if "error" not in b:
        for f in ("local_tracking_iqr_vox", "local_tracking_p10p90_vox",
                  "n_tiles_sheet_resolvable"):
            if f in b:
                a[f] = b[f]
    merged.append(a)
json.dump(merged, open(os.path.join(OUT, "depth_profiles_merged.json"), "w"), indent=1)
json.dump(merged, open(os.path.join(OUT, "depth_profiles.json"), "w"), indent=1)
print("merged local tracking for:",
      [x["key"] for x in merged if "local_tracking_iqr_vox" in x])

env = dict(os.environ, PYTHONIOENCODING="utf-8")
for s in ("make_table.py", "make_figures.py"):
    r = subprocess.run([PY, os.path.join(HUNT, s)], capture_output=True, text=True,
                       cwd=HUNT, env=env)
    print(s, "->", (r.stdout or r.stderr).strip().splitlines()[-1] if (r.stdout or r.stderr) else "ok")

# --- splice generated blocks into the markdown --------------------------
md_path = os.path.join(HUNT, "geometry_compare.md")
md = io.open(md_path, encoding="utf-8").read()

table = io.open(os.path.join(OUT, "master_table.md"), encoding="utf-8").read()
md = md.replace("<!--TABLE-->", table)

for x in merged:
    v = x.get("local_tracking_iqr_vox")
    md = md.replace(f"<!--TRK_{x['key']}-->", f"{v:.2f}" if v is not None else "—")

cm_path = os.path.join(OUT, "corpus_material.json")
if os.path.exists(cm_path):
    rows = [r for r in json.load(open(cm_path)) if "error" not in r]
    if rows:
        rows.sort(key=lambda r: -r["frac_no_material"])
        t = ["| Segment | Scroll | vertices | outside array | **no scanned material** | median vertex DN |",
             "|---|---|---|---|---|---|"]
        for r in rows:
            t.append(f"| `{r['name']}` | {r['scroll']} | {r['n_vertices']:,} | "
                     f"{r['frac_out_of_array']:.3f} | **{r['frac_no_material']:.3f}** | "
                     f"{r['vertex_lv3_med']:.1f} |")
        md = md.replace("<!--CORPUS_MATERIAL-->", "\n".join(t))

io.open(md_path, "w", encoding="utf-8").write(md)
print("spliced ->", md_path)
