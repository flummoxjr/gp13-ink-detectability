"""Alignment gate for grown patches on any GP scroll (gate_0358.py, parametrised).

  python hunt/gate_patches.py --scroll PHerc0826 --paths <dir with auto_grown_*/> --seeds hunt/seeds_0826.json --out <json>

For each patch: |n_z|, the angle between the mesh normal and the local sheet normal
at its seed (256^3 CT cube, cached on D:), PASS at < 30 deg. Patches are matched to
seeds by order (the tracer names patches by wall-clock, seeds launch in bank order
in waves) AND cross-checked by bbox containment of the seed; a mismatch is reported.
"""
import argparse
import json
import os
import sys

import numpy as np

T = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, T)
from k2c_separability import open_level  # noqa: E402
from hunt.mesh_lamella_alignment import mesh_normal, sheet_normal  # noqa: E402
from hunt.corpus_alignment_local import local_normal  # noqa: E402  (Aug-25 lesson: curved sheets need the LOCAL normal)
import glob
import re


def pairing_from_logs(logs_dir, scroll_id):
    """seed index -> patch name, from the tracer's own 'saving <dir>' line in G_<id>_NN.log.
    Exact, unlike bbox containment (overlapping ~9 cm2 patches contain several seeds)."""
    m = {}
    for p in glob.glob(os.path.join(logs_dir, f"G_{scroll_id}_*.log")):
        k = int(re.search(r"G_\w+?_(\d+)\.log$", os.path.basename(p)).group(1))
        for line in open(p, errors="replace"):
            if line.startswith("saving "):
                m[os.path.basename(line.split()[-1].strip())] = k
    return m

VOL = {"PHerc0358": "20250821151737-9.362um-1.2m-113keV-masked.zarr",
       "PHerc0813": "20250821151723-9.362um-1.2m-113keV-masked.zarr",
       "PHerc0826": "20250821151701-9.362um-1.2m-113keV-masked.zarr"}
ROI = 256


def seed_for(meta, seeds):
    """Seed whose (x,y,z) lies inside the mesh bbox; None if ambiguous/absent."""
    bb = meta.get("bbox")
    if not bb:
        return None
    (x0, y0, z0), (x1, y1, z1) = bb
    inside = [i for i, s in enumerate(seeds) if x0 - 2 <= s["x"] <= x1 + 2 and y0 - 2 <= s["y"] <= y1 + 2 and z0 - 2 <= s["z"] <= z1 + 2]
    return inside[0] if len(inside) == 1 else (inside if inside else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scroll", required=True, choices=sorted(VOL))
    ap.add_argument("--paths", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--logs", default=None, help="dir with the tracer logs G_<id>_NN.log (exact seed pairing)")
    a = ap.parse_args()
    log_pairs = pairing_from_logs(a.logs, a.scroll[-4:]) if a.logs else {}
    cache = os.path.join(r"D:\vesuvius-data\trackD", f"gate_{a.scroll[-4:]}_seeds")
    os.makedirs(cache, exist_ok=True)
    seeds = json.load(open(a.seeds))
    meshes = sorted(d for d in os.listdir(a.paths) if os.path.isdir(os.path.join(a.paths, d)))
    print(f"{a.scroll}: {len(meshes)} meshes, {len(seeds)} seeds")
    z0 = open_level(a.scroll, VOL[a.scroll], 0)
    rows = []
    for dn in meshes:
        md = os.path.join(a.paths, dn)
        meta = json.load(open(os.path.join(md, "meta.json")))
        si = log_pairs.get(dn, seed_for(meta, seeds))
        rec = dict(name=dn, area_cm2=round(float(meta.get("area_cm2", 0)), 3), bbox=meta.get("bbox"),
                   pairing="log" if dn in log_pairs else "bbox")
        if not isinstance(si, int):
            rec.update(status=f"seed match {'ambiguous' if si else 'none'}", PASS=False, seed_candidates=si)
            rows.append(rec)
            print(f"  {dn[-9:]}  area={rec['area_cm2']}  seed-match {rec['status']}")
            continue
        sd = seeds[si]
        cp = os.path.join(cache, f"seed{si:02d}.npy")
        if os.path.exists(cp):
            cube = np.load(cp)
        else:
            o = (max(sd["z"] - ROI // 2, 0), max(sd["y"] - ROI // 2, 0), max(sd["x"] - ROI // 2, 0))
            cube = np.asarray(z0[o[0]:o[0] + ROI, o[1]:o[1] + ROI, o[2]:o[2] + ROI])
            np.save(cp, cube)
        o = (max(sd["z"] - ROI // 2, 0), max(sd["y"] - ROI // 2, 0), max(sd["x"] - ROI // 2, 0))
        mn, nv = mesh_normal(md)                 # whole-patch (global) normal, reported
        ln, nl = local_normal(md, o)             # normal from the vertices inside the seed cube -- the GATE
        sn = sheet_normal(cube)
        rec.update(seed_index=si, seed_xyz=[sd["x"], sd["y"], sd["z"]], separability=sd.get("separability"),
                   n_vertices=nv, n_local_vertices=nl)
        if sn is None or (mn is None and ln is None):
            rec.update(status="unmeasurable", PASS=False)
        else:
            if mn is not None:
                rec["abs_nz"] = round(abs(float(mn[0])), 3)
                rec["angle_global_deg"] = round(float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(mn, sn))))))), 1)
            if ln is not None:
                rec["angle_local_deg"] = round(float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(ln, sn))))))), 1)
                rec.update(angle_deg=rec["angle_local_deg"], PASS=bool(rec["angle_local_deg"] < 30.0), status="ok")
            else:
                rec.update(angle_deg=rec.get("angle_global_deg"), PASS=False, status="no local vertices in cube")
        rows.append(rec)
        print(f"  {dn[-9:]}  seed{si:02d} [{rec['pairing']}]  nz={rec.get('abs_nz', '-')}  local={rec.get('angle_local_deg', '-')}  "
              f"global={rec.get('angle_global_deg', '-')}  area={rec['area_cm2']} cm2  {'PASS' if rec.get('PASS') else 'fail'}")
    out = dict(scroll=a.scroll, gate="angle_local_deg < 30 (mesh normal from the vertices inside the 256^3 seed cube vs the cube's sheet normal; global whole-patch angle reported beside it)",
               reference="published GP meshes 13.1 deg median; stale-build failures 68.1; random 60",
               n_meshes=len(rows), n_pass=sum(1 for r in rows if r.get("PASS")),
               area_pass_cm2=round(sum(r["area_cm2"] for r in rows if r.get("PASS")), 2), patches=rows)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\nPASS: {out['n_pass']}/{len(rows)} ({out['area_pass_cm2']} cm2) -> {a.out}")


if __name__ == "__main__":
    main()
