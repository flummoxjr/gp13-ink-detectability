"""Alignment gate for the first PHerc0358 surfaces — the handoff artifact.

Contract with the render/battery session (vsuvious-f4): for each grown patch, report
|n_z|, the angle between the mesh normal and the local sheet normal at its seed, and a
PASS/FAIL at <30 deg. Only PASS patches proceed to render + ink_9um + battery.

Sheet normals come from 256^3 cubes of the masked CT volume centred on each seed
(fetched here, cached to D:). Writes:
  trackD/hunt/pherc0358_first_surfaces/alignment_gate.json
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from k2c_separability import open_level  # noqa: E402
from hunt.mesh_lamella_alignment import mesh_normal, sheet_normal  # noqa: E402

T = r"C:\Users\benbl\Desktop\Vsuvious\trackD"
D = os.path.join(T, "hunt", "pherc0358_first_surfaces")
CACHE = r"D:\vesuvius-data\trackD\g0358_seeds"
LONG_ID = "20250821151737-9.362um-1.2m-113keV-masked.zarr"
ROI = 256


def main():
    os.makedirs(CACHE, exist_ok=True)
    seeds = json.load(open(os.path.join(T, "hunt", "seeds_0358.json")))
    meshes = sorted(d for d in os.listdir(os.path.join(D, "paths_0358"))
                    if os.path.isdir(os.path.join(D, "paths_0358", d)))
    print(f"{len(meshes)} meshes, {len(seeds)} seeds")
    z0 = open_level("PHerc0358", LONG_ID, 0)

    rows = []
    for i, (dn, sd) in enumerate(zip(meshes, seeds)):
        cp = os.path.join(CACHE, f"seed{i:02d}.npy")
        if os.path.exists(cp):
            a = np.load(cp)
        else:
            o = (max(sd["z"] - ROI // 2, 0), max(sd["y"] - ROI // 2, 0), max(sd["x"] - ROI // 2, 0))
            a = np.asarray(z0[o[0]:o[0] + ROI, o[1]:o[1] + ROI, o[2]:o[2] + ROI])
            np.save(cp, a)
        mn, nv = mesh_normal(os.path.join(D, "paths_0358", dn))
        sn = sheet_normal(a)
        meta = json.load(open(os.path.join(D, "paths_0358", dn, "meta.json")))
        rec = dict(name=dn, seed_xyz=[sd["x"], sd["y"], sd["z"]],
                   separability=sd["separability"], n_vertices=nv,
                   area_cm2=round(float(meta.get("area_cm2", 0)), 3))
        if mn is None or sn is None:
            rec.update(status="unmeasurable", PASS=False)
        else:
            nz = abs(float(mn[0]))
            ang = float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(mn, sn)))))))
            rec.update(abs_nz=round(nz, 3), angle_deg=round(ang, 1),
                       PASS=bool(ang < 30.0), status="ok")
        rows.append(rec)
        print(f"  {dn[-9:]}  nz={rec.get('abs_nz','-')}  angle={rec.get('angle_deg','-')}  "
              f"area={rec['area_cm2']} cm2  {'PASS' if rec.get('PASS') else 'fail'}")

    out = dict(scroll="PHerc0358", gate="angle_deg < 30 (mesh normal vs local sheet normal at seed)",
               reference="published GP meshes 13.1 deg median; stale-build failures 68.1; random 60",
               n_pass=sum(1 for r in rows if r.get("PASS")), patches=rows)
    json.dump(out, open(os.path.join(D, "alignment_gate.json"), "w"), indent=1)
    print(f"\nPASS: {out['n_pass']}/{len(rows)}  -> alignment_gate.json (handoff to render/battery session)")


if __name__ == "__main__":
    main()
