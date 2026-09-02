"""Assemble segment release v1 for a scroll from a growth bundle + the alignment gate.

  python make_release.py --scroll PHerc0826 --paths <bundle>/paths_0826 --gate <gate.json> \
      --seeds hunt/seeds_0826.json --out hunt/pherc0826_release_v1

Copies every gate-PASS patch (x/y/z/generations tif + meta.json) into <out>/paths/<name>/,
writes <out>/catalogue.json (per-patch: name, area_cm2, bbox, seed, separability, |n_z|,
angle to local lamellae, PASS, tracer, volume, prediction store, params) and <out>/README.md.
FAIL patches are listed in the catalogue with their reason but not copied."""
import argparse
import json
import os
import shutil

T = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
VOL = {"PHerc0358": "20250821151737-9.362um-1.2m-113keV-masked.zarr",
       "PHerc0813": "20250821151723-9.362um-1.2m-113keV-masked.zarr",
       "PHerc0826": "20250821151701-9.362um-1.2m-113keV-masked.zarr"}
PRED = {"PHerc0358": "20250821151737-surface-20260413222639-surface-m7-L0-th0.2",
        "PHerc0813": "20250821151723-surface-20260413222639-surface-m7-L0-th0.2",
        "PHerc0826": "20250821151701-surface-20260413222639-surface-m7-L0-th0.2"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scroll", required=True, choices=sorted(VOL))
    ap.add_argument("--paths", required=True)
    ap.add_argument("--gate", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tracer", default="vc_grow_seg_from_seed, villa main (release vc-tracer-de3c2494; built from e2442b7)")
    a = ap.parse_args()
    gate = json.load(open(a.gate))
    seeds = json.load(open(a.seeds))
    os.makedirs(os.path.join(a.out, "paths"), exist_ok=True)
    cat = []
    for r in gate["patches"]:
        src = os.path.join(a.paths, r["name"])
        meta = json.load(open(os.path.join(src, "meta.json")))
        entry = dict(name=r["name"], PASS=bool(r.get("PASS")), status=r.get("status"),
                     area_cm2=r.get("area_cm2"), bbox_xyz=meta.get("bbox"), scale=meta.get("scale"),
                     seed_xyz=r.get("seed_xyz"), separability=r.get("separability"),
                     abs_nz=r.get("abs_nz"), angle_to_lamellae_deg=r.get("angle_deg"),
                     n_vertices=r.get("n_vertices"))
        if r.get("PASS"):
            dst = os.path.join(a.out, "paths", r["name"])
            if not os.path.exists(dst):
                shutil.copytree(src, dst)
            entry["files"] = sorted(os.listdir(dst))
        cat.append(entry)
    n_pass = sum(1 for c in cat if c["PASS"])
    area = round(sum(c["area_cm2"] or 0 for c in cat if c["PASS"]), 2)
    out = dict(scroll=a.scroll, release="v1", date="2026-09-02",
               volume=VOL[a.scroll], voxel_um=9.362, prediction_store=PRED[a.scroll],
               tracer=a.tracer, params=dict(mode="seed", generations=75, min_area_cm=0.3, thread_limit=1,
                                             use_cuda=False, voxelsize=9.362, normal_grid="released .normal-grids"),
               seed_selection="separability index top ROIs (out/k2c_separability), support-gated on an m7 sheet voxel (hunt/pick_seeds.py)",
               alignment_gate=gate.get("gate"), n_grown=len(cat), n_pass=n_pass, area_pass_cm2=area, patches=cat)
    json.dump(out, open(os.path.join(a.out, "catalogue.json"), "w"), indent=1)
    lines = [f"# {a.scroll} — GrowPatch surfaces, release v1 (2026-09-02)", "",
             f"{n_pass} of {len(cat)} grown patches pass the alignment gate (< 30° to the local lamellae at the seed): "
             f"**{area} cm²** of correctly-oriented surface on a scroll with no published segmentation.", "",
             f"- Volume: `{VOL[a.scroll]}` (9.362 µm); surface prediction: `{PRED[a.scroll]}` + its released normal grids.",
             f"- Tracer: {a.tracer}. Params: seed mode, 75 generations, min_area 0.3 cm², single-threaded per seed.",
             "- Seeds: the scroll's 24 separability-ranked ROIs (structure-tensor planarity), each moved to the nearest m7 sheet voxel",
             "  (support gate against the ~49 % phantom rate of m7 positives beyond CT support).",
             "- Gate: mesh normal vs local sheet normal from a 256³ CT cube at the seed; published GP meshes sit at 13.1° median.",
             "- Format: tifxyz (`x.tif`, `y.tif`, `z.tif`, `generations.tif`, `meta.json`), coordinates in the volume's voxel units.",
             "- Known caveat: every tracer mesh stores `scale` as float32 0.05 → villa's `Tifxyz.shape` gives a canvas 1 px short",
             "  (`issue_drafts/filing/tifxyz_fullres_shape_truncation.md`); render onto `round(h/scale)`.", "",
             "| patch | area cm² | angle ° | abs n_z | separability | PASS |", "|---|---|---|---|---|---|"]
    for c in cat:
        lines.append(f"| {c['name']} | {c['area_cm2']} | {c.get('angle_to_lamellae_deg', '-')} | {c.get('abs_nz', '-')} | "
                     f"{round(c['separability'], 3) if c.get('separability') else '-'} | {'PASS' if c['PASS'] else 'fail (' + str(c['status']) + ')'} |")
    open(os.path.join(a.out, "README.md"), "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
    print(f"{a.scroll}: {n_pass}/{len(cat)} PASS, {area} cm2 -> {a.out}")


if __name__ == "__main__":
    main()
