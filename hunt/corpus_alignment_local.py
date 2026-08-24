"""Locally-restricted corpus alignment recompute — removes the curvature confound.

The first corpus audit (`corpus_alignment_audit.py`) compared each mesh's WHOLE-PATCH
average normal against the sheet normal in one central cube. That has a measured
confound: a large, strongly curved mesh scores high on curvature alone — the
human-traced w032 control reads 59.2 deg under that method, and nobody thinks w032
fails to follow its papyrus. Section 2.7 therefore carries "19 of 56 at >= 45 deg"
as an upper bound only.

This recompute restricts the mesh normal to ONLY the vertices inside the sampled
cube (the same repair validated on our PHerc0813 patches, where it made the result
slightly worse — 68.1 -> 72.9 deg — proving that failure real). Everything runs on
cubes and meshes already cached on disk: no downloads, no GPU.

Output: out/k2c_separability/corpus_alignment_local.json with, per segment, the
global angle (old), the local angle (new), and the vertex count backing the local
measurement. Segments with < 30 in-cube vertices are reported as not locally
measurable rather than guessed.
"""
import json
import os
import sys
import numpy as np
import tifffile
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hunt.mesh_lamella_alignment import mesh_normal, sheet_normal  # noqa: E402

T = r"C:\Users\benbl\Desktop\Vsuvious\trackD"
MESHCACHE = r"D:\vesuvius-data\trackD\corpus_meshes"
CUBECACHE = r"D:\vesuvius-data\trackD\corpus_cubes"
AUDIT = os.path.join(T, "out", "k2c_separability", "corpus_alignment.json")
OUT = os.path.join(T, "out", "k2c_separability", "corpus_alignment_local.json")
ROI = 256
MIN_VERTS = 30


def local_normal(d, origin_zyx):
    """Axial-mean normal from ONLY the valid mesh vertices inside the sampled cube."""
    x = tifffile.imread(os.path.join(d, "x.tif")).astype(np.float64)
    y = tifffile.imread(os.path.join(d, "y.tif")).astype(np.float64)
    z = tifffile.imread(os.path.join(d, "z.tif")).astype(np.float64)
    valid = (x >= 0) & (y >= 0) & (z >= 0) & ~((x == 0) & (y == 0) & (z == 0))
    core = ndi.binary_erosion(valid, structure=np.ones((3, 3), bool), border_value=0)
    o = origin_zyx
    inside = (core & (z >= o[0]) & (z < o[0] + ROI) & (y >= o[1]) & (y < o[1] + ROI)
              & (x >= o[2]) & (x < o[2] + ROI))
    if inside.sum() < MIN_VERTS:
        return None, int(inside.sum())
    tu = np.stack([np.gradient(a, axis=0) for a in (z, y, x)], -1)
    tv = np.stack([np.gradient(a, axis=1) for a in (z, y, x)], -1)
    n = np.cross(tu, tv)
    ok = inside & (np.linalg.norm(n, axis=-1) > 1e-6)
    if ok.sum() < MIN_VERTS:
        return None, int(ok.sum())
    nn = n[ok]
    nn = nn / np.linalg.norm(nn, axis=-1, keepdims=True)
    Tn = np.einsum('ij,ik->jk', nn, nn) / len(nn)
    w, V = np.linalg.eigh(Tn)
    return V[:, 2], int(ok.sum())


def main():
    audit = json.load(open(AUDIT))
    rows = []
    for r in audit["segments"]:
        if r.get("angle_deg") is None:
            continue
        name = r["name"]
        d = os.path.join(MESHCACHE, name)
        cp = os.path.join(CUBECACHE, f"{name}.npy")
        if not (os.path.isdir(d) and os.path.exists(cp)):
            rows.append(dict(name=name, scroll=r["scroll"], status="cache missing"))
            continue
        a = np.load(cp)
        # reconstruct the cube origin from the audited centre
        cz, cy, cx = r["centre_zyx"]
        o = (int(max(cz - ROI // 2, 0)), int(max(cy - ROI // 2, 0)), int(max(cx - ROI // 2, 0)))
        sn = sheet_normal(a)
        if sn is None:
            rows.append(dict(name=name, scroll=r["scroll"], status="no sheet normal"))
            continue
        ln, nv = local_normal(d, o)
        rec = dict(name=name, scroll=r["scroll"], angle_global_deg=r["angle_deg"],
                   n_local_vertices=nv, separability=r.get("separability"))
        if ln is None:
            rec["status"] = "too few in-cube vertices"
        else:
            rec["status"] = "ok"
            rec["angle_local_deg"] = float(np.degrees(
                np.arccos(min(1.0, abs(float(np.dot(ln, sn)))))))
        rows.append(rec)
        tag = f"{rec.get('angle_local_deg', float('nan')):5.1f}" if ln is not None else "  n/a"
        print(f"{r['scroll']:10} {name[:34]:34} global={r['angle_deg']:5.1f}  local={tag}  "
              f"verts={nv}", flush=True)

    ok = [r for r in rows if r.get("status") == "ok"]
    la = np.array([r["angle_local_deg"] for r in ok])
    ga = np.array([r["angle_global_deg"] for r in ok])
    summary = dict(
        n_locally_measured=len(ok),
        n_not_measurable=sum(1 for r in rows if r.get("status") != "ok"),
        median_local_deg=float(np.median(la)) if len(ok) else None,
        median_global_deg_same_set=float(np.median(ga)) if len(ok) else None,
        n_local_ge_45=int((la >= 45).sum()),
        n_global_ge_45_same_set=int((ga >= 45).sum()),
        n_local_within_30=int((la < 30).sum()),
        spearman_local_vs_global=float(__import__("scipy.stats", fromlist=["spearmanr"])
                                       .spearmanr(la, ga).statistic) if len(ok) > 3 else None,
    )
    json.dump({"summary": summary, "segments": rows}, open(OUT, "w"), indent=1)
    print("\n=== locally-restricted result ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
