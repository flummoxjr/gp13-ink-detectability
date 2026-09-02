#!/usr/bin/env python
"""alignment_gate: does a tifxyz surface actually follow the lamellae it sits in?

Standalone quality gate for scroll surface patches. Takes a tifxyz mesh
directory (x.tif / y.tif / z.tif) and a CT volume, measures the angle between
the mesh's surface normal and the local sheet (lamella) normal of the material
the mesh sits in, and returns a verdict.

  angle near 0 deg   mesh follows the sheets            -> ALIGNED
  angle  >  30 deg   mesh does not track the lamella    -> REJECT
  angle near 90 deg  mesh cuts straight across sheets   -> REJECT (worst case)

Calibrated reference (PHerc0813, measured 2026-08 with this exact method):
  published direction-field GP meshes : median 13.1 deg, 7/9 within 30 deg
  random-orientation null             : median 60.0 deg
  our field-free auto-grown patches   : median 68.1 deg, 0/8 within 30 deg
The gate threshold (reject > 30 deg) separates those two populations cleanly.

Why this gate exists: every one of the 8 rejected PHerc0813 patches LOOKED
healthy -- right surface area, ~99.99% valid vertices, non-zero DN sampled
everywhere. Only the angle showed they were oblique/perpendicular to the
lamellae, which is why their depth profiles had no lamella modulation
(0.037-0.073 vs 0.443 for an aligned control): sampling across the sheets
averages the modulation away. This failure mode is invisible to area, validity
and intensity checks.

Method (must not be changed without re-calibrating the reference numbers):
  mesh normal  : per-vertex cross product of the tifxyz grid tangents
                 (np.gradient along the two grid axes, in (z,y,x) space),
                 axially averaged via the leading eigenvector of the
                 orientation tensor (a normal's sign is arbitrary).
                 SENTINEL FIX: tifxyz marks invalid vertices with -1, NOT 0.
                 Validity = (x>=0)&(y>=0)&(z>=0)&~(x==y==z==0), then a
                 one-vertex 3x3 binary erosion so that every contributing
                 vertex has fully valid central-difference neighbours.
                 Masking on zeros instead silently keeps -1 sentinels as real
                 coordinates; their boundary gradients are enormous and skew
                 the average (published meshes are ~50% invalid, so the old
                 bug biased populations by very different amounts).
  sheet normal : structure tensor per 32^3 block over a 256^3 window of the
                 volume centred on the patch site (gaussian sigma=1.0,
                 np.gradient derivatives); blocks with <98% non-zero voxels
                 are dropped (air); per-block leading eigenvectors are
                 coherence-weighted and axially averaged; needs >= 8 blocks.
  angle        : arccos(|dot|) folded to [0, 90] degrees.

Inputs:  --mesh   tifxyz directory, repeatable
         --volume 3D volume as .npy, 3D .tif/.tiff stack, a directory of 2D
                  .tif slices (sorted by filename = z), or a .zarr array/group
                  (needs the `zarr` package; pick a group member with
                  --zarr-subpath, default "0" if present). Axis order (z,y,x);
                  tifxyz x.tif/y.tif/z.tif hold the voxel column/row/slice.
Exit codes: 0 all meshes ALIGNED, 1 at least one REJECT,
            2 at least one INDETERMINATE (and no REJECT), 3 bad input.

Deps: numpy, scipy, tifffile (zarr optional). No project-specific layout.
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import ndimage as ndi

__version__ = "1.0.0"

CALIBRATION = {
    "published_reference_median_deg": 13.1,
    "published_reference_within_30deg": "7/9",
    "random_null_median_deg": 60.0,
    "failed_population_median_deg": 68.1,
    "failed_population_within_30deg": "0/8",
    "reject_above_deg": 30.0,
    "source": "PHerc0813 mesh-lamella alignment, sentinel-fixed method, 2026-08",
}


def _tifffile():
    try:
        import tifffile
    except ImportError:
        raise SystemExit("alignment_gate needs the `tifffile` package")
    return tifffile


# ---------------------------------------------------------------- volume I/O

class NpyVolume:
    def __init__(self, path):
        self.a = np.load(path, mmap_mode="r")
        if self.a.ndim != 3:
            raise SystemExit(f"{path}: expected 3D array, got shape {self.a.shape}")
        self.shape = self.a.shape

    def read(self, z0, z1, y0, y1, x0, x1):
        return np.asarray(self.a[z0:z1, y0:y1, x0:x1])


class TiffStackVolume:
    """One multi-page TIFF holding a (Z,Y,X) stack. Reads only needed pages."""

    def __init__(self, path):
        tf = _tifffile()
        self.path = path
        with tf.TiffFile(path) as t:
            s = t.series[0]
            if len(s.shape) != 3:
                raise SystemExit(f"{path}: expected 3D series, got {s.shape}")
            self.shape = tuple(s.shape)

    def read(self, z0, z1, y0, y1, x0, x1):
        tf = _tifffile()
        a = tf.imread(self.path, key=range(z0, z1))
        if a.ndim == 2:
            a = a[None]
        return a[:, y0:y1, x0:x1]


class SliceDirVolume:
    """Directory of 2D .tif slices; sorted filename order = z index."""

    def __init__(self, path):
        tf = _tifffile()
        self.files = sorted(
            os.path.join(path, f) for f in os.listdir(path)
            if f.lower().endswith((".tif", ".tiff"))
        )
        if not self.files:
            raise SystemExit(f"{path}: no .tif slices found")
        first = tf.imread(self.files[0])
        if first.ndim != 2:
            raise SystemExit(f"{self.files[0]}: expected 2D slices")
        self.shape = (len(self.files),) + first.shape

    def read(self, z0, z1, y0, y1, x0, x1):
        tf = _tifffile()
        return np.stack([tf.imread(f)[y0:y1, x0:x1] for f in self.files[z0:z1]])


class ZarrVolume:
    def __init__(self, path, subpath=None):
        try:
            import zarr
        except ImportError:
            raise SystemExit("reading .zarr needs the `zarr` package")
        a = zarr.open(path, mode="r")
        if not hasattr(a, "ndim"):  # group
            keys = list(a.keys())
            sp = subpath if subpath is not None else ("0" if "0" in keys else None)
            if sp is None or sp not in keys:
                raise SystemExit(f"{path} is a zarr group; pick --zarr-subpath "
                                 f"from {keys}")
            a = a[sp]
        if a.ndim != 3:
            raise SystemExit(f"{path}: expected 3D zarr array, got {a.shape}")
        self.a = a
        self.shape = tuple(a.shape)

    def read(self, z0, z1, y0, y1, x0, x1):
        return np.asarray(self.a[z0:z1, y0:y1, x0:x1])


def open_volume(path, zarr_subpath=None):
    p = os.path.normpath(path)
    if os.path.isdir(p):
        if (p.rstrip("/\\").lower().endswith(".zarr")
                or os.path.exists(os.path.join(p, ".zarray"))
                or os.path.exists(os.path.join(p, ".zgroup"))
                or os.path.exists(os.path.join(p, "zarr.json"))):
            return ZarrVolume(p, zarr_subpath)
        return SliceDirVolume(p)
    if not os.path.exists(p):
        raise SystemExit(f"volume not found: {p}")
    ext = os.path.splitext(p)[1].lower()
    if ext == ".npy":
        return NpyVolume(p)
    if ext in (".tif", ".tiff"):
        return TiffStackVolume(p)
    raise SystemExit(f"unsupported volume type: {p} (want .npy, .tif stack, "
                     f"slice directory, or .zarr)")


# ---------------------------------------------------------------- mesh side

def load_tifxyz(d):
    """Load a tifxyz mesh directory -> (x, y, z, valid) float64/bool arrays.

    SENTINEL FIX: invalid tifxyz vertices are -1, NOT 0. Mask on sign, and
    also drop an exact (0,0,0) vertex (never a real surface point).
    """
    tf = _tifffile()
    arrs = []
    for name in ("x", "y", "z"):
        p = os.path.join(d, name + ".tif")
        if not os.path.exists(p):
            raise SystemExit(f"{d}: missing {name}.tif (not a tifxyz directory)")
        arrs.append(tf.imread(p).astype(np.float64))
    x, y, z = arrs
    if not (x.shape == y.shape == z.shape) or x.ndim != 2:
        raise SystemExit(f"{d}: x/y/z shape mismatch or not 2D grids")
    valid = (x >= 0) & (y >= 0) & (z >= 0) & ~((x == 0) & (y == 0) & (z == 0))
    return x, y, z, valid


def mesh_normal_field(x, y, z, valid):
    """Per-vertex unit normals (H,W,3) in (z,y,x) order + contributing mask.

    One-vertex erosion: a central difference at p uses p's 4-neighbours, so
    only vertices whose full 3x3 neighbourhood is valid may contribute --
    otherwise -1 sentinels leak into the gradients.
    """
    core = ndi.binary_erosion(valid, structure=np.ones((3, 3), bool),
                              border_value=0)
    tu = np.stack([np.gradient(a, axis=0) for a in (z, y, x)], -1)
    tv = np.stack([np.gradient(a, axis=1) for a in (z, y, x)], -1)
    n = np.cross(tu, tv)
    L = np.linalg.norm(n, axis=-1)
    ok = core & (L > 1e-6)
    with np.errstate(invalid="ignore", divide="ignore"):
        nu = np.where(ok[..., None], n / np.maximum(L, 1e-30)[..., None], 0.0)
    return nu, ok


def axial_mean(normals):
    """Leading eigenvector of the orientation tensor of unit vectors (N,3)."""
    T = np.einsum("ij,ik->jk", normals, normals) / len(normals)
    w, V = np.linalg.eigh(T)
    return V[:, 2]


def sheet_normal(a, block=32, sigma=1.0, min_blocks=8, min_fill=0.98):
    """Coherence-weighted axial mean of per-block structure-tensor normals.

    Returns (normal (z,y,x), n_blocks, mean_coherence) or (None, n, 0.0).
    """
    Z, Y, X = a.shape
    N, C = [], []
    for z in range(0, Z - block + 1, block):
        for y in range(0, Y - block + 1, block):
            for x in range(0, X - block + 1, block):
                bl = a[z:z + block, y:y + block, x:x + block]
                if (bl > 0).mean() < min_fill:
                    continue  # air / padding
                v = ndi.gaussian_filter(bl.astype(np.float32), sigma)
                g = np.gradient(v)
                J = np.array([[float((g[i] * g[j]).mean()) for j in range(3)]
                              for i in range(3)])
                w, V = np.linalg.eigh(J)
                C.append((w[2] - w[1]) / max(w[2] + w[1], 1e-9))
                N.append(V[:, 2])
    if len(N) < min_blocks:
        return None, len(N), 0.0
    N, C = np.array(N), np.array(C)
    T = np.einsum("i,ij,ik->jk", C / C.sum(), N, N)
    w, V = np.linalg.eigh(T)
    return V[:, 2], len(N), float(C.mean())


def axial_angle_deg(a, b):
    """Angle between two axial directions, folded to [0, 90] degrees."""
    return float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(a, b)))))))


# ---------------------------------------------------------------- the gate

def window_bounds(center, side, extent):
    """Clamp a window of `side` voxels centred at `center` into [0, extent)."""
    side = min(side, extent)
    lo = int(round(center)) - side // 2
    lo = max(0, min(lo, extent - side))
    return lo, lo + side


def gate_site(vol, cz, cy, cx, args):
    """Extract the volume window at one site and return its sheet normal."""
    z0, z1 = window_bounds(cz, args.cube, vol.shape[0])
    y0, y1 = window_bounds(cy, args.cube, vol.shape[1])
    x0, x1 = window_bounds(cx, args.cube, vol.shape[2])
    if min(z1 - z0, y1 - y0, x1 - x0) < 2 * args.block:
        return None, 0, 0.0, "volume window smaller than 2 blocks"
    cube = vol.read(z0, z1, y0, y1, x0, x1)
    sn, nb, coh = sheet_normal(cube, block=args.block, sigma=args.sigma)
    if sn is None:
        return None, nb, 0.0, f"only {nb} usable blocks (need 8; air/padding?)"
    return sn, nb, coh, None


def verdict_of(angle, reject_deg):
    if angle <= reject_deg:
        return "ALIGNED", "mesh follows the sheets"
    if angle < 60.0:
        return "REJECT", "oblique to the sheets"
    return "REJECT", "mesh cuts ACROSS the sheets"


def gate_mesh(meshdir, vol, args):
    x, y, z, valid = load_tifxyz(meshdir)
    nu, ok = mesh_normal_field(x, y, z, valid)
    H, W = valid.shape
    S = max(1, args.split)
    tiles = []
    for r in range(S):
        for c in range(S):
            r0, r1 = (H * r) // S, (H * (r + 1)) // S
            c0, c1 = (W * c) // S, (W * (c + 1)) // S
            m = np.zeros((H, W), bool)
            m[r0:r1, c0:c1] = True
            sel = ok & m
            nsel = int(sel.sum())
            tile = {"tile": [r, c], "n_vertices": nsel}
            if nsel < 50:
                tile.update(verdict="INDETERMINATE",
                            note=f"only {nsel} contributing vertices (need 50)")
                tiles.append(tile)
                continue
            mn = axial_mean(nu[sel])
            cz = float(np.median(z[sel])); cy = float(np.median(y[sel]))
            cx = float(np.median(x[sel]))
            sn, nb, coh, err = gate_site(vol, cz, cy, cx, args)
            tile.update(site_zyx=[cz, cy, cx],
                        mesh_normal_zyx=[float(v) for v in mn])
            if sn is None:
                tile.update(verdict="INDETERMINATE", note=err, n_blocks=nb)
                tiles.append(tile)
                continue
            ang = axial_angle_deg(mn, sn)
            v, note = verdict_of(ang, args.reject_deg)
            tile.update(angle_deg=round(ang, 2), verdict=v, note=note,
                        n_blocks=nb, mean_coherence=round(coh, 4),
                        sheet_normal_zyx=[float(v_) for v_ in sn])
            tiles.append(tile)

    angles = [t["angle_deg"] for t in tiles if "angle_deg" in t]
    res = {
        "mesh": meshdir,
        "grid_shape": [H, W],
        "valid_fraction": round(float(valid.mean()), 6),
        "n_contributing_vertices": int(ok.sum()),
        "tiles": tiles,
    }
    if not angles:
        res.update(verdict="INDETERMINATE", note="no tile produced an angle")
    else:
        med = float(np.median(angles))
        v, note = verdict_of(med, args.reject_deg)
        res.update(median_angle_deg=round(med, 2), verdict=v, note=note,
                   n_tiles_rejected=sum(1 for t in tiles
                                        if t.get("verdict") == "REJECT"),
                   n_tiles=len(tiles))
    return res


class _Parser(argparse.ArgumentParser):
    """Usage errors exit 3 (2 is taken by the INDETERMINATE verdict)."""

    def error(self, message):
        self.print_usage(sys.stderr)
        print(f"{self.prog}: error: {message}", file=sys.stderr)
        sys.exit(3)


def main(argv=None):
    ap = _Parser(
        prog="alignment_gate",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", action="append", required=True,
                    help="tifxyz mesh directory (x.tif/y.tif/z.tif); repeatable")
    ap.add_argument("--volume", required=True,
                    help=".npy | 3D .tif stack | directory of 2D .tif slices "
                         "| .zarr  -- axis order (z,y,x), same voxel frame the "
                         "tifxyz coordinates index into")
    ap.add_argument("--cube", type=int, default=256,
                    help="side of the volume window per site (default 256 "
                         "= 8x8x8 blocks of 32, matching the 256^3 cubes the "
                         "reference numbers were calibrated on)")
    ap.add_argument("--block", type=int, default=32,
                    help="structure-tensor block side (default 32)")
    ap.add_argument("--sigma", type=float, default=1.0,
                    help="gaussian pre-smoothing sigma (default 1.0)")
    ap.add_argument("--reject-deg", type=float, default=30.0,
                    help="reject when angle exceeds this (default 30; "
                         "published meshes median 13.1, random 60)")
    ap.add_argument("--split", type=int, default=1,
                    help="tile the mesh grid SxS and gate each tile at its own "
                         "site (default 1 = whole mesh, as calibrated)")
    ap.add_argument("--zarr-subpath", default=None,
                    help="array key inside a zarr group (default '0')")
    ap.add_argument("--out", default=None, help="write results JSON here")
    args = ap.parse_args(argv)

    vol = open_volume(args.volume, args.zarr_subpath)
    print(f"volume {args.volume}  shape {tuple(vol.shape)}")
    print(f"gate: reject > {args.reject_deg:g} deg   "
          f"(published reference median 13.1, random null 60)")
    print(f"{'mesh':40}{'angle':>8}{'verdict':>15}  note")
    results = []
    for md in args.mesh:
        r = gate_mesh(md, vol, args)
        results.append(r)
        a = r.get("median_angle_deg")
        astr = f"{a:.1f}" if a is not None else "-"
        print(f"{os.path.basename(os.path.normpath(md))[:40]:40}"
              f"{astr:>8}{r['verdict']:>15}  {r.get('note', '')}")

    verdicts = [r["verdict"] for r in results]
    overall = ("REJECT" if "REJECT" in verdicts else
               "INDETERMINATE" if "INDETERMINATE" in verdicts else "ALIGNED")
    print(f"\noverall: {overall}  "
          f"({verdicts.count('ALIGNED')} aligned, "
          f"{verdicts.count('REJECT')} rejected, "
          f"{verdicts.count('INDETERMINATE')} indeterminate)")

    payload = {
        "tool": "alignment_gate", "version": __version__,
        "params": {k: getattr(args, k) for k in
                   ("cube", "block", "sigma", "reject_deg", "split")},
        "calibration": CALIBRATION,
        "volume": {"path": args.volume, "shape": list(vol.shape)},
        "overall_verdict": overall,
        "meshes": results,
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=1)
        print(f"wrote {args.out}")

    return 1 if overall == "REJECT" else 2 if overall == "INDETERMINATE" else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit as e:
        if isinstance(e.code, str):  # bad input raised as SystemExit("msg")
            print(e.code, file=sys.stderr)
            sys.exit(3)
        raise
