#!/usr/bin/env python
"""Self-test for alignment_gate on a synthetic mesh+volume pair of KNOWN orientation.

Builds a 200^3 volume whose lamellae are horizontal planes (sheet normal = z axis,
period 12 voxels), plus four 64x64 tifxyz meshes:

  aligned            z = const plane            -> expect angle ~0,  ALIGNED
  aligned_sentinel   same plane, but with a 3-vertex ring AND an interior blob
                     of -1 invalid sentinels    -> expect angle ~0,  ALIGNED
                     (regression test for the sentinel fix: -1 is invalid, not 0)
  oblique45          plane tilted 45 deg        -> expect angle ~45, REJECT
  orthogonal         plane containing z axis    -> expect angle ~90, REJECT

Also unit-checks the sentinel mask itself: the fixed mask must exclude exactly
the -1 vertices (the old zero-mask matches nothing and keeps them all), and the
one-vertex erosion must exclude the sentinels' neighbours from contributing.

Run:  python selftest.py        (exit 0 = all assertions passed)
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import alignment_gate as ag  # noqa: E402

try:
    import tifffile
except ImportError:
    sys.exit("selftest needs tifffile")

DATA = os.path.join(HERE, "selftest_data")
H = W = 64          # mesh grid
N = 200             # volume side
PERIOD = 12.0       # lamella period, voxels


def build_volume(path):
    rng = np.random.default_rng(0)
    zz = np.arange(N, dtype=np.float32)
    prof = 120.0 + 80.0 * np.sin(2.0 * np.pi * zz / PERIOD)
    a = np.broadcast_to(prof[:, None, None], (N, N, N)).copy()
    a += rng.normal(0.0, 2.0, size=a.shape).astype(np.float32)
    a = np.clip(a, 20, 235).astype(np.uint8)   # all > 0: every block passes fill
    np.save(path, a)
    return a


def write_tifxyz(d, x, y, z):
    os.makedirs(d, exist_ok=True)
    tifffile.imwrite(os.path.join(d, "x.tif"), x.astype(np.float32))
    tifffile.imwrite(os.path.join(d, "y.tif"), y.astype(np.float32))
    tifffile.imwrite(os.path.join(d, "z.tif"), z.astype(np.float32))


def build_meshes():
    i = np.arange(H, dtype=np.float64)[:, None] * np.ones((1, W))
    j = np.ones((H, 1)) * np.arange(W, dtype=np.float64)[None, :]

    # aligned: z=80 plane; normal = z axis; angle vs sheets = 0
    write_tifxyz(os.path.join(DATA, "aligned"),
                 x=20 + 2 * j, y=20 + 2 * i, z=np.full((H, W), 80.0))

    # aligned with -1 sentinels: 3-vertex border ring + interior 4x4 blob
    x = 20 + 2 * j
    y = 20 + 2 * i
    z = np.full((H, W), 80.0)
    inval = np.zeros((H, W), bool)
    inval[:3, :] = inval[-3:, :] = inval[:, :3] = inval[:, -3:] = True
    inval[30:34, 30:34] = True
    for a in (x, y, z):
        a[inval] = -1.0
    write_tifxyz(os.path.join(DATA, "aligned_sentinel"), x=x, y=y, z=z)

    # oblique 45 deg: tangents (1.4,0,1.4) and (0,2,0) in (z,y,x)
    write_tifxyz(os.path.join(DATA, "oblique45"),
                 x=100 + 1.4 * (i - 32), y=20 + 2 * j, z=80 + 1.4 * (i - 32))

    # orthogonal: x=100 plane containing the z axis; normal = x axis; angle 90
    write_tifxyz(os.path.join(DATA, "orthogonal"),
                 x=np.full((H, W), 100.0), y=20 + 2 * j, z=20 + 2 * i)

    return inval


def unit_check_sentinels(inval):
    """The load-time mask and the erosion must handle -1 sentinels exactly."""
    x, y, z, valid = ag.load_tifxyz(os.path.join(DATA, "aligned_sentinel"))
    n_inval = int(inval.sum())
    assert valid.sum() == H * W - n_inval, \
        f"fixed mask: expected {H*W - n_inval} valid, got {int(valid.sum())}"

    # the OLD buggy mask (invalid == exact zeros) matches nothing: every
    # sentinel is kept as a real coordinate
    buggy = ~((x == 0) & (y == 0) & (z == 0))
    assert buggy.all(), "buggy zero-mask unexpectedly caught a sentinel"

    # one-vertex erosion: vertices 8-adjacent to the interior blob must not
    # contribute (their central differences would span a sentinel)
    _, ok = ag.mesh_normal_field(x, y, z, valid)
    assert not ok[29:35, 29:35].any(), \
        "erosion failed: a vertex adjacent to the -1 blob contributes"
    assert ok.sum() > 2000, f"too few contributing vertices: {int(ok.sum())}"

    # informational: the angle the buggy mask would produce on this mesh
    nu_b, ok_b = ag.mesh_normal_field(x, y, z, buggy)
    mn_b = ag.axial_mean(nu_b[ok_b])
    mn_f = ag.axial_mean(ag.mesh_normal_field(x, y, z, valid)[0][ok])
    zaxis = np.array([1.0, 0.0, 0.0])
    print(f"  sentinel mesh, mesh-normal vs true z axis: "
          f"fixed {ag.axial_angle_deg(mn_f, zaxis):.2f} deg, "
          f"buggy zero-mask {ag.axial_angle_deg(mn_b, zaxis):.2f} deg")
    print("  sentinel unit checks passed")


def main():
    if os.path.isdir(DATA):
        shutil.rmtree(DATA)
    os.makedirs(DATA)
    vol_path = os.path.join(DATA, "vol.npy")
    print("building synthetic volume (200^3, lamellae normal to z, period 12) ...")
    build_volume(vol_path)
    inval = build_meshes()

    print("sentinel unit checks ...")
    unit_check_sentinels(inval)

    out_json = os.path.join(DATA, "selftest_out.json")
    cmd = [sys.executable, os.path.join(HERE, "alignment_gate.py"),
           "--volume", vol_path, "--out", out_json]
    for m in ("aligned", "aligned_sentinel", "oblique45", "orthogonal"):
        cmd += ["--mesh", os.path.join(DATA, m)]
    print("running:", " ".join(os.path.basename(c) or c for c in cmd[1:]))
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout)
    if r.stderr:
        print(r.stderr, file=sys.stderr)
    assert r.returncode == 1, \
        f"expected exit 1 (rejects present), got {r.returncode}"

    res = {os.path.basename(m["mesh"]): m
           for m in json.load(open(out_json))["meshes"]}
    exp = {
        "aligned":          (0.0,  5.0, "ALIGNED"),
        "aligned_sentinel": (0.0,  5.0, "ALIGNED"),
        "oblique45":        (40.0, 50.0, "REJECT"),
        "orthogonal":       (85.0, 90.0, "REJECT"),
    }
    fails = []
    for name, (lo, hi, verdict) in exp.items():
        m = res[name]
        a = m.get("median_angle_deg")
        ok_ = a is not None and lo <= a <= hi and m["verdict"] == verdict
        print(f"  {name:18} angle {a} verdict {m['verdict']:8} "
              f"expected [{lo},{hi}] {verdict}  ->  {'OK' if ok_ else 'FAIL'}")
        if not ok_:
            fails.append(name)
    assert not fails, f"self-test FAILED for: {fails}"
    print("\nSELF-TEST PASSED (4/4 verdicts, sentinel regression included)")


if __name__ == "__main__":
    main()
