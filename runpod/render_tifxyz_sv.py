#!/usr/bin/env python
"""Render a centered N-slice uint8 surface volume from a tifxyz mesh + volume zarr."""
import argparse
import math
import numpy as np
from numcodecs import Blosc
from scipy.ndimage import map_coordinates

from vesuvius.tifxyz import read_tifxyz
from vesuvius.ink_detection.volume_io import open_volume, read_bbox_with_padding
from vesuvius.label_zarr import open_v2_group, create_v2_array


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("tifxyz_dir")
    ap.add_argument("volume", help="scroll volume (local path or s3:// OME-Zarr); level 0 is read")
    ap.add_argument("output_zarr")
    ap.add_argument("--num-slices", type=int, default=21)
    ap.add_argument("--slice-step", type=float, default=1.0)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--margin", type=int, default=2)
    ap.add_argument("--max-crop-voxels", type=float, default=1.5e9)
    ap.add_argument("--cache-dir", default=None, help="zarr-3 chunk cache dir (optional)")
    ap.add_argument("--cache-max-gb", type=float, default=100.0)
    return ap.parse_args()


def main():
    args = parse_args()
    surf = read_tifxyz(args.tifxyz_dir, load_mask=False, validate=False)
    surf.use_full_resolution()
    H, W = surf.shape
    n = int(args.num_slices)
    offsets = (np.arange(n, dtype=np.float64) - (n - 1) / 2.0) * float(args.slice_step)
    pad = int(math.ceil(np.abs(offsets).max())) + int(args.margin)
    print(f"full-res grid: {H} x {W}, {n} slices, offsets {offsets[0]}..{offsets[-1]}")

    kwargs = {}
    if args.cache_dir:
        kwargs.update(cache_dir=args.cache_dir, cache_max_gb=args.cache_max_gb)
    try:
        vol = open_volume(args.volume, 0, **kwargs)
    except NotImplementedError:  # disk cache requires zarr 3
        print("zarr<3: continuing without chunk cache")
        vol = open_volume(args.volume, 0)

    group = open_v2_group(args.output_zarr)
    comp = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    out = create_v2_array(group, "0", shape=(n, H, W), chunks=(n, 256, 256),
                          dtype=np.uint8, compressor=comp, fill_value=0)

    def render_tile(r0, r1, c0, c1):
        x, y, z, valid = surf[r0:r1, c0:c1]
        if not np.any(valid):
            return
        nx, ny, nz = surf.get_normals(r0, r1, c0, c1)
        ok = valid & np.isfinite(nx) & np.isfinite(ny) & np.isfinite(nz)
        if not np.any(ok):
            return
        z0 = int(np.floor(z[ok].min())) - pad; z1 = int(np.ceil(z[ok].max())) + pad + 1
        y0 = int(np.floor(y[ok].min())) - pad; y1 = int(np.ceil(y[ok].max())) + pad + 1
        x0 = int(np.floor(x[ok].min())) - pad; x1 = int(np.ceil(x[ok].max())) + pad + 1
        nvox = float(z1 - z0) * (y1 - y0) * (x1 - x0)
        if nvox > args.max_crop_voxels and (r1 - r0 > 64 or c1 - c0 > 64):
            rm, cm = (r0 + r1) // 2, (c0 + c1) // 2
            for (a, b, c, d) in ((r0, rm, c0, cm), (r0, rm, cm, c1),
                                 (rm, r1, c0, cm), (rm, r1, cm, c1)):
                if a < b and c < d:
                    render_tile(a, b, c, d)
            return
        crop, _ = read_bbox_with_padding(vol, (z0, y0, x0, z1, y1, x1), fill_value=0)
        crop = crop.astype(np.float32, copy=False)
        th, tw = r1 - r0, c1 - c0
        tile_out = np.zeros((n, th, tw), dtype=np.uint8)
        zi, yi, xi = z[ok] - z0, y[ok] - y0, x[ok] - x0
        nzo, nyo, nxo = nz[ok], ny[ok], nx[ok]
        for si, off in enumerate(offsets):
            coords = np.stack([zi + off * nzo, yi + off * nyo, xi + off * nxo])
            vals = map_coordinates(crop, coords, order=1, mode="constant", cval=0.0)
            plane = np.zeros((th, tw), dtype=np.float32)
            plane[ok] = vals
            tile_out[si] = np.clip(np.rint(plane), 0, 255).astype(np.uint8)
        out[:, r0:r1, c0:c1] = tile_out

    t = int(args.tile)
    rows = list(range(0, H, t))
    for i, r0 in enumerate(rows):
        for c0 in range(0, W, t):
            render_tile(r0, min(H, r0 + t), c0, min(W, c0 + t))
        print(f"row band {i + 1}/{len(rows)} done")

    # occupancy level "3" (YX max-pool by 8) so infer.py can skip empty tiles
    p = 8
    occ = create_v2_array(group, "3", shape=(n, (H + p - 1) // p, (W + p - 1) // p),
                          chunks=(n, 256, 256), dtype=np.uint8, compressor=comp,
                          fill_value=0)
    band = 4096  # multiple of p
    for r0 in range(0, H, band):
        r1 = min(H, r0 + band)
        block = np.asarray(out[:, r0:r1, :])
        h = block.shape[1]
        ph, pw = (-h) % p, (-W) % p
        if ph or pw:
            block = np.pad(block, ((0, 0), (0, ph), (0, pw)))
        pooled = block.reshape(n, (h + ph) // p, p, (W + pw) // p, p).max(axis=(2, 4))
        occ[:, r0 // p : r0 // p + pooled.shape[1], :] = pooled
    print("done:", args.output_zarr)


if __name__ == "__main__":
    main()
