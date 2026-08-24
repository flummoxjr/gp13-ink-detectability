"""Stage 1a prep — fetch PHerc1667 w032 ground truth (ink TIF + on-7.91um tifxyz).

Lists the segment dir on S3, downloads the 2.4um-model ink TIF, the 1.129um-model
ink TIF (for the two-scan-confirmed mask), and the on-7.91um coordinate grids.
Reports shapes, ink-probability histogram, and the 7.91um-frame bounding box of
the mapped surface -> tells us how much volume Stage 1a must stream.
"""
import io
import os
import xml.etree.ElementTree as ET

import numpy as np
import requests

BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
SEG = "PHerc1667/segments/20260105050000-w032_2026010505_flatboi"
CACHE = r"D:\vesuvius-data\trackD\w032"
os.makedirs(CACHE, exist_ok=True)
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def list_all(prefix):
    keys, token = [], None
    while True:
        url = f"{BUCKET}/?list-type=2&prefix={prefix}&max-keys=1000"
        if token:
            url += f"&continuation-token={requests.utils.quote(token)}"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        for e in root.findall("s3:Contents", NS):
            keys.append((e.find("s3:Key", NS).text, int(e.find("s3:Size", NS).text)))
        tok = root.find("s3:NextContinuationToken", NS)
        if tok is None:
            break
        token = tok.text
    return keys


def fetch(key, dest):
    if os.path.exists(dest):
        return dest
    r = requests.get(f"{BUCKET}/{key}", timeout=600, stream=True)
    r.raise_for_status()
    tmp = dest + ".part"
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    os.replace(tmp, dest)
    return dest


def main():
    keys = list_all(SEG + "/")
    print(f"{len(keys)} objects in segment dir; relevant ones:")
    wanted = []
    for k, s in keys:
        rel = k[len(SEG) + 1:]
        if ("ink-detection/" in k and k.endswith(".tif")) or "on-20231117161658" in k or rel == "meta.json":
            print(f"  {rel}  ({s/1e6:.1f} MB)")
            wanted.append((k, rel))

    ink24 = [k for k, r in wanted if "ink-detection" in k and "20251217075048" in k]
    ink11 = [k for k, r in wanted if "ink-detection" in k and "20260323082859" in k]
    grids = [(k, r) for k, r in wanted if "on-20231117161658" in k]
    print("\n2.4um ink TIF:", ink24)
    print("1.129um ink TIF:", ink11)
    print("grid files:", [r for _, r in grids])

    import tifffile

    # coordinate grids (small)
    grid = {}
    for k, r in grids:
        name = os.path.basename(k)
        dest = os.path.join(CACHE, "grid_" + name)
        fetch(k, dest)
        if name.endswith(".tif"):
            grid[name.split(".")[0]] = tifffile.imread(dest)
        else:
            print("meta.json:", open(dest).read()[:400])
    for n, g in grid.items():
        print(f"grid {n}: shape={g.shape} dtype={g.dtype} min={np.nanmin(g):.1f} max={np.nanmax(g):.1f}")

    # ink TIFs
    for tag, lst in [("ink24", ink24), ("ink11", ink11)]:
        if not lst:
            continue
        dest = os.path.join(CACHE, f"{tag}.tif")
        print(f"downloading {tag} ...")
        fetch(lst[0], dest)
        img = tifffile.imread(dest)
        print(f"{tag}: shape={img.shape} dtype={img.dtype}")
        v = img[::8, ::8].ravel()
        v = v[v > 0]
        qs = np.percentile(v, [50, 90, 99]) if len(v) else []
        print(f"  nonzero px {len(v)} of {img[::8, ::8].size}; p50/p90/p99 = {qs}")
        np.save(os.path.join(CACHE, f"{tag}_ds4.npy"), img[::4, ::4])

    # Bounding box of the mapped surface in the 7.91um frame
    x, y, z = grid.get("x"), grid.get("y"), grid.get("z")
    if x is not None:
        valid = (x > 0) | (y > 0) | (z > 0)
        print(f"\nmapped-surface voxel bbox in 20231117161658 (7.91um):")
        print(f"  x: [{x[valid].min():.0f}, {x[valid].max():.0f}]")
        print(f"  y: [{y[valid].min():.0f}, {y[valid].max():.0f}]")
        print(f"  z: [{z[valid].min():.0f}, {z[valid].max():.0f}]")
        span = [(int(a[valid].max() - a[valid].min())) for a in (z, y, x)]
        gb = (span[0] * span[1] * span[2]) / 1e9
        print(f"  span z,y,x = {span} -> dense bbox {gb:.2f} GB uint8")


if __name__ == "__main__":
    main()
