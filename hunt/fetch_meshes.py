#!/usr/bin/env python
"""Fetch the 10 tifxyz meshes for the geometry comparison into a local cache."""
import json
import os
import sys

import s3fs

B = "vesuvius-challenge-open-data"
CACHE = r"C:\Users\benbl\Desktop\Vsuvious\trackD\hunt\meshcache"

SEGS = [
    # key, scroll, tifxyz prefix (relative to bucket), volume, label
    dict(key="w035", scroll="PHerc0139", role="CONTROL",
         tifxyz="PHerc0139/segments/20260317000000-w035_2026031718/mesh/20260317000000-on-20250728140407-9.362um.tifxyz/",
         volume="PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr", vox_um=9.362, fwd_rev_r=0.076),
    dict(key="w032", scroll="PHerc0139", role="CONTROL2",
         tifxyz="PHerc0139/segments/20260203000000-w032_2026020303/mesh/20260203000000-on-20250728140407-9.362um.tifxyz/",
         volume="PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr", vox_um=9.362, fwd_rev_r=None),
    dict(key="1203_r399", scroll="PHerc1203", role="GP",
         tifxyz="PHerc1203/segments/raw/auto_grown_20250923164713356/",
         volume="PHerc1203/volumes/20250820131727-9.362um-1.2m-113keV-masked.zarr", vox_um=9.362, fwd_rev_r=0.399),
    dict(key="1203_r460", scroll="PHerc1203", role="GP",
         tifxyz="PHerc1203/segments/raw/auto_grown_20250923163217042/",
         volume="PHerc1203/volumes/20250820131727-9.362um-1.2m-113keV-masked.zarr", vox_um=9.362, fwd_rev_r=0.460),
    dict(key="1203_r747", scroll="PHerc1203", role="GP",
         tifxyz="PHerc1203/segments/raw/auto_grown_20251005230830031/",
         volume="PHerc1203/volumes/20250820131727-9.362um-1.2m-113keV-masked.zarr", vox_um=9.362, fwd_rev_r=0.747),
    dict(key="1447_r222", scroll="PHerc1447", role="GP",
         tifxyz="PHerc1447/segments/raw/auto_grown_20250702235910292/",
         volume="PHerc1447/volumes/20250521151220-8.640um-1.2m-116keV-masked.zarr", vox_um=8.640, fwd_rev_r=0.222),
    dict(key="1447_r623", scroll="PHerc1447", role="GP",
         tifxyz="PHerc1447/segments/raw/auto_grown_20250502161324419/",
         volume="PHerc1447/volumes/20250521151220-8.640um-1.2m-116keV-masked.zarr", vox_um=8.640, fwd_rev_r=0.623),
    dict(key="1447_r914", scroll="PHerc1447", role="GP",
         tifxyz="PHerc1447/segments/raw/auto_grown_20250502160708188/",
         volume="PHerc1447/volumes/20250521151220-8.640um-1.2m-116keV-masked.zarr", vox_um=8.640, fwd_rev_r=0.914),
    dict(key="0800_r329", scroll="PHerc0800", role="GP",
         tifxyz="PHerc0800/segments/20251028213516-auto_grown_20251028213516907/mesh/intermediate/tifxyz_original/",
         volume="PHerc0800/volumes/20250521135224-8.640um-1.2m-116keV-masked.zarr", vox_um=8.640, fwd_rev_r=0.329),
    dict(key="0800_r522", scroll="PHerc0800", role="GP",
         tifxyz="PHerc0800/segments/20251028220955-auto_grown_20251028220955262/mesh/intermediate/tifxyz_original/",
         volume="PHerc0800/volumes/20250521135224-8.640um-1.2m-116keV-masked.zarr", vox_um=8.640, fwd_rev_r=0.522),
]

if __name__ == "__main__":
    fs = s3fs.S3FileSystem(anon=True)
    os.makedirs(CACHE, exist_ok=True)
    for s in SEGS:
        d = os.path.join(CACHE, s["key"])
        os.makedirs(d, exist_ok=True)
        for f in ("x.tif", "y.tif", "z.tif", "meta.json", "mask.tif"):
            dst = os.path.join(d, f)
            if os.path.exists(dst):
                continue
            src = f"{B}/{s['tifxyz']}{f}"
            try:
                fs.get(src, dst)
                print("ok  ", s["key"], f, os.path.getsize(dst))
            except Exception as e:
                print("miss", s["key"], f, type(e).__name__)
    json.dump(SEGS, open(os.path.join(CACHE, "segs.json"), "w"), indent=1)
    print("cached to", CACHE)
