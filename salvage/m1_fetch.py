"""m1_fetch.py — stream CT (band L2) and surface-pred (its L0 = band-L2 frame)
64^3 subvolumes matching each sampled prob map. Cache as npy in cache/.

Band L2 = L0/4, so tile L0 origin (z,y,x) -> L2 slice [z//4 : z//4+64, ...].
"""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import zarr, fsspec

OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\salvage"
CACHE = os.path.join(OUT, "cache")
CT_URL = ("https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1203/volumes/"
          "20260319130212-2.403um-0.2m-77keV-masked.zarr")
SURF_URL = ("https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1203/representations/"
            "predictions/surfaces/20260319130212-surface-20260413222639-surface-m7-L2-th0.2.zarr")

inv = json.load(open(os.path.join(OUT, "inventory.json")))["samples"]
tiles = sorted({tuple(r["tile"]) for r in inv})
print(f"{len(tiles)} unique tile origins")

def open_arr(url, level):
    m = fsspec.get_mapper(url)
    store = zarr.storage.KVStore(m)
    cached = zarr.LRUStoreCache(store, max_size=2**31)  # 2GB chunk cache
    g = zarr.open_group(cached, mode="r")
    return g[str(level)]

ct = open_arr(CT_URL, 2)
surf = open_arr(SURF_URL, 0)
print("CT L2", ct.shape, ct.chunks, "SURF L0", surf.shape, surf.chunks)

def fetch(kind, arr, z, y, x):
    fn = os.path.join(CACHE, f"{kind}_{z}_{y}_{x}.npy")
    if os.path.exists(fn):
        return fn, "cached"
    zz, yy, xx = z // 4, y // 4, x // 4
    sub = arr[zz:zz + 64, yy:yy + 64, xx:xx + 64]
    if sub.shape != (64, 64, 64):
        pad = np.zeros((64, 64, 64), dtype=sub.dtype)
        pad[:sub.shape[0], :sub.shape[1], :sub.shape[2]] = sub
        sub = pad
    np.save(fn, sub)
    return fn, "fetched"

tasks = [("ct", ct, z, y, x) for (z, y, x) in tiles] + \
        [("surf", surf, z, y, x) for (z, y, x) in tiles]
t0 = time.time()
errs = []
with ThreadPoolExecutor(max_workers=8) as ex:
    futs = {ex.submit(fetch, *t): t for t in tasks}
    done = 0
    for f in as_completed(futs):
        t = futs[f]
        try:
            fn, status = f.result()
            done += 1
            if done % 20 == 0 or done == len(tasks):
                print(f"{done}/{len(tasks)} ({time.time()-t0:.0f}s)")
        except Exception as e:
            errs.append((t[0], t[2:], repr(e)))
            print("ERR", t[0], t[2:], repr(e))
print(f"done in {time.time()-t0:.0f}s, {len(errs)} errors")
if errs:
    sys.exit(1)
