"""QC test 4: find released Paris 4 ink predictions on S3 and sample their
value distribution as the healthy in-domain reference."""
import json

import numpy as np
import zarr
import fsspec

OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\qc_live\qc_paris4_result.json"
fs = fsspec.filesystem("s3", anon=True)
root = "vesuvius-challenge-open-data/PHercParis4/representations/predictions"

res = {"listing": {}}


def walk(path, depth=0, maxdepth=3):
    try:
        entries = fs.ls(path)
    except Exception as e:
        res["listing"][path] = f"ERROR {e}"
        return
    names = [e.rsplit("/", 1)[-1] for e in entries]
    res["listing"][path] = names[:50]
    print(path, "->", names[:50], flush=True)
    if depth >= maxdepth:
        return
    for e in entries:
        name = e.rsplit("/", 1)[-1]
        if name.endswith(".zarr"):
            continue  # don't descend into zarr internals
        if "." not in name:  # likely a directory
            walk(e, depth + 1, maxdepth)


walk(root)

# find ink-looking zarrs
ink_zarrs = []
for path, names in res["listing"].items():
    if isinstance(names, list):
        for nm in names:
            if nm.endswith(".zarr") and ("ink" in nm.lower() or "ink" in path.lower()):
                ink_zarrs.append(path + "/" + nm)
res["ink_zarrs"] = ink_zarrs
print("ink zarrs:", ink_zarrs, flush=True)

samples = {}
for zp in ink_zarrs[:2]:
    try:
        g = zarr.open(fs.get_mapper(zp), mode="r")
        try:
            arr = g["0"]
        except (KeyError, TypeError):
            arr = g
        sh = arr.shape
        print(zp, sh, arr.dtype, flush=True)
        # sample a few interior chunks
        rng = np.random.default_rng(0)
        vals = []
        f05s, f08s, pmaxs = [], [], []
        for _ in range(8):
            c = [int(rng.integers(s // 4, 3 * s // 4)) for s in sh]
            blk = np.asarray(arr[c[0]:c[0] + 64, c[1]:c[1] + 64, c[2]:c[2] + 64]).astype(np.float32)
            if blk.max() <= 1.5:
                p = blk
            else:
                p = blk / 255.0
            vals.append(p.ravel()[::16])
            f05s.append(float((p > 0.5).mean()))
            f08s.append(float((p > 0.8).mean()))
            pmaxs.append(float(p.max()))
        v = np.concatenate(vals)
        samples[zp] = {"shape": list(sh), "dtype": str(arr.dtype),
                       "pct": {q: round(float(np.percentile(v, q)), 4)
                               for q in (50, 90, 99, 99.9)},
                       "block_f05": [round(x, 4) for x in f05s],
                       "block_f08": [round(x, 4) for x in f08s],
                       "block_pmax": [round(x, 3) for x in pmaxs]}
    except Exception as e:
        samples[zp] = f"ERROR {e}"
res["samples"] = samples
with open(OUT, "w") as f:
    json.dump(res, f, indent=1)
print(json.dumps(res.get("samples", {}), indent=1))
print("WROTE", OUT)
