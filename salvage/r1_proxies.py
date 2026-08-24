"""r1_proxies.py — reframe analyst, H6 step 1.

Per-tile independent condition proxies for every scored tile (n=29,748):
  - ct_std_all   : std of the TRUE CT L4 16^3 block (all voxels)   [streamed]
  - ct_std_mat   : std of CT L4 block restricted to material (>5)  [streamed]
  - mat_frac_l4  : fraction of L4 block with CT>5 (independent fill check)
  - ct_mean_l4   : mean of L4 block (validation against tiles.parquet meanct)
  - surf_frac    : fraction of SURF-m7 L2 block (same 16^3 grid) > 127
                   = local sheet-surface density from the released binary mask
  - surf_fracnz / surf_meannz : graded-value variants if the mask is not binary
  - ct_std_l5    : std of the cached L5 8^3 block (cheap proxy, for comparison)
Output: proxies.parquet (merged onto scored tiles).

CT L4 and SURF L2 share the same (947,1656,1656) grid; 1 tile = 16^3 voxels.
CT chunks 128^3 raw uint8; SURF chunks 192^3 blosc.
"""
import io, json, os, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import requests
from numcodecs import blosc

OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\salvage"
CT4 = ("https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1203/volumes/"
       "20260319130212-2.403um-0.2m-77keV-masked.zarr/4")
SF2 = ("https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1203/representations/"
       "predictions/surfaces/20260319130212-surface-20260413222639-surface-m7-L2-th0.2.zarr/2")

df = pd.read_parquet(os.path.join(OUT, "tiles.parquet"))
sc = df[~df.skipped].copy().reset_index(drop=True)
sc["ti"], sc["tj"], sc["tk"] = sc.z // 256, sc.y // 256, sc.x // 256
print(f"{len(sc)} scored tiles")

ses = requests.Session()

def fetch_chunk(base, idx, cshape, compressed):
    """Fetch one zarr chunk, trying '/' then '.' dimension separators."""
    for sep in ("/", "."):
        url = base + "/" + sep.join(str(i) for i in idx)
        r = ses.get(url, timeout=180)
        if r.status_code == 200:
            raw = r.content
            if compressed:
                raw = blosc.decompress(raw)
            a = np.frombuffer(raw, dtype=np.uint8)
            return a.reshape(cshape)
        if r.status_code != 404:
            r.raise_for_status()
    raise FileNotFoundError(f"{base} {idx}")

# ---- group tiles by chunk ----
ct_groups, sf_groups = {}, {}
for r in sc.itertuples():
    ct_groups.setdefault((r.ti // 8, r.tj // 8, r.tk // 8), []).append(r.Index)
    sf_groups.setdefault((r.ti // 12, r.tj // 12, r.tk // 12), []).append(r.Index)
print(f"CT chunks {len(ct_groups)}, SURF chunks {len(sf_groups)}")

n = len(sc)
res = {k: np.full(n, np.nan) for k in
       ["ct_mean_l4", "ct_std_all", "ct_std_mat", "mat_frac_l4",
        "surf_frac", "surf_fracnz", "surf_meannz"]}

def do_ct(idx):
    a = fetch_chunk(CT4, idx, (128, 128, 128), compressed=False).astype(np.float32)
    out = []
    for i in ct_groups[idx]:
        r = sc.iloc[i]
        z0, y0, x0 = (int(r.ti) % 8) * 16, (int(r.tj) % 8) * 16, (int(r.tk) % 8) * 16
        b = a[z0:z0 + 16, y0:y0 + 16, x0:x0 + 16]
        mat = b > 5
        mf = mat.mean()
        out.append((i, b.mean(), b.std(), b[mat].std() if mat.sum() >= 10 else np.nan, mf))
    return ("ct", out)

def do_sf(idx):
    a = fetch_chunk(SF2, idx, (192, 192, 192), compressed=True)
    out = []
    for i in sf_groups[idx]:
        r = sc.iloc[i]
        z0, y0, x0 = (int(r.ti) % 12) * 16, (int(r.tj) % 12) * 16, (int(r.tk) % 12) * 16
        b = a[z0:z0 + 16, y0:y0 + 16, x0:x0 + 16]
        nz = b > 0
        out.append((i, (b > 127).mean(), nz.mean(),
                    b[nz].mean() if nz.any() else np.nan))
    return ("sf", out)

t0 = time.time()
tasks = [(do_ct, k) for k in ct_groups] + [(do_sf, k) for k in sf_groups]
done, errs = 0, []
with ThreadPoolExecutor(max_workers=16) as ex:
    futs = {ex.submit(fn, k): (fn.__name__, k) for fn, k in tasks}
    for f in as_completed(futs):
        try:
            kind, out = f.result()
            if kind == "ct":
                for i, m, sa, sm, mf in out:
                    res["ct_mean_l4"][i] = m
                    res["ct_std_all"][i] = sa
                    res["ct_std_mat"][i] = sm
                    res["mat_frac_l4"][i] = mf
            else:
                for i, f127, fnz, mnz in out:
                    res["surf_frac"][i] = f127
                    res["surf_fracnz"][i] = fnz
                    res["surf_meannz"][i] = mnz
        except Exception as e:
            errs.append((futs[f], repr(e)))
            print("ERR", futs[f], repr(e))
        done += 1
        if done % 40 == 0 or done == len(tasks):
            print(f"{done}/{len(tasks)} chunks ({time.time()-t0:.0f}s)")
if errs:
    raise SystemExit(f"{len(errs)} chunk errors")

# ---- cheap L5-based std for comparison ----
ct5 = np.load(r"D:\vesuvius-data\trackD\ct1203_L5.npy", mmap_mode="r")
l5std = np.full(n, np.nan)
for slab, g in sc.groupby("ti"):
    z0 = int(slab) * 8
    plane = np.asarray(ct5[z0:z0 + 8]).astype(np.float32)
    for r in g.itertuples():
        b = plane[:, int(r.tj) * 8:(int(r.tj) + 1) * 8, int(r.tk) * 8:(int(r.tk) + 1) * 8]
        l5std[r.Index] = b.std()
res["ct_std_l5"] = l5std

for k, v in res.items():
    sc[k] = v
sc.to_parquet(os.path.join(OUT, "proxies.parquet"))

# ---- sanity ----
ok = ~np.isnan(sc.ct_mean_l4)
print(f"\nfilled: ct {ok.sum()}, surf {(~np.isnan(sc.surf_frac)).sum()}")
print("corr(ct_mean_l4, meanct from L5):", np.corrcoef(sc.ct_mean_l4[ok], sc.meanct[ok])[0, 1])
print("corr(mat_frac_l4, fill):", np.corrcoef(sc.mat_frac_l4[ok], sc["fill"][ok])[0, 1])
print("corr(ct_std_all true-L4, ct_std_l5 proxy):", np.corrcoef(sc.ct_std_all[ok], sc.ct_std_l5[ok])[0, 1])
print("surf_frac: min/med/max", np.nanmin(sc.surf_frac), np.nanmedian(sc.surf_frac), np.nanmax(sc.surf_frac))
print("surf graded? fracnz==frac127 everywhere:",
      bool(np.nanmax(np.abs(sc.surf_fracnz - sc.surf_frac)) == 0))
print(f"total {time.time()-t0:.0f}s")
