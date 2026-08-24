"""TRACK E — does every published GP surface actually track the lamellae it sits on?

WHY THIS MATTERS TO THE HEADLINE NEGATIVE
-----------------------------------------
Section 2.4 reports that 0 of 71 published GP segments show text, and section 2.7 already
bounds that negative two ways: 23 of 66 unique surfaces are debug dumps, and 18 place part
of their surface outside the scanned volume. Tonight's PHerc0813 work exposes a third,
previously unmeasured defect: a surface can lie in perfectly good laminated material and
still be oriented ACROSS the sheets rather than along them.

That is not a cosmetic problem. The instrument renders a 21-slice surface volume by
sampling at surface + t*normal. If the mesh normal is oblique to the sheet normal, those
samples cross lamellae, the depth axis averages over different sheets, and the ink model
is handed mush. A null from such a segment says nothing about ink.

Calibration established on 9 meshes (hunt/alignment_control.py): published GP meshes sit
at a median 13.1 deg from the local sheet normal with 7 of 9 within 30 deg, against 60 deg
for two independent random directions -- but 2 of those 9 were at 48 and 59 deg. If that
rate holds corpus-wide, roughly a fifth of the screened surfaces carry an uninterpretable
result, and the report must say so.

WHAT THIS MEASURES, PER SEGMENT
-------------------------------
  angle_deg      mesh normal (tifxyz grid tangent cross-product, axial mean)
                 vs local sheet normal (leading structure-tensor eigenvector)
  separability   whether the material there is laminated at all (K2c statistic)
  fill           whether the cube is inside the scanned volume (the 2.7 defect)

then joins each segment to its screen outcome (corpus v2 z, empirical p, fwd/rev r) so
alignment can be tested against the result rather than merely reported beside it.

Everything streams anonymously from the public bucket. No GPU, no cost.
"""
import json
import os
import sys
import numpy as np
import tifffile
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from k2c_analyze import coh_med  # noqa: E402
from hunt.mesh_lamella_alignment import mesh_normal, sheet_normal  # noqa: E402

BUCKET = "vesuvius-challenge-open-data"
T = r"C:\Users\benbl\Desktop\Vsuvious\trackD"
MESHCACHE = r"D:\vesuvius-data\trackD\corpus_meshes"
CUBECACHE = r"D:\vesuvius-data\trackD\corpus_cubes"
OUT = os.path.join(T, "out", "k2c_separability", "corpus_alignment.json")
ROI = 256


def s3():
    import s3fs
    return s3fs.S3FileSystem(anon=True)


def fetch_mesh(fs, row):
    d = os.path.join(MESHCACHE, row["name"])
    os.makedirs(d, exist_ok=True)
    for f in ("x.tif", "y.tif", "z.tif"):
        lp = os.path.join(d, f)
        if os.path.exists(lp) and os.path.getsize(lp) > 0:
            continue
        rp = f"{BUCKET}/{row['tifxyz'].rstrip('/')}/{f}"
        try:
            fs.get(rp, lp)
        except Exception:
            return None
    return d


def cube_at(scroll, volume, centre_zyx):
    import zarr, fsspec
    url = f"https://{BUCKET}.s3.amazonaws.com/{scroll}/volumes/{volume}"
    z0 = zarr.open(fsspec.get_mapper(url), mode="r")["0"]
    sh = z0.shape
    o = tuple(int(np.clip(centre_zyx[i] - ROI // 2, 0, sh[i] - ROI)) for i in range(3))
    return o, np.asarray(z0[o[0]:o[0] + ROI, o[1]:o[1] + ROI, o[2]:o[2] + ROI])


def one(args):
    fs, row = args
    name = row["name"]
    cp = os.path.join(CUBECACHE, f"{name}.npy")
    try:
        d = fetch_mesh(fs, row)
        if d is None:
            return dict(name=name, scroll=row["scroll"], status="mesh fetch failed")
        mn, nverts = mesh_normal(d)
        if mn is None:
            return dict(name=name, scroll=row["scroll"], status="mesh degenerate", n_vertices=nverts)
        x = tifffile.imread(os.path.join(d, "x.tif")).astype(np.float64)
        y = tifffile.imread(os.path.join(d, "y.tif")).astype(np.float64)
        z = tifffile.imread(os.path.join(d, "z.tif")).astype(np.float64)
        # tifxyz marks invalid vertices with -1, NOT 0 (see ledger row 16)
        v = (x >= 0) & (y >= 0) & (z >= 0) & ~((x == 0) & (y == 0) & (z == 0))
        if v.sum() < 100:
            return dict(name=name, scroll=row["scroll"], status="no valid vertices")
        centre = (float(np.median(z[v])), float(np.median(y[v])), float(np.median(x[v])))
        if os.path.exists(cp):
            a = np.load(cp)
            o = None
        else:
            o, a = cube_at(row["scroll"], row["volume"], centre)
            np.save(cp, a)
        fill = float((a > 0).mean())
        rec = dict(name=name, scroll=row["scroll"], n_vertices=int(v.sum()),
                   valid_frac=round(float(v.mean()), 4),
                   centre_zyx=[round(c, 1) for c in centre], fill=fill, status="ok")
        if fill < 0.5:
            rec["status"] = "cube mostly outside scanned volume"
            return rec
        sn = sheet_normal(a)
        if sn is None:
            rec["status"] = "too few in-material blocks"
            return rec
        rec["angle_deg"] = float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(mn, sn)))))))
        rec["separability"] = float(coh_med(a))
        return rec
    except Exception as e:
        return dict(name=name, scroll=row["scroll"], status=f"error: {type(e).__name__}: {e}")


def main():
    os.makedirs(MESHCACHE, exist_ok=True)
    os.makedirs(CUBECACHE, exist_ok=True)
    cat = json.load(open(os.path.join(T, "runpod", "segment_catalog.json")))
    # de-duplicate: the catalogue has 80 rows over 66 unique surfaces
    seen, rows = set(), []
    for r in cat:
        if r["tifxyz"] in seen:
            continue
        seen.add(r["tifxyz"])
        rows.append(r)
    print(f"{len(cat)} catalogue rows -> {len(rows)} unique surfaces", flush=True)

    done = {}
    if os.path.exists(OUT):
        done = {r["name"]: r for r in json.load(open(OUT)).get("segments", [])
                if r.get("status") == "ok"}
        print(f"resuming: {len(done)} already measured", flush=True)

    fs = s3()
    todo = [r for r in rows if r["name"] not in done]
    out = list(done.values())
    with ThreadPoolExecutor(max_workers=4) as ex:
        for i, rec in enumerate(ex.map(one, [(fs, r) for r in todo]), 1):
            out.append(rec)
            ang = rec.get("angle_deg")
            print(f"[{i}/{len(todo)}] {rec['scroll']:10} {rec['name'][:34]:34} "
                  + (f"angle={ang:5.1f} deg  sep={rec.get('separability', float('nan')):.3f}"
                     if ang is not None else f"-- {rec['status']}"), flush=True)
            if i % 10 == 0:
                json.dump({"segments": out}, open(OUT, "w"), indent=1)

    ok = [r for r in out if r.get("angle_deg") is not None]
    summary = {}
    if ok:
        ang = np.array([r["angle_deg"] for r in ok])
        summary = dict(n_measured=len(ok), median_angle_deg=float(np.median(ang)),
                       n_within_30=int((ang < 30).sum()),
                       n_beyond_45=int((ang >= 45).sum()),
                       n_beyond_60=int((ang >= 60).sum()),
                       random_null_median_deg=60.0,
                       published_control_median_deg=13.1)
        print(f"\n=== {len(ok)} surfaces measured ===")
        print(f"  median angle to the sheets: {np.median(ang):.1f} deg "
              f"(9-mesh control sample: 13.1; random: 60.0)")
        print(f"  within 30 deg: {(ang < 30).sum()}/{len(ang)}")
        print(f"  45 deg or worse: {(ang >= 45).sum()}/{len(ang)}  "
              f"<- these surfaces' screen results are not interpretable")
        print(f"  60 deg or worse: {(ang >= 60).sum()}/{len(ang)}")
    json.dump({"summary": summary, "segments": out}, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
