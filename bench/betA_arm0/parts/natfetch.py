"""Native PHerc0139 eval crops (the held-out tier): for each of w035/w039/w040/w041/w044,
crop = supervision bbox (plane z=14) padded to 128-multiples (w035 forced to the p2a_v3
control crop rows 512:2944 cols 384:3072), fetched as raw level-0 chunks from S3 and
written as a 28-layer zarr; the crop's ink/sup label planes saved as npy."""
import json, os, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
import numpy as np
sys.path.insert(0, os.environ["SCRIPTS"])
import curvelib as cl

MAN = json.load(open(os.path.join(os.environ["SCRIPTS"], "manifest.json")))
EV = MAN["eval_native_volumes_s3_level0"]
LAB = MAN["labels_hf"]["heldout_native_eval"]
LABELS, NATIVE = os.environ["LABELS"], os.environ["NATIVE"]
UA = {"User-Agent": "curl/8"}
THREADS = int(os.environ.get("FETCH_THREADS", "32"))
CH = 128


def http(url, tries=5, timeout=300):
    waits = [0, 3, 10, 30, 60]
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if i == tries - 1:
                raise
            time.sleep(waits[i + 1])
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(waits[i + 1])


def label_planes(w):
    import zarr
    fam = "native9-scrollprizeorg-21slices"
    z = int(LAB[w]["stores"]["inklabels"].get("annotated_plane_z", 14))
    ink = np.asarray(zarr.open(os.path.join(LABELS, fam, w, f"{w}_inklabels.zarr"), mode="r")["0"][z]) > 0
    sup = np.asarray(zarr.open(os.path.join(LABELS, fam, w, f"{w}_supervision_mask.zarr"), mode="r")["0"][z]) > 0
    return ink, sup


def crop_for(w, sup):
    if w == "w035":
        return tuple(EV["w035"]["eval_crop_rows_cols"])
    ys, xs = np.nonzero(sup)
    H, W = sup.shape
    y0 = max(0, (int(ys.min()) - 128) // CH * CH); x0 = max(0, (int(xs.min()) - 128) // CH * CH)
    y1 = min(H, -(-(int(ys.max()) + 129) // CH) * CH); x1 = min(W, -(-(int(xs.max()) + 129) // CH) * CH)
    return (y0, y1, x0, x1)


def build(w):
    spec = EV[w]
    store, nz = spec["store"], spec["shape"][0]
    ink, sup = label_planes(w)
    y0, y1, x0, x1 = crop_for(w, sup)
    Hc, Wc = y1 - y0, x1 - x0
    keys = [(cy, cx) for cy in range(y0 // CH, (y1 - 1) // CH + 1) for cx in range(x0 // CH, (x1 - 1) // CH + 1)]
    cdir = os.path.join(NATIVE, f"{w}_chunks"); os.makedirs(cdir, exist_ok=True)
    todo = [k for k in keys if not os.path.exists(os.path.join(cdir, f"{k[0]}_{k[1]}"))]
    cl.say(f"NATIVE {w}: crop rows {y0}:{y1} cols {x0}:{x1} ({Hc}x{Wc}), {len(keys)} chunks, {len(todo)} to fetch")
    absent = [0]

    def one(k):
        cy, cx = k
        b = http(f"{store}/0/0/{cy}/{cx}")
        tmp = os.path.join(cdir, f"{cy}_{cx}.part")
        if b is None:
            absent[0] += 1; open(tmp, "wb").close()
        else:
            assert len(b) == spec["bytes_per_chunk"], (w, k, len(b))
            open(tmp, "wb").write(b)
        os.replace(tmp, os.path.join(cdir, f"{cy}_{cx}"))
    with ThreadPoolExecutor(max_workers=THREADS) as ex:
        list(ex.map(one, todo))
    vol = np.zeros((nz, Hc, Wc), np.uint8)
    for cy, cx in keys:
        raw = open(os.path.join(cdir, f"{cy}_{cx}"), "rb").read()
        if not raw:
            continue
        arr = np.frombuffer(raw, np.uint8).reshape(nz, CH, CH)
        ys, xs = cy * CH - y0, cx * CH - x0
        ty0, ty1, tx0, tx1 = max(0, ys), min(Hc, ys + CH), max(0, xs), min(Wc, xs + CH)
        vol[:, ty0:ty1, tx0:tx1] = arr[:, ty0 - ys:ty1 - ys, tx0 - xs:tx1 - xs]
    zp = os.path.join(NATIVE, f"{w}_crop.zarr")
    if not os.path.exists(zp):
        cl.write_group_zarr(zp, vol)
    ci, cs = ink[y0:y1, x0:x1], sup[y0:y1, x0:x1]
    np.save(os.path.join(NATIVE, f"{w}_ink.npy"), ci); np.save(os.path.join(NATIVE, f"{w}_sup.npy"), cs)
    rec = dict(crop=[y0, y1, x0, x1], shape=[nz, Hc, Wc], n_pos=int((ci & cs).sum()), n_neg=int((cs & ~ci).sum()),
               zero_frac=float((vol[nz // 2] == 0).mean()), absent_chunks=absent[0])
    cl.say(f"NATIVE {w}: zarr ready {rec['shape']}, pos={rec['n_pos']} neg={rec['n_neg']} zero_frac={rec['zero_frac']:.3f}")
    assert rec["n_pos"] > 1000 and rec["zero_frac"] < 0.5, (w, rec)
    return rec


if __name__ == "__main__":
    res = {}
    for w in sys.argv[1:]:
        res[w] = build(w)
    p = os.path.join(cl.RESULTS, "native_crops.json")
    old = json.load(open(p)) if os.path.exists(p) else {}
    old.update(res); json.dump(old, open(p, "w"), indent=1)
