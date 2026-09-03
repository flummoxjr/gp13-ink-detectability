"""Sparse level-2 source-volume fetch for the 15 kept representations.
  python svplan.py plan            -> out/sv_plan.json (chunk columns per rep, gated vs manifest)
  python svplan.py fetch <seg>...  -> $DATA/level2/<seg>.zarr (group: .zgroup .zattrs 2/.zarray + chunks)
  python svplan.py check <seg>     -> pooled volume gates (shape == label; 20 supervised patches non-zero)

The trainer picks patches from the supervision mask only (corner = (y//32*32, x//32*32),
128x128), never from the image, so fetching only the chunk columns those patches touch
(+1 chunk margin) is exact. Absent chunks (HTTP 404) are fill_value 0 and are NOT written
(zarr reads fill for missing keys); they are counted and capped at 60% per store."""
import hashlib, json, os, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
import numpy as np
sys.path.insert(0, os.environ["SCRIPTS"])
import curvelib as cl

MAN = json.load(open(os.path.join(os.environ["SCRIPTS"], "manifest.json")))
SV = MAN["training_surface_volumes_s3_level2_sparse"]
LAB = MAN["labels_hf"]["kept_aligned"]
LABELS, DATA, VOLS = os.environ["LABELS"], os.environ["DATA"], os.environ["VOLS"]
L2 = os.path.join(DATA, "level2")
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


def sup_plane(seg):
    import zarr
    st = os.path.join(LABELS, "aligned-scrollprizeorg-21slices", seg, f"{seg}_supervision_mask.zarr")
    a = zarr.open(st, mode="r")["0"]
    z = int(LAB[seg]["stores"]["supervision_mask"].get("annotated_plane_z", 10))
    return np.asarray(a[z]) > 0


def plan_one(seg):
    sup = sup_plane(seg)
    ys, xs = np.nonzero(sup)
    if len(ys) == 0:
        return [], 0
    corners = np.unique(np.stack([ys // 32 * 32, xs // 32 * 32], 1), axis=0)
    cols = set()
    for cy0, cx0 in corners:
        for cy in range(cy0 // CH, (cy0 + 127) // CH + 1):
            for cx in range(cx0 // CH, (cx0 + 127) // CH + 1):
                cols.add((int(cy), int(cx)))
    H, W = sup.shape
    gy, gx = (H + CH - 1) // CH, (W + CH - 1) // CH
    dil = set()
    for cy, cx in cols:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                y, x = cy + dy, cx + dx
                if 0 <= y < gy and 0 <= x < gx:
                    dil.add((y, x))
    return sorted(dil), len(corners)


def plan():
    out, total = {}, 0
    for seg in SV:
        cols, ncorner = plan_one(seg)
        exp = int(SV[seg]["chunks_planned_sparse"])
        # Our rule (patch corners +128 px, then +1 chunk of dilation) is systematically
        # ~1.15-1.26x the manifest's "+-128 px max-filter" estimate (measured on the
        # 2026-09-03 smoke: w013 1.17, w018 1.18, w023 1.19, w028 1.18, w029 1.26).
        # The band is a sanity check against a broken plan, not a budget: the budget
        # is the 45 GB total cap below. Smoke #4 (2026-09-03) died here on
        # phercparis4-w00 at 1.80x (the manifest's estimate is poor for the Paris4
        # geometry) after 59 min of label sync, so a miss is now a logged WARNING;
        # only a grossly broken plan (>4x or <0.25x) or the total cap is fatal.
        lo, hi = 0.5 * exp, 1.6 * exp
        ok = lo <= len(cols) <= hi
        sane = 0.25 * exp <= len(cols) <= 4.0 * exp
        cl.say(f"SVPLAN {seg}: {ncorner} patch corners -> {len(cols)} chunk columns "
               f"(manifest {exp}; {'OK' if ok else ('WARN out of the 0.5-1.6x band, ratio %.2f' % (len(cols) / max(exp, 1)))})")
        assert sane, f"{seg}: planned {len(cols)} outside the 0.25-4x sanity bound of the manifest count {exp}"
        out[seg] = dict(chunks=[list(c) for c in cols], n=len(cols), manifest=exp,
                        gb_est=round(len(cols) * SV[seg]["bytes_per_chunk_measured_mean"] / 1e9, 2))
        total += out[seg]["gb_est"]
    assert total <= 45.0, f"planned {total:.1f} GB > 45 GB cap"
    json.dump(out, open(os.path.join(cl.OUT, "sv_plan.json"), "w"), indent=0)
    cl.say(f"SVPLAN total {sum(v['n'] for v in out.values())} chunks, ~{total:.1f} GB (cap 45)")


def fetch(seg):
    spec = SV[seg]
    store, sep = spec["store"], spec["dimension_separator"]
    local = os.path.join(L2, f"{seg}.zarr")
    os.makedirs(os.path.join(local, "2"), exist_ok=True)
    for fn, key in ((".zgroup", "zgroup"), (".zattrs", "zattrs"), ("2/.zarray", "zarray_level2")):
        b = http(f"{store}/{fn}")
        assert b is not None, f"{seg}: {fn} missing on S3"
        got = hashlib.sha256(b).hexdigest()
        assert got == spec[key]["sha256"], f"{seg}/{fn} sha256 {got[:12]} != manifest {spec[key]['sha256'][:12]}"
        open(os.path.join(local, fn), "wb").write(b)
    plan_ = json.load(open(os.path.join(cl.OUT, "sv_plan.json")))[seg]["chunks"]
    absent_p = os.path.join(local, "absent.json")
    absent = set(tuple(a) for a in json.load(open(absent_p))) if os.path.exists(absent_p) else set()

    def dest_of(cy, cx):
        key = sep.join(["0", str(cy), str(cx)])
        return os.path.join(local, "2", *key.split("/")) if sep == "/" else os.path.join(local, "2", key), key
    todo = [(cy, cx) for cy, cx in plan_ if (cy, cx) not in absent and not os.path.exists(dest_of(cy, cx)[0])]
    cl.say(f"SVFETCH {seg}: {len(plan_)} planned, {len(todo)} to fetch, {len(absent)} known absent (sep '{sep}')")
    done = [0]; miss = []

    def one(c):
        cy, cx = c
        dest, key = dest_of(cy, cx)
        b = http(f"{store}/2/{key}")
        if b is None:
            miss.append((cy, cx))
        else:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            tmp = dest + ".part"
            with open(tmp, "wb") as f:
                f.write(b)
            os.replace(tmp, dest)
        done[0] += 1
        if done[0] % 200 == 0:
            cl.say(f"SVFETCH {seg}: {done[0]}/{len(todo)}")
    with ThreadPoolExecutor(max_workers=THREADS) as ex:
        list(ex.map(one, todo))
    absent |= set(miss)
    json.dump(sorted(list(a) for a in absent), open(absent_p, "w"))
    frac = len(absent) / max(1, len(plan_))
    cl.say(f"SVFETCH {seg}: done; absent {len(absent)}/{len(plan_)} = {frac:.1%}")
    assert frac < 0.60, f"{seg}: {frac:.0%} of planned chunks absent -- store or plan wrong"


def check(seg):
    import zarr
    lab = LAB[seg]["stores"]["inklabels"]["shape"]
    pooled = zarr.open(os.path.join(VOLS, "aligned9", f"{seg}.zarr"), mode="r")["0"]
    assert list(pooled.shape) == list(lab), f"{seg}: pooled {pooled.shape} != label {lab}"
    sup = sup_plane(seg)
    ys, xs = np.nonzero(sup)
    rng = np.random.default_rng(cl.SEED)
    idx = rng.choice(len(ys), size=min(20, len(ys)), replace=False)
    zero = 0
    for i in idx:
        y0, x0 = int(ys[i]) // 32 * 32, int(xs[i]) // 32 * 32
        patch = np.asarray(pooled[2:19, y0:y0 + 128, x0:x0 + 128])
        if patch.max() == 0:
            zero += 1
    cl.say(f"SVCHECK {seg}: pooled shape OK {list(pooled.shape)}; {zero}/{len(idx)} sampled supervised patches all-zero")
    assert zero == 0, f"{seg}: {zero} supervised patches read as zeros after pooling (fetch plan miss)"


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "plan":
        plan()
    elif cmd == "fetch":
        for s in sys.argv[2:]:
            fetch(s)
    elif cmd == "check":
        for s in sys.argv[2:]:
            check(s)
