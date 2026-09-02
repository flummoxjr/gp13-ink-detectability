"""CTL arms on PHerc0139 w035 (the in-domain control). argv[1]: fetch | build.
fetch: the 399 raw S3 chunks covering the label-bbox crop (threaded,
resumable; a 404 chunk is an all-zero chunk under fill_value 0 and is
recorded as such). build: ctl_native (as released), ctl_scalefault (x1.9504
in-plane and depth -- exactly v2's pitch error), ctl_half (x0.5)."""
import json, os, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
import numpy as np
sys.path.insert(0, os.environ["SCRIPTS"])
import curvelib as cl

CH = 128
CHUNKDIR = os.path.join(cl.DATA, "ctl_chunks")
UA = {"User-Agent": "curl/8"}
Y0, Y1, X0, X1 = cl.CTL_CROP
H, W = Y1 - Y0, X1 - X0
NZ = cl.CTL_SHAPE[0]

def http_get(url, tries=4):
    waits = [0, 5, 15, 45]
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=300) as r:
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

def chunk_keys():
    keys = []
    for cy in range(Y0 // CH, (Y1 - 1) // CH + 1):
        for cx in range(X0 // CH, (X1 - 1) // CH + 1):
            keys.append((cy, cx))
    return keys

def dest(cy, cx):
    return os.path.join(CHUNKDIR, f"{cy}_{cx}")

def fetch():
    os.makedirs(CHUNKDIR, exist_ok=True)
    za = json.loads(http_get(cl.CTL_SV + "/0/.zarray").decode())
    assert za["shape"] == list(cl.CTL_SHAPE), za
    assert za["chunks"] == list(cl.CTL_CHUNK), za
    assert za["dtype"] == "|u1" and za["compressor"] is None, za
    assert za.get("dimension_separator") == "/", za
    assert za.get("fill_value", 0) == 0, za
    cl.say("CTL_FETCH .zarray parsed and matches embedded expectation "
           "(shape/chunks/dtype/raw/separator)")
    keys = chunk_keys()
    assert len(keys) == 399, len(keys)
    todo = [k for k in keys if not os.path.exists(dest(*k))]
    cl.say(f"CTL_FETCH {len(keys)} chunks total, {len(todo)} to fetch "
           f"(8 threads, resumable)")
    done = [0]; missing = [0]
    def one(k):
        cy, cx = k
        b = http_get(f"{cl.CTL_SV}/0/0/{cy}/{cx}")
        tmp = dest(cy, cx) + ".part"
        if b is None:
            missing[0] += 1
            open(tmp, "wb").close()           # zero-length = all-zero chunk
        else:
            assert len(b) == cl.CTL_CHUNK_BYTES, (k, len(b))
            with open(tmp, "wb") as f:
                f.write(b)
        os.replace(tmp, dest(cy, cx))
        done[0] += 1
        if done[0] % 100 == 0:
            cl.say(f"CTL_FETCH progress {done[0]}/{len(todo)}")
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(one, todo))
    left = [k for k in keys if not os.path.exists(dest(*k))]
    assert not left, f"missing chunks after fetch: {left[:5]}..."
    cl.say(f"CTL_FETCH complete: {len(keys)} chunks on disk "
           f"({missing[0]} absent-on-server = all-zero)")

def assemble():
    vol = np.zeros((NZ, H, W), dtype=np.uint8)
    for cy, cx in chunk_keys():
        raw = open(dest(cy, cx), "rb").read()
        if not raw:
            continue
        arr = np.frombuffer(raw, np.uint8).reshape(NZ, CH, CH)
        ys, xs = cy * CH - Y0, cx * CH - X0
        ty0, ty1 = max(0, ys), min(H, ys + CH)
        tx0, tx1 = max(0, xs), min(W, xs + CH)
        vol[:, ty0:ty1, tx0:tx1] = arr[:, ty0 - ys:ty1 - ys, tx0 - xs:tx1 - xs]
    return vol

def build():
    pos, neg = cl.load_ctl_labels()        # asserts sha256 + exact counts
    cl.say(f"CTL_BUILD embedded labels verified: pos={int(pos.sum())} "
           f"neg={int(neg.sum())} crop={H}x{W}")
    vol = assemble()
    sub = vol[::4, ::4, ::4].astype(np.float32)
    stats = dict(mean=float(sub.mean()), std=float(sub.std()),
                 zero_frac=float((sub == 0).mean()), max=int(sub.max()))
    json.dump(stats, open(os.path.join(cl.RESULTS, "ctl_volume_stats.json"),
                          "w"), indent=1)
    cl.say(f"CTL_BUILD crop stats mean={stats['mean']:.2f} sd={stats['std']:.2f} "
           f"zero_frac={stats['zero_frac']:.4f} max={stats['max']}")
    assert stats["zero_frac"] < 0.20, ("crop is mostly empty", stats)
    assert stats["max"] == 255, ("crop never reaches 255", stats)
    zn = os.path.join(cl.DATA, "ctl_native.zarr")
    if not os.path.exists(zn):
        cl.write_group_zarr(zn, vol)
    cl.save_preview(vol[NZ // 2], os.path.join(cl.OUT, "previews",
                                               "ctl_native_midslice.png"), ds=2)
    cl.say("CTL_BUILD ctl_native.zarr ready (28x{}x{})".format(H, W))
    for name, factor in (("ctl_scalefault", cl.CTL_FAULT_FACTOR),
                         ("ctl_half", cl.CTL_HALF_FACTOR)):
        zp = os.path.join(cl.DATA, f"{name}.zarr")
        if os.path.exists(zp):
            continue
        shape = cl.ctl_arm_shape(factor)
        cl.say(f"CTL_BUILD {name}: x{factor:.4f} -> {shape}")
        tmp = os.path.join(cl.DATA, "tmp", f"{name}.f32")
        mm = cl.resample_stack(lambda z: vol[z], NZ, H, W, shape, tmp,
                               tag=name)
        v = np.clip(np.rint(np.asarray(mm)), 0, 255).astype(np.uint8)
        del mm; os.remove(tmp)
        cl.write_group_zarr(zp, v)
        cl.save_preview(v[shape[0] // 2], os.path.join(
            cl.OUT, "previews", f"{name}_midslice.png"), ds=2)
        del v
        cl.say(f"CTL_BUILD {name}.zarr ready {shape}")

if __name__ == "__main__":
    {"fetch": fetch, "build": build}[sys.argv[1]]()
