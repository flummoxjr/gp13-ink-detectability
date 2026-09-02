"""Fetch the 500p2a raster labels + the exact zarr chunks covering the three
windows from the HF bucket (public, verified live). Threaded, resumable."""
import json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.environ["SCRIPTS"])
import curvelib as cl

CH = 128
CHUNKDIR = os.path.join(cl.DATA, "p2a_chunks")
UA = {"User-Agent": "curl/8"}

def http_get(url, tries=4):
    waits = [0, 5, 15, 45]
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        except Exception as e:
            if i == tries - 1:
                raise
            time.sleep(waits[i + 1])

def fetch_file(rel, dest, expect_bytes=None):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return False
    b = http_get(cl.BUCKET + "/" + rel)
    if expect_bytes is not None and len(b) != expect_bytes:
        raise AssertionError(f"{rel}: got {len(b)} bytes, expected {expect_bytes}")
    tmp = dest + ".part"
    with open(tmp, "wb") as f:
        f.write(b)
    os.replace(tmp, dest)
    return True

def chunk_keys():
    keys = set()
    for w in cl.WINDOWS.values():
        cy0, cy1 = w["y0"] // CH, (w["y0"] + w["size"] - 1) // CH
        cx0, cx1 = w["x0"] // CH, (w["x0"] + w["size"] - 1) // CH
        for cy in range(cy0, cy1 + 1):
            for cx in range(cx0, cx1 + 1):
                keys.add(f"0.{cy}.{cx}")
    return sorted(keys)

def main():
    os.makedirs(CHUNKDIR, exist_ok=True)
    za = json.loads(http_get(cl.BUCKET + "/500p2a.zarr/0/.zarray").decode())
    assert za["shape"] == list(cl.P2A_SHAPE), za
    assert za["chunks"] == list(cl.P2A_CHUNK), za
    assert za["dtype"] == "|u1" and za["compressor"]["id"] == "blosc", za
    sep = za.get("dimension_separator", ".")
    assert sep == ".", f"unexpected dimension_separator {sep!r}"
    with open(os.path.join(cl.DATA, "p2a_zarray.json"), "w") as f:
        json.dump(za, f)
    cl.say("EXPB_FETCH .zarray parsed and matches embedded expectation "
           "(shape/chunks/dtype/compressor/separator)")
    fetch_file("500p2a_inklabels.tif",
               os.path.join(cl.DATA, "500p2a_inklabels.tif"),
               cl.P2A_LABEL_BYTES)
    fetch_file("500p2a_supervision_mask.tif",
               os.path.join(cl.DATA, "500p2a_supervision_mask.tif"),
               cl.P2A_MASK_BYTES)
    cl.say("EXPB_FETCH raster labels fetched (byte sizes exact)")
    keys = chunk_keys()
    todo = [k for k in keys
            if not (os.path.exists(os.path.join(CHUNKDIR, k))
                    and os.path.getsize(os.path.join(CHUNKDIR, k)) > 0)]
    cl.say(f"EXPB_FETCH {len(keys)} chunks total, {len(todo)} to fetch "
           f"(8 threads, resumable)")
    done = [0]
    def one(k):
        fetch_file("500p2a.zarr/0/" + k, os.path.join(CHUNKDIR, k))
        done[0] += 1
        if done[0] % 200 == 0:
            cl.say(f"EXPB_FETCH progress {done[0]}/{len(todo)}")
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(one, todo))
    missing = [k for k in keys
               if not (os.path.exists(os.path.join(CHUNKDIR, k))
                       and os.path.getsize(os.path.join(CHUNKDIR, k)) > 0)]
    assert not missing, f"missing chunks after fetch: {missing[:5]}..."
    cl.say(f"EXPB_FETCH complete: {len(keys)} chunks on disk")

if __name__ == "__main__":
    main()
