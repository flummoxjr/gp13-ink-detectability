"""Mirror ink_9um label stores from the HF bucket, skipping the all-zero chunks.

  python hfsync.py <family> <segment> [<segment> ...]
     family = aligned-scrollprizeorg-21slices | native9-scrollprizeorg-21slices

For every store of the segment named in the manifest (inklabels, supervision_mask,
validation_mask if present): list the bucket tree (paginated via the Link header),
download every file whose xetHash is not the all-zero chunk, preserving the store
layout under $LABELS/<family>/<segment>/; verify .zattrs / .zgroup / 0/.zarray
sha256 against the manifest; report the annotated-plane non-zero count."""
import hashlib, json, os, re, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
import numpy as np
sys.path.insert(0, os.environ["SCRIPTS"])
import curvelib as cl

MAN = json.load(open(os.path.join(os.environ["SCRIPTS"], "manifest.json")))
LABELS = os.environ["LABELS"]
TREE = "https://huggingface.co/api/buckets/scrollprize/datasets/tree/"
RESOLVE = "https://huggingface.co/buckets/scrollprize/datasets/resolve/"
ZERO_HASHES = {MAN["conventions"]["aligned_label_all_zero_chunk"]["xetHash"],
               MAN["conventions"]["native_label_all_zero_chunk"]["xetHash"]}
UA = {"User-Agent": "curl/8"}
THREADS = int(os.environ.get("FETCH_THREADS", "32"))


def http(url, tries=5, timeout=120):
    waits = [0, 3, 10, 30, 60]
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read(), dict(r.headers)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None, {}
            if e.code == 429 or i == tries - 1:
                if i == tries - 1:
                    raise
            time.sleep(waits[i + 1] * (2 if e.code == 429 else 1))
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(waits[i + 1])


def list_tree(path):
    url = TREE + path + "?limit=1000&recursive=true"
    out = []
    while url:
        body, hdr = http(url, timeout=180)
        if body is None:
            raise RuntimeError(f"tree 404: {path}")
        out.extend(json.loads(body.decode()))
        link = hdr.get("Link") or hdr.get("link") or ""
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = m.group(1) if m else None
    return [e for e in out if e.get("type") == "file"]


def sync_store(family, seg, store_name, expect):
    rel_store = f"ink_9um/labels/{family}/{seg}/{seg}_{store_name}.zarr"
    local_store = os.path.join(LABELS, family, seg, f"{seg}_{store_name}.zarr")
    entries = list_tree(rel_store)
    keep = [e for e in entries if e.get("xetHash") not in ZERO_HASHES]
    skipped = len(entries) - len(keep)
    todo = []
    for e in keep:
        rel = e["path"][len(rel_store) + 1:]
        dest = os.path.join(local_store, rel)
        if os.path.exists(dest) and os.path.getsize(dest) == e["size"]:
            continue
        todo.append((e["path"], dest, e["size"]))
    cl.say(f"HFSYNC {seg}/{store_name}: {len(entries)} files listed, {skipped} all-zero skipped, "
           f"{len(todo)} to fetch")
    done = [0]
    failed = []

    def one(item):
        path, dest, size = item
        try:
            body = None
            for attempt in range(4):
                body, _ = http(RESOLVE + path)
                if body is not None:
                    break
                time.sleep(5 * (attempt + 1))       # listed-but-404: xet propagation lag, retry
            if body is None:
                raise RuntimeError(f"404 on listed file {path}")
            if len(body) != size:
                raise RuntimeError(f"size mismatch {path}: {len(body)} != {size}")
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            tmp = dest + ".part"
            with open(tmp, "wb") as f:
                f.write(body)
            os.replace(tmp, dest)
            done[0] += 1
            if done[0] % 500 == 0:
                cl.say(f"HFSYNC {seg}/{store_name}: {done[0]}/{len(todo)}")
        except Exception as e:                      # collect; retried below at low concurrency
            failed.append((item, f"{type(e).__name__}: {str(e)[:160]}"))
    with ThreadPoolExecutor(max_workers=THREADS) as ex:
        list(ex.map(one, todo))
    if failed:
        cl.say(f"HFSYNC {seg}/{store_name}: {len(failed)} files failed at {THREADS} threads "
               f"(first: {failed[0][1]}); retrying them at 4 threads after 30 s")
        time.sleep(30)
        retry_items = [f[0] for f in failed]
        failed.clear()
        with ThreadPoolExecutor(max_workers=4) as ex:
            list(ex.map(one, retry_items))
        if failed:
            cl.say(f"HFSYNC {seg}/{store_name}: STILL FAILING {len(failed)}: {failed[0][1]}")
            raise RuntimeError(f"{seg}/{store_name}: {len(failed)} files could not be fetched")
    # metadata gates
    for fn, key in ((".zattrs", "zattrs"), ("0/.zarray", "zarray_0")):
        p = os.path.join(local_store, fn)
        assert os.path.exists(p), f"{seg}/{store_name}: missing {fn}"
        got = hashlib.sha256(open(p, "rb").read()).hexdigest()
        exp = expect.get(key, {}).get("sha256")
        if exp:
            assert got == exp, f"{seg}/{store_name}/{fn} sha256 {got[:12]} != manifest {exp[:12]}"
    assert os.path.exists(os.path.join(local_store, ".zgroup")), f"{seg}/{store_name}: missing .zgroup"
    return local_store, len(entries), skipped


def plane_stats(local_store, z):
    import zarr
    a = zarr.open(local_store, mode="r")["0"]
    plane = a[z]
    return int((plane > 0).sum()), list(plane.shape)


def main():
    family = sys.argv[1]
    segs = sys.argv[2:]
    block = MAN["labels_hf"]["kept_aligned"] if family.startswith("aligned") else MAN["labels_hf"]["heldout_native_eval"]
    report = {}
    for seg in segs:
        spec = block[seg]
        rep = {}
        for store_name, expect in spec["stores"].items():
            if expect is None:
                continue
            local_store, n, skipped = sync_store(family, seg, store_name, expect)
            zplane = int(expect.get("annotated_plane_z", 10))
            nz, shape = plane_stats(local_store, zplane)
            rep[store_name] = dict(files=n, skipped_zero=skipped, plane_z=zplane, plane_nonzero=nz, shape=shape)
            cl.say(f"HFSYNC {seg}/{store_name}: OK sha256 metadata; plane z={zplane} nonzero={nz} shape={shape}")
        assert rep["inklabels"]["shape"] == rep["supervision_mask"]["shape"], (seg, rep)
        report[seg] = rep
    p = os.path.join(cl.RESULTS, f"labels_{family.split('-')[0]}.json")
    old = json.load(open(p)) if os.path.exists(p) else {}
    old.update(report)
    json.dump(old, open(p, "w"), indent=1)


if __name__ == "__main__":
    main()
