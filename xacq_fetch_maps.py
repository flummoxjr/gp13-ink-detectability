"""Fetch the ds8 ink maps for the 112 verified-independent cross-acquisition pairs.

Resumable: skips files already on disk with the right size. Throttled (0.5 s
between requests) so it never loads the machine or the bucket. ~90 MB total.
"""
import json, time, sys
from pathlib import Path
import requests
import xml.etree.ElementTree as ET

EP = "https://vesuvius-challenge-open-data.s3.dualstack.us-east-1.amazonaws.com/"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
ROOT = Path(r"C:\Users\benbl\Desktop\Vsuvious\trackD")
OUT = Path(r"D:\vesuvius-data\trackD\xacq_maps")

PAIRS = {
    "PHerc0139":   ("20260102150214", "20260413113053"),
    "PHerc1667":   ("20251217075048", "20260323082859"),
    "PHerc0814":   ("20260309142202", "20260521123630"),
    "PHercParis4": ("20260411134726", "20260608103018"),
}

def s3keys(prefix):
    keys, token = [], None
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        r = requests.get(EP, params=params, timeout=60)
        root = ET.fromstring(r.content)
        keys += [(k.find(NS + "Key").text, int(k.find(NS + "Size").text))
                 for k in root.iter(NS + "Contents")]
        t = root.find(NS + "NextContinuationToken")
        if t is None:
            return keys
        token = t.text

def main():
    segs = json.loads((ROOT / "out" / "xacq_segments.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    manifest, fetched, skipped, missing = [], 0, 0, 0
    for sample, (va, vb) in PAIRS.items():
        for seg in segs[sample]["both"]:
            keys = s3keys(f"{sample}/segments/{seg}/ink-detection/")
            ds8 = [(k, sz) for k, sz in keys if k.endswith("-ds8.jpg")]
            for vid, tag in ((va, "A"), (vb, "B")):
                match = [(k, sz) for k, sz in ds8 if vid in k]
                if not match:
                    print(f"MISSING {sample}/{seg} {tag}", flush=True)
                    missing += 1
                    continue
                # newest if several
                k, sz = sorted(match)[-1]
                dest = OUT / f"{sample}__{seg}__{tag}.jpg"
                manifest.append({"sample": sample, "seg": seg, "arm": tag,
                                 "key": k, "size": sz, "file": dest.name})
                if dest.exists() and dest.stat().st_size == sz:
                    skipped += 1
                    continue
                r = requests.get(EP + k, timeout=120)
                r.raise_for_status()
                dest.write_bytes(r.content)
                fetched += 1
                if fetched % 20 == 0:
                    print(f"fetched {fetched} (skipped {skipped})", flush=True)
                time.sleep(0.5)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"DONE fetched={fetched} skipped={skipped} missing={missing} "
          f"manifest={len(manifest)}", flush=True)

if __name__ == "__main__":
    main()
