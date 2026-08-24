"""Probe S3 layout of the Paris 4 45.5um dual-energy volumes (anonymous HTTPS listing)."""
import requests
import xml.etree.ElementTree as ET

BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def list_prefix(prefix, delimiter="/", max_keys=200):
    r = requests.get(
        f"{BUCKET}/?list-type=2&prefix={prefix}&delimiter={delimiter}&max-keys={max_keys}",
        timeout=30,
    )
    r.raise_for_status()
    root = ET.fromstring(r.text)
    dirs = [e.find("s3:Prefix", NS).text for e in root.findall("s3:CommonPrefixes", NS)]
    files = [
        (e.find("s3:Key", NS).text, int(e.find("s3:Size", NS).text))
        for e in root.findall("s3:Contents", NS)
    ]
    return dirs, files


print("=== PHercParis4/ top ===")
d, f = list_prefix("PHercParis4/")
print("dirs:", d)
print("files:", f[:10])

print("\n=== PHercParis4/volumes/ ===")
d, f = list_prefix("PHercParis4/volumes/")
print("dirs:", d)
for k, s in f[:10]:
    print(" file:", k, s)

for vid in ["20260310170716", "20260310173927"]:
    hits = [x for x in d if vid in x]
    print(f"\n{vid} match:", hits)
    if hits:
        vd, vf = list_prefix(hits[0])
        print(" inside dirs:", vd)
        print(" inside files:", [(k.split('/')[-1], s) for k, s in vf[:12]])
        # fetch zarr metadata (v2 .zattrs / v3 zarr.json)
        for metaname in ["zarr.json", ".zattrs", ".zgroup"]:
            r = requests.get(f"{BUCKET}/{hits[0]}{metaname}", timeout=30)
            print(f"  GET {metaname}: {r.status_code}")
            if r.status_code == 200:
                print("  ", r.text[:900].replace("\n", " "))
                break
        # level 0 array metadata
        for lvl_meta in ["0/zarr.json", "0/.zarray"]:
            r = requests.get(f"{BUCKET}/{hits[0]}{lvl_meta}", timeout=30)
            print(f"  GET {lvl_meta}: {r.status_code}")
            if r.status_code == 200:
                print("  ", r.text[:700].replace("\n", " "))
                break
