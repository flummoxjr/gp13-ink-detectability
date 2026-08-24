"""Enumerate every published segment across GP scrolls with a tifxyz mesh."""
import json
import xml.etree.ElementTree as ET

import requests

BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
SCROLLS = {
    "PHerc1203": "20250820131727-9.362um-1.2m-113keV-masked.zarr",
    "PHerc1447": "20250521151220-8.640um-1.2m-116keV-masked.zarr",
    "PHerc0800": "20250521135224-8.640um-1.2m-116keV-masked.zarr",
}


def list_dirs(prefix):
    out, token = [], None
    while True:
        url = f"{BUCKET}/?list-type=2&prefix={prefix}&delimiter=/&max-keys=1000"
        if token:
            url += f"&continuation-token={requests.utils.quote(token)}"
        r = requests.get(url, timeout=40)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        out += [e.find("s3:Prefix", NS).text for e in root.findall("s3:CommonPrefixes", NS)]
        t = root.find("s3:NextContinuationToken", NS)
        if t is None:
            break
        token = t.text
    return out


def has_tifxyz(prefix):
    """Return the prefix holding x/y/z.tif for this segment, or None."""
    r = requests.get(f"{BUCKET}/?list-type=2&prefix={prefix}&max-keys=400", timeout=40)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    keys = [e.find("s3:Key", NS).text for e in root.findall("s3:Contents", NS)]
    for k in keys:
        if k.endswith("/x.tif") or k.endswith("x.tif"):
            base = k.rsplit("/", 1)[0] + "/"
            if any(kk.startswith(base) and kk.endswith("y.tif") for kk in keys) and \
               any(kk.startswith(base) and kk.endswith("z.tif") for kk in keys):
                return base
    return None


catalog = []
for scroll, vol in SCROLLS.items():
    for sub in ("segments/", "segments/raw/"):
        for d in list_dirs(f"{scroll}/{sub}"):
            if d.rstrip("/").endswith("/raw"):
                continue
            base = has_tifxyz(d)
            if base:
                catalog.append({"scroll": scroll, "volume": vol, "seg_dir": d, "tifxyz": base,
                                "name": d.rstrip("/").split("/")[-1]})
                print(f"{scroll}: {catalog[-1]['name']}", flush=True)

with open(r"C:\Users\benbl\Desktop\Vsuvious\trackD\runpod\segment_catalog.json", "w") as f:
    json.dump(catalog, f, indent=1)
print(f"\nTOTAL SEGMENTS WITH MESHES: {len(catalog)}")
for s in SCROLLS:
    print(f"  {s}: {sum(1 for c in catalog if c['scroll'] == s)}")
