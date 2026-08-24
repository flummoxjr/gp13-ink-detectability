"""m0_inventory.py — inventory unique sampled prob maps from round_1 + round_2.

Dedupes by (worker, tile). Verifies duplicate pulls are byte-identical.
Output: inventory.json with one record per unique sample.
"""
import json, os, hashlib
import numpy as np

QC = r"C:\Users\benbl\Desktop\Vsuvious\trackD\qc_live"
OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\salvage"

records = {}          # key (worker, z,y,x) -> record
dupes_checked = []

for rnd in ["round_1", "round_2"]:
    man = json.load(open(os.path.join(QC, rnd, "manifest.json")))
    for w, info in man["pods"].items():
        if "samples" not in info:
            continue
        for s in info["samples"]:
            z, y, x = s["tile"]
            key = (w, z, y, x)
            path = os.path.join(QC, rnd, s["file"])
            if not os.path.exists(path):
                print("MISSING FILE", path)
                continue
            rec = {
                "worker": w, "tile": [z, y, x], "round": rnd,
                "file": path, "pmax": s["pmax"], "f05": s["f05"],
                "f08": s["f08"], "fill": s["fill"],
            }
            if key in records:
                # verify identical content
                a = np.load(records[key]["file"])
                b = np.load(path)
                same = a.shape == b.shape and np.array_equal(a, b)
                dupes_checked.append({"key": list(key), "identical": bool(same)})
                records[key]["rounds"] = sorted(set(records[key].get("rounds", [records[key]["round"]]) + [rnd]))
                if not same:
                    print("WARN: duplicate pull differs:", key)
            else:
                rec["rounds"] = [rnd]
                records[key] = rec

inv = sorted(records.values(), key=lambda r: (r["tile"][0], r["tile"][1], r["tile"][2]))
# sanity: shapes
for r in inv:
    a = np.load(r["file"], mmap_mode="r")
    r["shape"] = list(a.shape)
    r["dtype"] = str(a.dtype)

print(f"unique samples: {len(inv)}")
print(f"duplicate pulls verified: {len(dupes_checked)}, all identical: {all(d['identical'] for d in dupes_checked)}")
zs = sorted(set(r["tile"][0] for r in inv))
print("z slabs:", zs)
print("pmax range:", min(r["pmax"] for r in inv), max(r["pmax"] for r in inv))
print("f05 range:", min(r["f05"] for r in inv), max(r["f05"] for r in inv))
fills = [r["fill"] for r in inv]
print("fill: min %.2f median %.2f" % (min(fills), float(np.median(fills))))

json.dump({"n": len(inv), "dupes_checked": dupes_checked, "samples": inv},
          open(os.path.join(OUT, "inventory.json"), "w"), indent=1)
print("wrote inventory.json")
