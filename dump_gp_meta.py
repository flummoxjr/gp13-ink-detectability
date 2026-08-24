"""Dump GP-scroll volume nodes + search for Paganin/processing params across the catalog."""
import json
import os

META = r"C:\Users\benbl\AppData\Local\Temp\claude\C--Users-benbl-Desktop-Vsuvious\b3441997-0118-49b2-8364-cbdf28fc6397\scratchpad\metadata_plain.json"
OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\meta"

with open(META, encoding="utf-8") as f:
    data = json.load(f)
samples = data["samples"]

GP = ["PHerc0125", "PHerc0191", "PHerc0211", "PHerc0257", "PHerc0268", "PHerc0358",
      "PHerc0800", "PHerc0813", "PHerc0826", "PHerc1218", "PHerc1447", "PHerc1545"]

for name in GP:
    with open(os.path.join(OUT, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(samples[name], f, indent=1)

# Volume summary for all GP scrolls incl. PHerc1203 (already dumped)
print("=== GP volume summary ===")
for name in GP + ["PHerc1203"]:
    for vid, v in samples[name].get("volumes", {}).items():
        p = v.get("properties", {})
        c = v.get("creation", {}).get("metadata", {})
        print(f"{name} {vid} | {p.get('pixel_size_um')}um {p.get('energy_keV')}keV "
              f"| shape={p.get('shape')} | win_f32=[{c.get('target_window_f32_min')},{c.get('target_window_f32_max')}] "
              f"| long_id={v.get('long_id')}")

# Hunt for Paganin / delta_beta / paganin keys anywhere in one scan node
def find_keys(obj, needles, path="", hits=None):
    if hits is None:
        hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if any(n in k.lower() for n in needles):
                hits.append((path + "/" + k, v if not isinstance(v, (dict, list)) else type(v).__name__))
            find_keys(v, needles, path + "/" + k, hits)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:5]):
            find_keys(v, needles, f"{path}[{i}]", hits)
    return hits

print("\n=== delta_beta / paganin / unsharp keys in PHerc0813 + PHerc1203 + PHercParis4 ===")
for name in ["PHerc0813", "PHerc1203", "PHercParis4"]:
    hits = find_keys(samples[name], ["paganin", "delta", "beta", "unsharp", "processing", "nabu", "recon"])
    print(name, "->", len(hits), "hits")
    for h in hits[:25]:
        print("  ", h)
