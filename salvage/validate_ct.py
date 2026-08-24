"""Validate L5-derived tile mean CT against 16^3 block means of streamed L4
chunks. One 128^3 L4 chunk covers 8x8x8 tiles (each tile = 16^3 at L4)."""
import json
import numpy as np
import pandas as pd
import requests

BASE = ("https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1203/"
        "volumes/20260319130212-2.403um-0.2m-77keV-masked.zarr/4")
df = pd.read_parquet(r"C:\Users\benbl\Desktop\Vsuvious\trackD\salvage\tiles.parquet")
sc = df[~df.skipped].copy()
sc["ti"], sc["tj"], sc["tk"] = sc.z // 256, sc.y // 256, sc.x // 256
sc["ci"], sc["cj"], sc["ck"] = sc.ti // 8, sc.tj // 8, sc.tk // 8

rng = np.random.default_rng(0)
chunks = (sc.groupby(["ci", "cj", "ck"]).size().reset_index(name="n"))
chunks = chunks[chunks.n >= 100]  # chunks well-populated with scored tiles
pick = chunks.sample(6, random_state=1)
print("validating chunks:\n", pick)

pairs = []
for _, row in pick.iterrows():
    ci, cj, ck = int(row.ci), int(row.cj), int(row.ck)
    url = f"{BASE}/{ci}/{cj}/{ck}"
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    buf = np.frombuffer(r.content, dtype=np.uint8)
    a = buf.reshape(128, 128, 128).astype(np.float64)
    m = a.reshape(8, 16, 8, 16, 8, 16).mean(axis=(1, 3, 5))  # (8,8,8) tile means
    sub = sc[(sc.ci == ci) & (sc.cj == cj) & (sc.ck == ck)]
    for _, t in sub.iterrows():
        l4v = m[int(t.ti) % 8, int(t.tj) % 8, int(t.tk) % 8]
        pairs.append((t.meanct, l4v))

p = np.array(pairs)
d = p[:, 0] - p[:, 1]
print(f"n tiles compared: {len(p)}")
print(f"corr(L5 8^3 mean, L4 16^3 mean) = {np.corrcoef(p[:,0], p[:,1])[0,1]:.6f}")
print(f"mean abs diff = {np.abs(d).mean():.4f} DN; max abs diff = {np.abs(d).max():.4f} DN")
print(f"mean diff (bias) = {d.mean():.4f} DN")
json.dump({"n": len(p), "corr": float(np.corrcoef(p[:,0], p[:,1])[0,1]),
           "mad": float(np.abs(d).mean()), "maxad": float(np.abs(d).max()),
           "bias": float(d.mean())},
          open(r"C:\Users\benbl\Desktop\Vsuvious\trackD\salvage\ct_proxy_validation.json", "w"),
          indent=1)
