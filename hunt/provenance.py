#!/usr/bin/env python
"""How was each mesh made?  Pull the tracker's own provenance + self-reported cost."""
import json
import os

import numpy as np
import s3fs

CACHE = r"C:\Users\benbl\Desktop\Vsuvious\trackD\hunt\meshcache"
OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\hunt\out"
B = "vesuvius-challenge-open-data"

EXTRA = {  # published segments keep the growth provenance in mesh/intermediate/
    "w035": "PHerc0139/segments/20260317000000-w035_2026031718/mesh/intermediate/tifxyz_original/meta.json",
    "w032": "PHerc0139/segments/20260203000000-w032_2026020303/mesh/intermediate/tifxyz_original/meta.json",
}

if __name__ == "__main__":
    fs = s3fs.S3FileSystem(anon=True)
    segs = json.load(open(os.path.join(CACHE, "segs.json")))
    rows = []
    for s in segs:
        m = json.load(open(os.path.join(CACHE, s["key"], "meta.json")))
        if s["key"] in EXTRA:
            with fs.open(f"{B}/{EXTRA[s['key']]}") as f:
                m = {**json.load(f), **{k: v for k, v in m.items() if k not in ("bbox",)}}
        r = {"key": s["key"], "scroll": s["scroll"], "role": s["role"]}
        r["grown_on_volume"] = m.get("volume", "(scroll volume, native)")
        r["seed_surface_id"] = m.get("seed_surface_id")
        r["source"] = m.get("source", "vc_grow_seg_from_seed" if "max_gen" in m else "?")
        r["max_gen"] = m.get("max_gen")
        r["elapsed_time_s"] = round(float(m.get("elapsed_time_s", np.nan)), 2)
        r["avg_cost"] = m.get("avg_cost")
        gmc = m.get("gen_max_cost")
        if gmc:
            g = np.asarray(gmc, float)
            g = g[np.isfinite(g)]
            r["gen_max_cost_final"] = float(g[-1])
            r["gen_max_cost_max"] = float(g.max())
            early = np.median(g[: max(3, len(g) // 10)])
            r["cost_blowup_ratio"] = round(float(g.max() / max(early, 1e-18)), 1)
            over = np.nonzero(g > 10 * max(early, 1e-18))[0]
            r["first_gen_cost_10x"] = int(over[0]) if over.size else None
            r["n_gens_recorded"] = int(len(g))
        gac = m.get("gen_avg_cost")
        if gac:
            g = np.asarray(gac, float)
            r["gen_avg_cost_final"] = float(g[-1])
            r["gen_avg_cost_first"] = float(g[0])
        p = m.get("vc_gsfs_params")
        if p:
            r["direction_fields"] = [d.get("zarr") for d in p.get("direction_fields", [])]
            r["gsfs_generations"] = p.get("generations")
            r["gsfs_mode"] = m.get("vc_gsfs_mode")
        rows.append(r)
        print(json.dumps(r), flush=True)
    json.dump(rows, open(os.path.join(OUT, "provenance.json"), "w"), indent=1)
    print("wrote", os.path.join(OUT, "provenance.json"))
