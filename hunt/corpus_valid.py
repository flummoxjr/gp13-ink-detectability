#!/usr/bin/env python
"""Corpus-wide: for all 80 screened segments, compute mesh grid coverage and
combine with the survey's rendered nonzero fraction.  nonzero/valid is the
fraction of the mesh that landed on scanned material (the renderer writes 0
where the mesh is invalid OR where the masked volume is empty)."""
import json
import os

import numpy as np
import s3fs
from vesuvius.tifxyz import read_tifxyz

B = "vesuvius-challenge-open-data"
ROOT = r"C:\Users\benbl\Desktop\Vsuvious\trackD"
CACHE = os.path.join(ROOT, "hunt", "corpuscache")
OUT = os.path.join(ROOT, "hunt", "out")

if __name__ == "__main__":
    fs = s3fs.S3FileSystem(anon=True)
    cat = json.load(open(os.path.join(ROOT, "runpod", "segment_catalog.json")))
    surv = {x["name"]: x for x in json.load(open(os.path.join(ROOT, "out", "survey", "survey_all.json")))}
    os.makedirs(CACHE, exist_ok=True)
    rows = []
    for i, c in enumerate(cat):
        d = os.path.join(CACHE, c["name"])
        os.makedirs(d, exist_ok=True)
        try:
            for f in ("x.tif", "y.tif", "z.tif", "meta.json"):
                dst = os.path.join(d, f)
                if not os.path.exists(dst):
                    fs.get(f"{B}/{c['tifxyz']}{f}", dst)
            su = read_tifxyz(d, load_mask=False, validate=False)
            v = su._valid_mask
            r = {"name": c["name"], "scroll": c["scroll"],
                 "stored_shape": list(v.shape), "valid_frac": round(float(v.mean()), 4),
                 "full_shape": list(su.full_resolution_shape)}
            s = surv.get(c["name"], {})
            fwd = s.get("forward") or {}
            r["survey_nonzero_frac"] = fwd.get("nonzero_frac")
            r["survey_canvas"] = fwd.get("canvas")
            r["fwd_rev_r"] = s.get("fwd_rev_r")
            if r["survey_nonzero_frac"] is not None and r["valid_frac"] > 0:
                same = list(r["survey_canvas"] or []) == list(r["full_shape"])
                r["canvas_matches_mesh"] = bool(same)
                if same:
                    r["material_ratio"] = round(float(r["survey_nonzero_frac"] / r["valid_frac"]), 4)
            rows.append(r)
        except Exception as e:
            rows.append({"name": c["name"], "scroll": c["scroll"], "error": f"{type(e).__name__}: {e}"})
        if i % 10 == 0:
            print(i, c["name"], flush=True)
    json.dump(rows, open(os.path.join(OUT, "corpus_valid.json"), "w"), indent=1)
    print("wrote", os.path.join(OUT, "corpus_valid.json"))
