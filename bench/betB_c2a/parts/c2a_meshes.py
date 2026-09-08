"""Bet B C2a, stage `meshes`: fetch the published 9.362 um tifxyz meshes and re-express them in the
2.4 um volumes (tifxyz_transform.py), keeping only the ctl sub-crop / the C1-selected tiles.

  python c2a_meshes.py <selection.json> <meshes dir>

Writes <meshes dir>/ctl_w035/ and <meshes dir>/seg_<name>/ (x/y/z.tif + meta.json + geom.json) and
<meshes dir>/meshes.json. geom.json carries, per kept tile, its pixel box on the 2.4 um canvas:
input stored cell i <-> output stored cell (i + 0.5) * factor - 0.5; one output cell = 20 px.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import requests

HTTPB = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
SCRIPTS = os.environ.get("SCRIPTS", os.path.dirname(os.path.abspath(__file__)))
PYTHON = sys.executable


def say(msg):
    st = os.environ.get("STATUS")
    line = time.strftime("%Y-%m-%dT%H:%M:%SZ ", time.gmtime()) + msg
    print(line, flush=True)
    if st:
        with open(st, "a") as f:
            f.write(line + "\n")


def fetch(seg_dir, dest):
    os.makedirs(dest, exist_ok=True)
    for f in ("x.tif", "y.tif", "z.tif", "meta.json"):
        p = os.path.join(dest, f)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            continue
        url = f"{HTTPB}/{seg_dir.rstrip('/')}/{f}"
        for attempt in range(4):
            try:
                r = requests.get(url, timeout=300)
                r.raise_for_status()
                open(p, "wb").write(r.content)
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 3:
                    raise
                time.sleep(5 * (attempt + 1))
                print("retry", url, e, flush=True)


def transform(in_dir, out_dir, matrix_path, inverse, crop=None, keep_tiles=None, tile=9, margin=1):
    cmd = [PYTHON, os.path.join(SCRIPTS, "tifxyz_transform.py"), in_dir, out_dir, "--matrix", matrix_path]
    if inverse:
        cmd.append("--inverse")
    if crop:
        cmd += ["--crop"] + [str(int(v)) for v in crop]
    if keep_tiles:
        cmd += ["--keep-tiles", keep_tiles, "--tile", str(tile), "--margin", str(margin)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    print(out.stdout, out.stderr, flush=True)
    if out.returncode != 0:
        raise RuntimeError(f"tifxyz_transform failed: {out.stderr[-500:]}")
    note = json.load(open(os.path.join(out_dir, "meta.json")))["transform_note"]
    factor = float(note.split("factor=")[1].split()[0])
    return factor


def cell_box_px(i0, i1, j0, j1, factor, cell_px=20):
    """Input stored cells [i0,i1) x [j0,j1) -> output pixel box [R0,R1) x [C0,C1) on the 2.4 um canvas."""
    r = lambda c: int(round(((c + 0.5) * factor - 0.5) * cell_px))  # noqa: E731
    return [r(i0), r(i1), r(j0), r(j1)]


def main():
    sel = json.load(open(sys.argv[1]))
    root = sys.argv[2]
    raw = os.path.join(root, "raw")
    os.makedirs(raw, exist_ok=True)
    tile, margin = int(sel.get("tile", 9)), int(sel.get("margin", 1))
    mx0139 = os.path.join(root, "matrix_0139.json"); json.dump(sel["ctl"]["matrix"], open(mx0139, "w"))
    mx1203 = os.path.join(root, "matrix_1203.json"); json.dump(sel["matrix_1203"], open(mx1203, "w"))
    summary = dict(ctl=None, segments={}, tile=tile, margin=margin)

    # ---- ctl: w035 9.362 mesh -> 2.399 volume, sub-crop
    c = sel["ctl"]
    src = os.path.join(raw, "w035_9362")
    fetch(c["mesh"], src)
    outd = os.path.join(root, "ctl_w035")
    f = transform(src, outd, mx0139, bool(c.get("inverse", True)), crop=c["crop_cells"])
    r0, r1, c0, c1 = c["crop_cells"]
    y0, y1, x0, x1 = c["crop_px_9362"]                  # the label sub-crop, exact (fractional cells)
    geom = dict(kind="ctl", px_um=float(c["px_um"]), factor=f, crop_cells=c["crop_cells"], crop_px_9362=c["crop_px_9362"],
                label_crop=c["label_crop"], box_px=cell_box_px(y0 / 20.0, y1 / 20.0, x0 / 20.0, x1 / 20.0, f),
                mesh_box_px=cell_box_px(r0, r1, c0, c1, f), source_mesh=c["mesh"], matrix=sel["ctl"]["matrix"])
    json.dump(geom, open(os.path.join(outd, "geom.json"), "w"), indent=1)
    summary["ctl"] = geom
    say(f"MESH ctl_w035: cells [{r0},{r1})x[{c0},{c1}) of the 9.362 mesh -> box_px {geom['box_px']} at 2.399 um (factor {f:.5f})")

    # ---- 1203 segments
    for name, s in sel["segments"].items():
        src = os.path.join(raw, name)
        fetch(s["seg_dir"], src)
        outd = os.path.join(root, f"seg_{name}")
        kt = os.path.join(root, f"keep_{name}.json")
        json.dump([dict(ij=t["ij"]) for t in s["tiles"]], open(kt, "w"))
        f = transform(src, outd, mx1203, True, keep_tiles=kt, tile=tile, margin=margin)
        tiles = []
        for t in s["tiles"]:
            i, j = t["ij"]
            tiles.append(dict(ij=[i, j], set=t["set"], contrast=t["contrast"], box_px=cell_box_px(i, i + tile, j, j + tile, f)))
        geom = dict(kind="seg", name=name, px_um=float(sel["px_um_1203"]), factor=f, tile=tile, tiles=tiles, source_mesh=s["seg_dir"])
        json.dump(geom, open(os.path.join(outd, "geom.json"), "w"), indent=1)
        summary["segments"][name] = dict(n_tiles=len(tiles), n_H=sum(t["set"] == "H" for t in tiles), n_L=sum(t["set"] == "L" for t in tiles), factor=f)
        say(f"MESH seg_{name}: {len(tiles)} tiles (H {summary['segments'][name]['n_H']}, L {summary['segments'][name]['n_L']}) -> 2.403 um (factor {f:.5f})")
    json.dump(summary, open(os.path.join(root, "meshes.json"), "w"), indent=1)
    say(f"MESH done: ctl + {len(summary['segments'])} segments, {sum(v['n_tiles'] for v in summary['segments'].values())} tiles")


if __name__ == "__main__":
    main()
