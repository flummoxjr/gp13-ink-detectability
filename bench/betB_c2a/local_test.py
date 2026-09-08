"""Offline functional test of the C2a programs ($0, no GPU): the real mesh stage (public meshes over HTTP,
the real selection) and the scorer on synthetic maps. Mirrors the pod's env layout in a temp ROOT."""
import json, os, subprocess, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRACKD = os.path.normpath(os.path.join(HERE, "..", ".."))
PY = sys.executable
P2A = os.path.join(TRACKD, "bench", "p2a_v3", "pod_p2a_v3.sh")


def extract(lines, dest):
    i = next(k for k, l in enumerate(lines) if l.startswith(f'cat > "$SCRIPTS/{dest}" <<\''))
    tag = lines[i].split("<<'")[1].rstrip("'")
    j = next(k for k in range(i + 1, len(lines)) if lines[k] == tag)
    return "\n".join(lines[i + 1:j]) + "\n"


def main():
    root = tempfile.mkdtemp(prefix="c2a_test_")
    S = os.path.join(root, "scripts"); os.makedirs(S)
    for d in ("out/results", "out/maps", "out/previews", "meshes", "data", "out/preds", "var"):
        os.makedirs(os.path.join(root, d), exist_ok=True)
    p2a = open(P2A, encoding="utf-8").read().replace("\r\n", "\n").split("\n")
    open(os.path.join(S, "curvelib.py"), "w", encoding="utf-8", newline="\n").write(extract(p2a, "curvelib.py"))
    open(os.path.join(S, "ctl_labels.b64"), "w", encoding="utf-8", newline="\n").write(extract(p2a, "ctl_labels.b64"))
    for src, dst in ((os.path.join(TRACKD, "hunt", "tifxyz_transform.py"), "tifxyz_transform.py"),
                     (os.path.join(HERE, "parts", "c2a_meshes.py"), "c2a_meshes.py"),
                     (os.path.join(HERE, "parts", "c2a_score.py"), "c2a_score.py"),
                     (os.path.join(HERE, "parts", "selection.json"), "selection.json")):
        open(os.path.join(S, dst), "w", encoding="utf-8", newline="\n").write(open(src, encoding="utf-8").read())
    env = dict(os.environ, ROOT=root, OUT=os.path.join(root, "out"), DATA=os.path.join(root, "data"), PREDS=os.path.join(root, "out", "preds"),
               RESULTS=os.path.join(root, "out", "results"), STATUS=os.path.join(root, "out", "status.txt"), SCRIPTS=S, SEED="20260908",
               MESHES=os.path.join(root, "meshes"))
    # reuse cached meshes to save downloads
    raw = os.path.join(root, "meshes", "raw"); os.makedirs(raw, exist_ok=True)
    import shutil
    mc = os.path.join(TRACKD, "hunt", "meshcache")
    if os.path.isdir(os.path.join(mc, "0139_w035_9362")):
        shutil.copytree(os.path.join(mc, "0139_w035_9362"), os.path.join(raw, "w035_9362"), dirs_exist_ok=True)
    for n in os.listdir(mc):
        if n.startswith("1203_auto_grown_"):
            shutil.copytree(os.path.join(mc, n), os.path.join(raw, n[5:]), dirs_exist_ok=True)
    # ---- stage meshes (real)
    r = subprocess.run([PY, os.path.join(S, "c2a_meshes.py"), os.path.join(S, "selection.json"), os.path.join(root, "meshes")], env=env, capture_output=True, text=True)
    print(r.stdout[-2500:], r.stderr[-1500:])
    assert r.returncode == 0, "meshes stage failed"
    ms = json.load(open(os.path.join(root, "meshes", "meshes.json")))
    print("meshes.json:", json.dumps(ms)[:600])
    # ---- scorer on synthetic maps
    import tifffile
    geom = json.load(open(os.path.join(root, "meshes", "ctl_w035", "geom.json")))
    R0, R1, C0, C1 = geom["box_px"]; print("ctl box_px", geom["box_px"], "mesh_box_px", geom["mesh_box_px"])
    rng = np.random.default_rng(0)
    H, W = R1 + 64, C1 + 64
    ours = np.zeros((H, W), np.uint8); ref = np.zeros((H, W), np.uint8)
    base = rng.integers(20, 120, size=(R1 - R0, C1 - C0), dtype=np.uint8)
    # a bright blob (letters stand-in) so the tripwire has something on ink: we don't know where ink is, so paint a large blob
    ours[R0:R1, C0:C1] = base; ref[R0:R1, C0:C1] = np.clip(base.astype(int) + rng.integers(-5, 5, base.shape), 1, 255).astype(np.uint8)
    ours[R0 + 2000:R0 + 2600, C0 + 2000:C0 + 2600] = 250; ref[R0 + 2000:R0 + 2600, C0 + 2000:C0 + 2600] = 240
    fo, fr = os.path.join(root, "out", "preds", "ctl_fwd_8.tif"), os.path.join(root, "data", "ref.tif")
    tifffile.imwrite(fo, ours, compression="zlib"); tifffile.imwrite(fr, ref, compression="zlib")
    r = subprocess.run([PY, os.path.join(S, "c2a_score.py"), "ctl", fo, fr, os.path.join(root, "meshes", "ctl_w035"), "ctl_fwd_8"], env=env, capture_output=True, text=True)
    print("ctl rc", r.returncode, r.stdout[-800:], r.stderr[-1200:])
    res = json.load(open(os.path.join(root, "out", "results", "ctl_ctl_fwd_8.json")))
    print({k: res[k] for k in ("r_ds4_joint", "auc_fwd", "auc_ref_published", "blank_p99", "tripwire_components", "tripwire_on_ink", "n_pos", "n_neg")})
    rev = os.path.join(root, "out", "preds", "ctl_rev_8.tif"); tifffile.imwrite(rev, np.clip(ours.astype(int) // 2, 0, 255).astype(np.uint8), compression="zlib")
    r = subprocess.run([PY, os.path.join(S, "c2a_score.py"), "ctlrev", fo, rev, os.path.join(root, "meshes", "ctl_w035"), "ctl_8"], env=env, capture_output=True, text=True)
    print("ctlrev rc", r.returncode, r.stdout[-400:], r.stderr[-800:])
    # tiles on the smallest segment
    segs = sorted(ms["segments"].items(), key=lambda kv: kv[1]["n_tiles"])
    name = segs[0][0]; g = json.load(open(os.path.join(root, "meshes", f"seg_{name}", "geom.json")))
    boxes = np.array([t["box_px"] for t in g["tiles"]]); Hs, Ws = int(boxes[:, 1].max()) + 8, int(boxes[:, 3].max()) + 8
    fwd = np.zeros((Hs, Ws), np.uint8); rv = np.zeros((Hs, Ws), np.uint8)
    for t in g["tiles"]:
        a, b, c, d = t["box_px"]; fwd[a:b, c:d] = rng.integers(10, 200, size=(b - a, d - c), dtype=np.uint8); rv[a:b, c:d] = rng.integers(10, 60, size=(b - a, d - c), dtype=np.uint8)
    ff, rf = os.path.join(root, "out", "preds", f"seg_{name}_fwd.tif"), os.path.join(root, "out", "preds", f"seg_{name}_rev.tif")
    tifffile.imwrite(ff, fwd, compression="zlib"); tifffile.imwrite(rf, rv, compression="zlib")
    r = subprocess.run([PY, os.path.join(S, "c2a_score.py"), "tiles", ff, rf, os.path.join(root, "meshes", f"seg_{name}"), str(res["blank_p99"]), f"seg_{name}"], env=env, capture_output=True, text=True)
    print("tiles rc", r.returncode, r.stdout[-600:], r.stderr[-1200:])
    tr = json.load(open(os.path.join(root, "out", "results", f"tiles_seg_{name}.json")))
    print("tiles summary:", json.dumps(tr["summary"])[:500]); print("first tile:", json.dumps(tr["tiles"][0])[:400])
    print("TEST ROOT", root)


if __name__ == "__main__":
    main()
