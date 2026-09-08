"""Assemble pod_betB_c2a.sh: p2a_v3 machinery (verbatim: status server, helpers, pyrun = villa-pin uv env)
+ the C2a parts; embed curvelib + the w035 labels (from p2a_v3), the renderer (runpod/), the mesh
transform (hunt/), the C2a programs, the selection and the prereg; validate (bash -n, py_compile, DRY walk)."""
import hashlib, json, os, py_compile, subprocess, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, "parts")
TRACKD = os.path.normpath(os.path.join(HERE, "..", ".."))
P2A = os.path.join(TRACKD, "bench", "p2a_v3", "pod_p2a_v3.sh")
OUT = os.path.join(HERE, "pod_betB_c2a.sh")


def read(p):
    return open(p, encoding="utf-8").read().replace("\r\n", "\n")


def extract(lines, dest):
    i = next(k for k, l in enumerate(lines) if l.startswith(f'cat > "$SCRIPTS/{dest}" <<\''))
    tag = lines[i].split("<<'")[1].rstrip("'")
    j = next(k for k in range(i + 1, len(lines)) if lines[k] == tag)
    return "\n".join(lines[i + 1:j]) + "\n"


def heredoc(dest, tag, body):
    assert not any(l.strip() == tag for l in body.split("\n")), (dest, tag)
    return f'cat > "$SCRIPTS/{dest}" <<\'{tag}\'\n{body if body.endswith(chr(10)) else body + chr(10)}{tag}\n\n'


def main():
    p2a = read(P2A).split("\n")
    i_l1 = next(i for i, l in enumerate(p2a) if l.startswith("# ===") and " L1 " in l)
    i_pre = next(i for i, l in enumerate(p2a) if l.startswith('cat > "$OUT/prereg.json"'))
    i_prsha = next(i for i, l in enumerate(p2a) if l.startswith("PRSHA="))
    i_ws = next(i for i, l in enumerate(p2a) if l.startswith("write_scripts() {"))
    i_sep = max(i for i in range(i_prsha, i_ws) if p2a[i].startswith("# ===="))
    mach_a = "\n".join(p2a[i_l1:i_pre]).replace("BOOT pod_p2a_v3", "BOOT pod_betB_c2a")
    mach_b = "\n".join(p2a[i_prsha:i_sep])
    assert 'pyrun() { (cd /workspace/villa/vesuvius && uv run --no-sync --extra models python "$@"); }' in mach_b
    prereg = read(os.path.join(P, "prereg.json")); json.loads(prereg)
    prereg_block = "cat > \"$OUT/prereg.json\" <<'PREREG_JSON'\n" + prereg + ("" if prereg.endswith("\n") else "\n") + "PREREG_JSON\n"
    selection = read(os.path.join(P, "selection.json")); sel = json.loads(selection)
    assert sel["n_H"] > 0 and sel["n_L"] == sel["n_H"], (sel["n_H"], sel["n_L"])
    lib = extract(p2a, "curvelib.py")
    labels = extract(p2a, "ctl_labels.b64")
    programs = [("curvelib.py", "PY_LIB", lib),
                ("ctl_labels.b64", "B64_LABELS", labels),
                ("render_tifxyz_sv.py", "PY_RENDER", read(os.path.join(TRACKD, "runpod", "render_tifxyz_sv.py"))),
                ("tifxyz_transform.py", "PY_TRANSFORM", read(os.path.join(TRACKD, "hunt", "tifxyz_transform.py"))),
                ("c2a_meshes.py", "PY_MESHES", read(os.path.join(P, "c2a_meshes.py"))),
                ("c2a_score.py", "PY_SCORE", read(os.path.join(P, "c2a_score.py"))),
                ("selection.json", "SELECTION_JSON", selection)]
    scripts = "write_scripts() {\n\n" + "".join(heredoc(*p) for p in programs) + "}\nwrite_scripts\nsay \"scripts written to $SCRIPTS\"\n"
    script = read(os.path.join(P, "header.sh")) + "\n" + mach_a + "\n" + prereg_block + mach_b + "\n\n" + scripts + read(os.path.join(P, "stages.sh"))
    open(OUT, "w", encoding="utf-8", newline="\n").write(script)
    print(f"wrote {OUT} ({len(script)} bytes, {script.count(chr(10))} lines)")
    r = subprocess.run(["bash", "-n", OUT], capture_output=True, text=True); assert r.returncode == 0, r.stderr
    print("bash -n: OK")
    tmpd = tempfile.mkdtemp(); lines = script.split("\n")
    for dest, tag, body in programs:
        if not dest.endswith(".py"):
            continue
        i = next(k for k, l in enumerate(lines) if l == f'cat > "$SCRIPTS/{dest}" <<\'{tag}\'')
        j = next(k for k in range(i + 1, len(lines)) if lines[k] == tag)
        p = os.path.join(tmpd, dest); open(p, "w", encoding="utf-8", newline="\n").write("\n".join(lines[i + 1:j]) + "\n")
        py_compile.compile(p, doraise=True)
    print("py_compile: OK")
    d = tempfile.mkdtemp()
    env = dict(os.environ, DRY="1", LINGER_EXIT="1", ROOT=d.replace("\\", "/") + "/c2a", PORT="8771",
               PYTHON_BIN=r"C:/Users/benbl/Desktop/Vsuvious/.venv/Scripts/python.exe")
    r = subprocess.run(["bash", OUT], capture_output=True, text=True, env=env, timeout=180)
    ok = "ALL DONE (DRY)" in r.stdout and "PREREG locked" in r.stdout
    warn = [l for l in (r.stderr or "").split("\n") if "warning" in l.lower()]
    print("DRY walk:", "OK" if ok else "FAILED"); print(r.stdout[-1200:] if not ok else "", r.stderr[-800:] if (not ok or warn) else "")
    assert ok and not warn, ("DRY walk failed or bash warned", warn)
    print("sha256", hashlib.sha256(script.encode()).hexdigest()[:16])


if __name__ == "__main__":
    main()
