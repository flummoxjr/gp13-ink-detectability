"""Assemble pod_w059_c0.sh: p2a_v3 machinery (verbatim) + w059 parts; validate."""
import hashlib, json, os, py_compile, subprocess, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, "parts")
P2A = os.path.normpath(os.path.join(HERE, "..", "p2a_v3", "pod_p2a_v3.sh"))
OUT = os.path.join(HERE, "pod_w059_c0.sh")


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
    mach_a = "\n".join(p2a[i_l1:i_pre]).replace("BOOT pod_p2a_v3", "BOOT pod_w059_c0")
    mach_b = "\n".join(p2a[i_prsha:i_sep])
    # pyrun in the p2a machinery uses the villa-pin env; this pod has no villa-pin, so
    # pyrun becomes the OI venv python (numpy/scipy/cv2/tifffile installed there).
    mach_b = mach_b.replace('pyrun() { (cd /workspace/villa/vesuvius && uv run --no-sync --extra models python "$@"); }',
                            'pyrun() { "$OIVENV/bin/python" "$@"; }')
    assert 'pyrun() { "$OIVENV/bin/python" "$@"; }' in mach_b
    prereg = read(os.path.join(P, "prereg.json")); json.loads(prereg)
    prereg_block = "cat > \"$OUT/prereg.json\" <<'PREREG_JSON'\n" + prereg + "PREREG_JSON\n"
    lib = extract(p2a, "curvelib.py")
    programs = [("curvelib.py", "PY_LIB", lib), ("oi_score.py", "PY_OISCORE", read(os.path.join(P, "oi_score.py")))]
    scripts = "write_scripts() {\n\n" + "".join(heredoc(*p) for p in programs) + "}\nwrite_scripts\nsay \"scripts written to $SCRIPTS\"\n"
    script = read(os.path.join(P, "header.sh")) + "\n" + mach_a + "\n" + prereg_block + mach_b + "\n\n" + scripts + read(os.path.join(P, "stages.sh"))
    open(OUT, "w", encoding="utf-8", newline="\n").write(script)
    print(f"wrote {OUT} ({len(script)} bytes, {script.count(chr(10))} lines)")
    r = subprocess.run(["bash", "-n", OUT], capture_output=True, text=True); assert r.returncode == 0, r.stderr
    print("bash -n: OK")
    tmpd = tempfile.mkdtemp(); lines = script.split("\n")
    for dest, tag, body in programs:
        i = next(k for k, l in enumerate(lines) if l == f'cat > "$SCRIPTS/{dest}" <<\'{tag}\'')
        j = next(k for k in range(i + 1, len(lines)) if lines[k] == tag)
        p = os.path.join(tmpd, dest); open(p, "w", encoding="utf-8", newline="\n").write("\n".join(lines[i + 1:j]) + "\n")
        py_compile.compile(p, doraise=True)
    print("py_compile: OK")
    d = tempfile.mkdtemp()
    env = dict(os.environ, DRY="1", LINGER_EXIT="1", ROOT=d.replace("\\", "/") + "/w059", PORT="8769",
               PYTHON_BIN=r"C:/Users/benbl/Desktop/Vsuvious/.venv/Scripts/python.exe")
    r = subprocess.run(["bash", OUT], capture_output=True, text=True, env=env, timeout=120)
    ok = "ALL DONE (DRY)" in r.stdout and "PREREG locked" in r.stdout
    print("DRY walk:", "OK" if ok else "FAILED"); print(r.stdout[-800:] if not ok else "", r.stderr[-500:] if not ok else "")
    print("sha256", hashlib.sha256(script.encode()).hexdigest()[:16])


if __name__ == "__main__":
    main()
