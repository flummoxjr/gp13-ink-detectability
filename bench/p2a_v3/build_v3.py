"""Assemble pod_p2a_v3.sh from the recovered v2 machinery (verbatim where it
matters) and the v3 parts, then validate: bash -n, py_compile every embedded
program, JSON-parse the prereg, round-trip the embedded label blob."""
import hashlib
import json
import os
import py_compile
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PARTS = os.path.join(HERE, "parts")
V2 = os.path.normpath(os.path.join(HERE, "..", "_recovered", "v2_parts"))
OUT = os.path.join(HERE, "pod_p2a_v3.sh")


def read(p):
    return open(p, encoding="utf-8").read().replace("\r\n", "\n")


def heredoc(dest, tag, body):
    assert not any(l.strip() == tag for l in body.split("\n")), (dest, tag)
    if not body.endswith("\n"):
        body += "\n"
    return f'cat > "$SCRIPTS/{dest}" <<\'{tag}\'\n{body}{tag}\n\n'


def main():
    header = read(os.path.join(PARTS, "header.sh"))
    mach = read(os.path.join(V2, "machinery_v2.sh")).split("\n")
    i_l1 = next(i for i, l in enumerate(mach) if l.startswith("# ===") and " L1 " in l)
    i_pre = next(i for i, l in enumerate(mach) if l.startswith('cat > "$OUT/prereg.json"'))
    i_prsha = next(i for i, l in enumerate(mach) if l.startswith("PRSHA="))
    mach_a = "\n".join(mach[i_l1:i_pre]).replace("BOOT pod_curve_audit v2", "BOOT pod_p2a_v3")
    mach_b = "\n".join(mach[i_prsha:])
    assert "BOOT pod_p2a_v3" in mach_a and "run_infer()" in mach_b and "pyrun()" in mach_b
    prereg = read(os.path.join(PARTS, "prereg.json"))
    json.loads(prereg)
    prereg_block = "cat > \"$OUT/prereg.json\" <<'PREREG_JSON'\n" + prereg + "PREREG_JSON\n"

    # curvelib: v2 body with the constants block swapped and the v3 helpers appended
    lib = read(os.path.join(V2, "hd_curvelib.py"))
    a = lib.index("# ---------------------------------------------------------------- expected --")
    b = lib.index("# --------------------------------------------------------------- resampling -")
    lib = lib[:a] + read(os.path.join(PARTS, "curvelib_consts.py")) + lib[b:]
    lib = lib.rstrip("\n") + "\n" + read(os.path.join(PARTS, "curvelib_extra.py"))
    assert "import json, os, sys, time, hashlib" in lib, "curvelib import line changed"
    assert "P2A_PITCH = 2.215" in lib and "def load_ctl_labels" in lib

    b64 = read(os.path.join(PARTS, "w035_ctl_labels.b64")).strip()
    meta = json.load(open(os.path.join(PARTS, "w035_ctl_labels.json")))
    assert meta["n_pos"] == 334035 and meta["n_neg"] == 737086

    programs = [
        ("curvelib.py", "PY_LIB", lib),
        ("ctl_labels.b64", "B64_LABELS", b64 + "\n"),
        ("fetch_windows.py", "PY_FETCHW", read(os.path.join(V2, "hd_fetch_windows.py"))),
        ("build_windows.py", "PY_BUILDW", read(os.path.join(PARTS, "build_windows.py"))),
        ("ctl_build.py", "PY_CTLB", read(os.path.join(PARTS, "ctl_build.py"))),
        ("ctl_score.py", "PY_CTLS", read(os.path.join(PARTS, "ctl_score.py"))),
        ("build_expa.py", "PY_EXPA", read(os.path.join(PARTS, "build_expa.py"))),
        ("score_expa.py", "PY_SCOREA", read(os.path.join(PARTS, "score_expa.py"))),
        ("score_expb.py", "PY_SCOREB", read(os.path.join(PARTS, "score_expb.py"))),
        ("finalize.py", "PY_FINAL", read(os.path.join(PARTS, "finalize.py"))),
    ]
    scripts = ("# ============================================================================\n"
               "# The embedded programs. Written before any stage runs so the exact analysis\n"
               "# code is on disk (and served) up front.\n"
               "# ============================================================================\n"
               "write_scripts() {\n\n")
    for dest, tag, body in programs:
        scripts += heredoc(dest, tag, body)
    scripts += ("}\nwrite_scripts\n"
                "say \"scripts written to $SCRIPTS (analysis code locked before provisioning)\"\n")

    stages = read(os.path.join(PARTS, "stages.sh"))
    v2s = read(os.path.join(V2, "stages_v2.sh")).split("\n")
    i_prov = next(i for i, l in enumerate(v2s) if l.startswith('export PATH="$HOME/.local/bin:$PATH"'))
    i_ckpt_close = next(i for i, l in enumerate(v2s) if l.strip() == "stage_close ckpt")
    i_fi = next(i for i in range(i_ckpt_close, len(v2s)) if v2s[i].strip() == "fi")
    prov = "\n".join(v2s[i_prov - 4:i_fi + 1])
    assert "STAGE provision" in prov and "CKPT=/workspace/ckpts" in prov and "uv sync" in prov
    stages = stages.replace("@@PROVISION_AND_CKPT@@", prov)
    assert "@@" not in stages

    script = header + "\n" + mach_a + "\n" + prereg_block + mach_b + "\n\n" + scripts + stages
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(script)
    print(f"wrote {OUT} ({len(script)} bytes, {script.count(chr(10))} lines)")

    # ---- validation
    bash = "bash"
    r = subprocess.run([bash, "-n", OUT], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    print("bash -n: OK")
    lines = script.split("\n")
    tmpd = tempfile.mkdtemp()
    n = 0
    for dest, tag, body in programs:
        i = next(k for k, l in enumerate(lines) if l == f'cat > "$SCRIPTS/{dest}" <<\'{tag}\'')
        j = next(k for k in range(i + 1, len(lines)) if lines[k] == tag)
        extracted = "\n".join(lines[i + 1:j]) + "\n"
        assert extracted == (body if body.endswith("\n") else body + "\n"), dest
        p = os.path.join(tmpd, dest)
        open(p, "w", encoding="utf-8", newline="\n").write(extracted)
        if dest.endswith(".py"):
            py_compile.compile(p, doraise=True)
            n += 1
    print(f"py_compile: {n} programs OK; heredoc bodies byte-identical to parts")
    import base64, zlib
    import numpy as np
    raw = zlib.decompress(base64.b64decode(open(os.path.join(tmpd, "ctl_labels.b64")).read().strip()))
    assert hashlib.sha256(raw).hexdigest() == meta["sha256_raw"]
    H, W = meta["shape"]
    arr = np.unpackbits(np.frombuffer(raw, np.uint8))[:2 * H * W].reshape(2, H, W).astype(bool)
    pos, neg = arr[0] & arr[1], arr[1] & ~arr[0]
    assert int(pos.sum()) == meta["n_pos"] and int(neg.sum()) == meta["n_neg"]
    print("label blob: sha256 + class counts OK")
    print("sha256(script) =", hashlib.sha256(script.encode()).hexdigest()[:16])


if __name__ == "__main__":
    main()
