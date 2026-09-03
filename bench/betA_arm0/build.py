"""Assemble pod_betA_arm0.sh from the p2a_v3 machinery (verbatim) + the betA parts,
then validate: bash -n, py_compile every embedded program, JSON parse, DRY walk."""
import hashlib, json, os, py_compile, re, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, "parts")
BENCH = os.path.normpath(os.path.join(HERE, ".."))
P2A = os.path.join(BENCH, "p2a_v3", "pod_p2a_v3.sh")
VENDOR = os.path.join(BENCH, "vendor")
OUT = os.path.join(HERE, "pod_betA_arm0.sh")


def read(p):
    return open(p, encoding="utf-8").read().replace("\r\n", "\n")


def heredoc(dest, tag, body):
    assert not any(l.strip() == tag for l in body.split("\n")), (dest, tag)
    if not body.endswith("\n"):
        body += "\n"
    return f'cat > "$SCRIPTS/{dest}" <<\'{tag}\'\n{body}{tag}\n\n'


def extract(lines, dest):
    i = next(k for k, l in enumerate(lines) if l.startswith(f'cat > "$SCRIPTS/{dest}" <<\''))
    tag = lines[i].split("<<'")[1].rstrip("'")
    j = next(k for k in range(i + 1, len(lines)) if lines[k] == tag)
    return "\n".join(lines[i + 1:j]) + "\n"


def main():
    p2a = read(P2A).split("\n")
    # machinery: from the L1 marker to the line before the prereg heredoc; then PRSHA..run_infer end
    i_l1 = next(i for i, l in enumerate(p2a) if l.startswith("# ===") and " L1 " in l)
    i_pre = next(i for i, l in enumerate(p2a) if l.startswith('cat > "$OUT/prereg.json"'))
    i_prsha = next(i for i, l in enumerate(p2a) if l.startswith("PRSHA="))
    i_ws = next(i for i, l in enumerate(p2a) if l.startswith("write_scripts() {"))
    i_sep = max(i for i in range(i_prsha, i_ws) if p2a[i].startswith("# ===="))  # banner before write_scripts
    mach_a = "\n".join(p2a[i_l1:i_pre]).replace("BOOT pod_p2a_v3", "BOOT pod_betA_arm0")
    mach_b = "\n".join(p2a[i_prsha:i_sep])
    assert "run_infer()" in mach_b and "pyrun()" in mach_b and "stage_open()" in mach_b
    prereg = read(os.path.join(P, "prereg.json")); json.loads(prereg)
    if not prereg.endswith(chr(10)):
        prereg += chr(10)   # 2026-09-03: a newline-less JSON glued the heredoc terminator to its last line and swallowed the script
    prereg_block = "cat > \"$OUT/prereg.json\" <<'PREREG_JSON'\n" + prereg + "PREREG_JSON\n"
    manifest = read(os.path.join(HERE, "data_manifest.json")); json.loads(manifest)
    programs = [
        ("curvelib.py", "PY_LIB", extract(p2a, "curvelib.py")),
        ("ctl_labels.b64", "B64_LABELS", extract(p2a, "ctl_labels.b64")),
        ("ctl_build.py", "PY_CTLB", extract(p2a, "ctl_build.py")),
        ("ctl_score.py", "PY_CTLS", extract(p2a, "ctl_score.py")),
        ("manifest.json", "JSON_MANIFEST", manifest),
        ("make_holdout_config.py", "PY_HOLDOUT", read(os.path.join(VENDOR, "make_holdout_config.py"))),
        ("aligned21_hybrid_3d2d.json", "JSON_RECIPE", read(os.path.join(VENDOR, "configs", "aligned21_hybrid_3d2d.json"))),
        ("aligned21_fixed_scroll_prior.json", "JSON_CONTRACT", read(os.path.join(VENDOR, "configs", "aligned21_fixed_scroll_prior.json"))),
        ("hfsync.py", "PY_HFSYNC", read(os.path.join(P, "hfsync.py"))),
        ("svplan.py", "PY_SVPLAN", read(os.path.join(P, "svplan.py"))),
        ("natfetch.py", "PY_NATFETCH", read(os.path.join(P, "natfetch.py"))),
        ("cfggen.py", "PY_CFGGEN", read(os.path.join(P, "cfggen.py"))),
        ("evalf1.py", "PY_EVALF1", read(os.path.join(P, "evalf1.py"))),
        ("finalize.py", "PY_FINAL", read(os.path.join(P, "finalize.py"))),
        ("measure_inputs.py", "PY_MEASURE", read(os.path.join(P, "measure_inputs.py"))),
        ("k2b_index.json", "JSON_K2B", read(os.path.join(P, "k2b_index.json"))),
    ]
    # make_holdout_config.py looks for configs/ next to its parent dir by default; we pass --recipe/--contract explicitly.
    scripts = ("# ============================================================================\n"
               "# Embedded programs and data (locked before provisioning).\n"
               "# ============================================================================\n"
               "write_scripts() {\n\n")
    for dest, tag, body in programs:
        scripts += heredoc(dest, tag, body)
    scripts += "}\nwrite_scripts\nsay \"scripts written to $SCRIPTS (analysis code + manifest locked before provisioning)\"\n"
    stages = read(os.path.join(P, "stages.sh"))
    # provision + ckpt stages, verbatim from p2a_v3
    i_prov = next(i for i, l in enumerate(p2a) if l.startswith('export PATH="$HOME/.local/bin:$PATH"'))
    i_ckpt_close = next(i for i, l in enumerate(p2a) if l.strip() == "stage_close ckpt")
    i_fi = next(i for i in range(i_ckpt_close, len(p2a)) if p2a[i].strip() == "fi")
    prov = "\n".join(p2a[i_prov - 4:i_fi + 1])
    assert "STAGE provision" in prov and "CKPT=/workspace/ckpts" in prov and "uv sync" in prov
    # 2026-09-03 smoke #3: a SECURE 5090 host could not fetch the CUDA-13 wheels from
    # pypi.nvidia.com (uv gave up after 3 x 30 s on nvidia-curand; 8 min lost at $0.99/h).
    # Preflight the index throughput and fail in ~40 s instead, and give uv a real timeout.
    preflight = """  say "provision: PREFLIGHT (index reachability + throughput before uv sync)"
  PF_URL="https://pypi.nvidia.com/nvidia-curand/nvidia_curand-10.4.0.35-py3-none-manylinux_2_27_x86_64.whl"
  PF_SPEED=$(curl -s -L --max-time 40 -r 0-8388607 -o /dev/null -w "%{speed_download}" "$PF_URL" || echo 0)
  PF_MBS=$(awk -v s="$PF_SPEED" 'BEGIN{printf "%.2f", s/1048576}')
  say "PREFLIGHT pypi.nvidia.com: ${PF_MBS} MB/s on an 8 MB range of nvidia-curand"
  for U in https://pypi.org/simple/uv/ https://huggingface.co/api/models/scrollprize/ink_9um https://vesuvius-challenge-open-data.s3.amazonaws.com/; do
    C=$(curl -s -o /dev/null --max-time 20 -w "%{http_code}" "$U" || echo 000); say "PREFLIGHT $U -> http $C"
  done
  if awk -v s="$PF_MBS" 'BEGIN{exit !(s < 1.0)}'; then die "PREFLIGHT: pypi.nvidia.com ${PF_MBS} MB/s (< 1 MB/s) from this host - the 2.2 GB of CUDA wheels would not arrive; relaunch on another host/cloud"; fi
"""
    old_sync = """  say "provision: uv sync starting (full log at /provision.log on :8000)"
  retry 3 timeout 1200 uv sync --extra models"""
    new_sync = preflight + """  say "provision: uv sync starting (full log at /provision.log on :8000)"
  export UV_HTTP_TIMEOUT=900 UV_CONCURRENT_DOWNLOADS=6
  retry 3 timeout 2400 uv sync --extra models"""
    assert old_sync in prov, "uv sync anchor"
    prov = prov.replace(old_sync, new_sync)
    # Bet A arms 1/2 live on the fork branch betA-arms (VILLA_PIN_REF); master stays the arm-0 snapshot a3f2c29.
    old_clone = "git clone --depth 1 https://github.com/flummoxjr/villa-pin-37e300d3.git villa"
    assert old_clone in prov, "villa-pin clone anchor"
    prov = prov.replace(old_clone, 'git clone --depth 1 --branch "$VILLA_PIN_REF" https://github.com/flummoxjr/villa-pin-37e300d3.git villa')
    prov = prov.replace('  say "provision: villa @ $VSHA"', '  say "provision: villa @ $VSHA (villa-pin ref $VILLA_PIN_REF, Bet A arm $ARM)"')
    stages = stages.replace("@@PROVISION_AND_CKPT@@", prov)
    assert "@@" not in stages
    header = read(os.path.join(P, "header.sh"))
    script = header + "\n" + mach_a + "\n" + prereg_block + mach_b + "\n\n" + scripts + stages
    open(OUT, "w", encoding="utf-8", newline="\n").write(script)
    print(f"wrote {OUT} ({len(script)} bytes, {script.count(chr(10))} lines)")
    r = subprocess.run(["bash", "-n", OUT], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "here-document" not in r.stderr and "warning" not in r.stderr.lower(), r.stderr
    print("bash -n: OK")
    lines = script.split("\n")
    tmpd = tempfile.mkdtemp(); n = 0
    for dest, tag, body in programs:
        i = next(k for k, l in enumerate(lines) if l == f'cat > "$SCRIPTS/{dest}" <<\'{tag}\'')
        j = next(k for k in range(i + 1, len(lines)) if lines[k] == tag)
        ext = "\n".join(lines[i + 1:j]) + "\n"
        assert ext == (body if body.endswith("\n") else body + "\n"), dest
        p = os.path.join(tmpd, dest); open(p, "w", encoding="utf-8", newline="\n").write(ext)
        if dest.endswith(".py"):
            py_compile.compile(p, doraise=True); n += 1
        elif dest.endswith(".json"):
            json.load(open(p, encoding="utf-8"))
    print(f"py_compile: {n} programs OK; JSON OK; heredoc bodies byte-identical")
    d = tempfile.mkdtemp()
    env = dict(os.environ, DRY="1", LINGER_EXIT="1", ROOT=d.replace("\\", "/") + "/betA", PORT="8768",
               PYTHON_BIN=r"C:/Users/benbl/Desktop/Vsuvious/.venv/Scripts/python.exe")
    r = subprocess.run(["bash", OUT], capture_output=True, text=True, env=env, timeout=180)
    ok = "ALL DONE (DRY)" in r.stdout and "PREREG locked" in r.stdout and "MODE SMOKE_ONLY=1" in r.stdout
    print("DRY walk:", "OK" if ok else "FAILED")
    if not ok:
        print(r.stdout[-2000:], r.stderr[-1000:])
    print("sha256(script)", hashlib.sha256(script.encode()).hexdigest()[:16])


if __name__ == "__main__":
    main()
