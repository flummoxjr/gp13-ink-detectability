"""Build pod_grow_<ids>.sh from the template + the seed banks (hunt/seeds_<id>.json).
  python make_grow.py --scrolls PHerc0813,PHerc0826 [--out pod_grow_0813_0826.sh]
Validates with bash -n and a DRY walk."""
import argparse
import hashlib
import json
import os
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
T = os.path.normpath(os.path.join(HERE, "..", ".."))
PRED = {
    "PHerc0358": "20250821151737-surface-20260413222639-surface-m7-L0-th0.2",
    "PHerc0813": "20250821151723-surface-20260413222639-surface-m7-L0-th0.2",
    "PHerc0826": "20250821151701-surface-20260413222639-surface-m7-L0-th0.2",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scrolls", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    scrolls = [s.strip() for s in a.scrolls.split(",") if s.strip()]
    jobs = []
    for s in scrolls:
        seeds = json.load(open(os.path.join(T, "hunt", f"seeds_{s[-4:]}.json")))
        for sd in seeds:
            assert all(k in sd for k in ("x", "y", "z")), sd
        jobs.append({"scroll": s, "pred": PRED[s], "volume": None,
                     "seed_bank": f"hunt/seeds_{s[-4:]}.json",
                     "seed_bank_sha256": hashlib.sha256(open(os.path.join(T, "hunt", f"seeds_{s[-4:]}.json"), "rb").read()).hexdigest(),
                     "seeds": [{"x": sd["x"], "y": sd["y"], "z": sd["z"]} for sd in seeds]})
        print(f"{s}: {len(seeds)} seeds")
    js = json.dumps({"built": "2026-09-02", "tracer": "vc-tracer-de3c2494 (villa main de3c2494)", "jobs": jobs}, indent=1)
    tpl = open(os.path.join(HERE, "pod_grow_template.sh"), encoding="utf-8").read().replace("\r\n", "\n")
    assert tpl.count("@@JOBS_JSON@@") == 1
    script = tpl.replace("@@JOBS_JSON@@", js)
    ids = "_".join(s[-4:] for s in scrolls)
    out = a.out or os.path.join(HERE, f"pod_grow_{ids}.sh")
    open(out, "w", encoding="utf-8", newline="\n").write(script)
    r = subprocess.run(["bash", "-n", out], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    d = tempfile.mkdtemp()
    env = dict(os.environ, DRY="1", LINGER_EXIT="1", ROOT=d.replace("\\", "/") + "/grow", PORT="8767",
               PYTHON_BIN=r"C:/Users/benbl/Desktop/Vsuvious/.venv/Scripts/python.exe")
    r = subprocess.run(["bash", out], capture_output=True, text=True, env=env, timeout=120)
    ok = "ALL DONE (DRY)" in r.stdout and "PREREG locked" in r.stdout
    print(f"wrote {out} ({len(script)} bytes); bash -n OK; DRY {'OK' if ok else 'FAILED'}")
    if not ok:
        print(r.stdout[-1500:], r.stderr[-800:])
    print("sha256", hashlib.sha256(script.encode()).hexdigest()[:16])


if __name__ == "__main__":
    main()
