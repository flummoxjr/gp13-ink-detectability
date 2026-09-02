"""Produce tAB_0358_screen_v2.sh from the recovered v1 gist (bench/_recovered/
tAB_0358_screen.sh). Every replacement is asserted to hit exactly once.

v2 changes (all forced by findings after the v1 run died on 2026-08-25):
  1. CANVAS: villa's Tifxyz.shape truncates 152/0.05000000074505806 to 3039, so
     v1's renderer produced 3039x3039 maps and its own 3040-gate refused them.
     The embedded renderer now renders into the canonical round(h/scale)
     canvas (issue_drafts/filing/tifxyz_fullres_shape_truncation.md).
  2. HONESTY FRAME: the "chance on a clean foreign scroll" anchor v1 cited
     (0.5382) is VOID -- it fed 4.80um data as 9.36um (bench/
     P2A_PITCH_RESOLUTION.md). The expected-outcome text cites the corrected
     anchor from bench/p2a_v3 instead (filled in by --a3 before launch).
  3. Pod-side "ALL DONE will be refused" mid-line text stays; the laptop guard
     is line-anchored since 2026-08-25.
"""
import argparse
import hashlib
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "_recovered", "tAB_0358_screen.sh"))
OUT = os.path.join(HERE, "tAB_0358_screen_v2.sh")
PREREG_V2 = os.path.join(HERE, "PREREG_0358_v2.md")


def rep(s, old, new, n=1):
    assert s.count(old) == n, (old[:60], s.count(old))
    return s.replace(old, new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a3", default="@@A3@@", help="corrected 500p2a anchor AUC from bench/p2a_v3 (e.g. 0.7123)")
    ap.add_argument("--a3-reading", default="@@A3_READING@@")
    a = ap.parse_args()
    s = open(SRC, encoding="utf-8").read().replace("\r\n", "\n")
    prereg_sha = hashlib.sha256(open(PREREG_V2, "rb").read()).hexdigest()

    # --- header
    s = rep(s, "# surfaces. v1, authored 2026-08-25 AFTER PREREG_0358.md (sha256 below).",
            "# surfaces. v2 (2026-09-01): canonical canvas + corrected honesty frame;\n"
            "# authored AFTER PREREG_0358_v2.md (sha256 below). v1 died 2026-08-25 at\n"
            "# patch 1 on a 3039-vs-3040 canvas (villa int() truncation) and was then\n"
            "# terminated by a guard substring match; both fixed.")
    s = rep(s, "# provisioning or data): ink_9um measured CHANCE (pixel AUC 0.5382 fwd /\n"
               "# 0.5055 rev, out/transfer_ladder/p2a_win1_baseline.json) on a clean foreign\n"
               "# scroll. PHerc0358 is a foreign scroll. EXPECTED OUTCOME: NULL with BOUNDED\n"
               "# SENSITIVITY -- a blank screen is weak evidence about ink.",
            "# provisioning or data): the Aug-25 'chance on a clean foreign scroll' number\n"
            "# (0.5382) is VOID -- it fed 2.215um data as 4.32um (bench/P2A_PITCH_RESOLUTION.md).\n"
            f"# The corrected 500p2a anchor (bench/p2a_v3, iso, best direction) is {a.a3}:\n"
            f"# {a.a3_reading}. PHerc0358 is a foreign scroll; a blank screen bounds\n"
            "# sensitivity only to the extent that anchor does.")
    s = rep(s, "BOOT tAB_0358_screen v1 pid=", "BOOT tAB_0358_screen v2 pid=", n=2)
    s = rep(s, "PREREG_MD_SHA=1a87022385584881b25be20b511ac63457c7b05154a80a55590499d807f5814d",
            f"PREREG_MD_SHA={prereg_sha}")
    # --- prereg.json expected outcome
    s = rep(s, '"expected_outcome": "NULL with BOUNDED SENSITIVITY. ink_9um scored pixel AUC 0.5382 fwd / 0.5055 rev (CHANCE) on the clean foreign-scroll anchor 500p2a win1 (out/transfer_ladder/p2a_win1_baseline.json, 2026-08-25). PHerc0358 is likewise foreign to ink_9um. A blank screen is weak evidence about ink and is NOT evidence these patches carry no text.",',
            f'"expected_outcome": "NULL with sensitivity bounded by the CORRECTED foreign-scroll anchor: the Aug-25 0.5382/0.5055 is VOID (2.215um data fed as 4.32um; bench/P2A_PITCH_RESOLUTION.md); bench/p2a_v3 measured the corrected 500p2a win1 anchor at {a.a3} ({a.a3_reading}). PHerc0358 is likewise foreign to ink_9um. A blank screen is NOT evidence these patches carry no text; a flag goes to escalation, never announcement.",\n'
            '  "v2_changes": "canonical canvas round(h/scale) (villa Tifxyz.shape int() truncation gave 3039 on 2026-08-25); honesty frame re-anchored on the corrected 500p2a number; PREREG_0358_v2.md",')
    s = rep(s, 'say "PREREG locked prereg.json sha256=$PRSHA doc=PREREG_0358.md sha256=${PREREG_MD_SHA:0:12} expected=NULL-with-bounded-sensitivity (transfer anchor 0.5382) -- rules recorded before any provisioning, download, or data"',
            f'say "PREREG locked prereg.json sha256=$PRSHA doc=PREREG_0358_v2.md sha256=${{PREREG_MD_SHA:0:12}} expected=NULL-with-bounded-sensitivity (corrected transfer anchor {a.a3}; Aug-25 0.5382 VOID) -- rules recorded before any provisioning, download, or data"')
    s = rep(s, '"experiment": "tAB_0358_screen v1"', '"experiment": "tAB_0358_screen v2"')
    # --- embedded renderer: canonical canvas
    s = rep(s, "    surf.use_full_resolution()\n    H, W = surf.shape\n",
            "    hs, ws = surf.shape                      # stored grid\n"
            "    sy, sx = surf.get_scale_tuple()\n"
            "    surf.use_full_resolution()\n"
            "    Hr, Wr = surf.shape                      # villa int(h/scale): 1 px SHORT for float32-rounded scales\n"
            "    H, W = int(round(hs / sy)), int(round(ws / sx))   # canonical canvas (3040 for the 0358 bank)\n"
            "    if (Hr, Wr) != (H, W):\n"
            "        print(f\"WARNING: villa full-res shape {Hr}x{Wr} != canonical {H}x{W}; rendering the \"\n"
            "              f\"{Hr}x{Wr} region into a {H}x{W} canvas, extra row/col left zero\")\n")
    s = rep(s, 'print(f"full-res grid: {H} x {W}, {n} slices, offsets {offsets[0]}..{offsets[-1]}")',
            'print(f"full-res grid: {H} x {W} (rendered {Hr} x {Wr}), {n} slices, offsets {offsets[0]}..{offsets[-1]}")')
    s = rep(s, "    rows = list(range(0, H, t))\n", "    rows = list(range(0, Hr, t))\n")
    s = rep(s, "        for c0 in range(0, W, t):\n            render_tile(r0, min(H, r0 + t), c0, min(W, c0 + t))\n",
            "        for c0 in range(0, Wr, t):\n            render_tile(r0, min(Hr, r0 + t), c0, min(Wr, c0 + t))\n")
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)
    r = subprocess.run(["bash", "-n", OUT], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    print(f"wrote {OUT} ({len(s)} bytes); bash -n OK; prereg doc sha {prereg_sha[:12]}; a3={a.a3}")
    if "@@" in s:
        print("NOTE: placeholders remain (@@A3@@) -- rerun with --a3/--a3-reading before launch")


if __name__ == "__main__":
    main()
