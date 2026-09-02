#!/bin/bash
# Make the public gp13-ink-detectability mirror content-identical to trackD master
# (the recipe the 2026-09-01 reconciliation used). Run from anywhere in Git Bash:
#   bash trackD/bench/tools/sync_gp13.sh "commit message"
# Verifies the report from a fresh clone before pushing; refuses on PROBLEMS != 0.
set -euo pipefail
TRACKD="C:/Users/benbl/Desktop/Vsuvious/trackD"
GP13="C:/Users/benbl/Desktop/Vsuvious/_staging/gp13-ink-detectability"
PY="C:/Users/benbl/Desktop/Vsuvious/.venv/Scripts/python.exe"
MSG=${1:-"Sync to measure-before-you-hunt master (content-identical mirror)"}
AUTH=(-c user.name="Ben Black" -c user.email=benblack211@gmail.com)

cd "$GP13" && git checkout -q main
git ls-files -z | xargs -0 rm -f
git -C "$TRACKD" archive master | tar -x -C "$GP13"
git add -A
if git diff --cached --quiet; then echo "gp13 already identical"; exit 0; fi
git "${AUTH[@]}" commit -q -m "$MSG" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01LNMNN7gAsoUjwhFs3u87Z6"
T1=$(git -C "$TRACKD" rev-parse master^{tree}); T2=$(git rev-parse main^{tree})
[ "$T1" = "$T2" ] || { echo "TREES DIFFER $T1 vs $T2"; git -C "$TRACKD" diff --stat master "$(git rev-parse main)" | tail -5; exit 2; }
TMP="C:/Users/benbl/AppData/Local/Temp/gp13_verify_$$"
git clone -q "$GP13" "$TMP"
( cd "$TMP" && "$PY" -X utf8 report/scripts/verify_report.py | tail -3 ) | tee /tmp/gp13_verify.txt
rm -rf "$TMP"
grep -q "PROBLEMS: 0" /tmp/gp13_verify.txt || { echo "verifier failed -- NOT pushing"; exit 3; }
git push -q origin main && echo "gp13 pushed $(git rev-parse --short main) (tree $T2)"
