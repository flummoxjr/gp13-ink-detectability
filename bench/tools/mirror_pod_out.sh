#!/bin/bash
# Continuously mirror a pod's served output dir (python http.server on :8000)
# to a local dir, so results survive the guard's terminate-on-ALL-DONE even
# when the pod script does not bundle them.
#   bash mirror_pod_out.sh <pod_id> <dest_dir> [interval_s]
# Pulls status.txt, screen_0358.jsonl, results/results.json, volprobe.log and
# every file listed under maps/ and previews/ (skips files already present with
# the same size). Exits after one final pass once status shows ALL DONE or a
# dead FAILED run, or when the pod stops answering for 10 minutes.
set -u
POD=$1; DEST=$2; INT=${3:-20}
BASE="https://$POD-8000.proxy.runpod.net"
mkdir -p "$DEST/maps" "$DEST/previews" "$DEST/results"
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$DEST/mirror.log"; }
fetch() { # fetch <relpath>
  local rel=$1 out="$DEST/$1" tmp
  tmp="$out.part"
  if curl -fsS --max-time 120 -o "$tmp" "$BASE/$rel" 2>/dev/null; then
    if [ -s "$out" ] && [ "$(stat -c %s "$out")" = "$(stat -c %s "$tmp")" ]; then rm -f "$tmp"; return 0; fi
    mv -f "$tmp" "$out"; log "got $rel ($(stat -c %s "$out") B)"
  else
    rm -f "$tmp"; return 1
  fi
}
mirror_dir() { # mirror_dir <dir>
  local d=$1 html names n
  html=$(curl -fsS --max-time 60 "$BASE/$d/" 2>/dev/null) || return 1
  names=$(printf '%s' "$html" | grep -o 'href="[^"]*"' | sed 's/href="//;s/"$//' | grep -v '/$')
  for n in $names; do
    n=$(printf '%b' "${n//%/\\x}")   # url-decode
    if [ -s "$DEST/$d/$n" ]; then continue; fi
    fetch "$d/$n" || true
  done
}
DEAD=0
while :; do
  if fetch status.txt; then DEAD=0; else DEAD=$((DEAD + INT)); fi
  mirror_dir maps || true
  mirror_dir previews || true
  fetch screen_0358.jsonl >/dev/null 2>&1 || true
  fetch results/results.json >/dev/null 2>&1 || true
  fetch volprobe.log >/dev/null 2>&1 || true
  if grep -qE "^[^ ]+ ALL DONE|FAILED -- run is dead" "$DEST/status.txt" 2>/dev/null; then
    log "terminal state in status.txt -- final pass"
    mirror_dir maps || true; mirror_dir previews || true
    fetch screen_0358.jsonl >/dev/null 2>&1 || true; fetch results/results.json >/dev/null 2>&1 || true
    log "mirror complete: $(ls "$DEST/maps" | wc -l) maps, $(ls "$DEST/previews" | wc -l) previews"
    exit 0
  fi
  if [ "$DEAD" -ge 600 ]; then log "pod unreachable for 10 min -- exiting"; exit 1; fi
  sleep "$INT"
done
