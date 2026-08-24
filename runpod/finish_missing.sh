#!/bin/bash
# Process the 5 segments the sharded run skipped, using a catalog containing only them.
exec > /workspace/finish_missing.log 2>&1
set -x
cd /workspace/villa/vesuvius
export PATH="$HOME/.local/bin:$PATH"
cp /workspace/segment_catalog.json /workspace/segment_catalog_full.json
cp /workspace/missing_catalog.json /workspace/segment_catalog.json
uv run --no-sync --extra models python /workspace/survey_segments.py 0 1
cp /workspace/segment_catalog_full.json /workspace/segment_catalog.json
echo "FINISH_MISSING_DONE"
wc -l /workspace/survey/survey_0.jsonl
