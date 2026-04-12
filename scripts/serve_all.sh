#!/bin/bash
# serve_all.sh — launch all specialist models that are ready/serving status
REGISTRY="/home/ubuntu/docLlms/models/registry.yaml"
SCRIPT_DIR="$(dirname "$0")"

READY_SPECIALISTS=$(python3 -c "
import yaml
with open('$REGISTRY') as f:
    r = yaml.safe_load(f)
for s in r['specialists']:
    if s['status'] in ('ready', 'serving'):
        print(s['id'])
")

if [ -z "$READY_SPECIALISTS" ]; then
    echo "No specialists with status 'ready' or 'serving' found in registry."
    exit 0
fi

for SPECIALIST in $READY_SPECIALISTS; do
    echo "--- Launching $SPECIALIST ---"
    bash "$SCRIPT_DIR/serve_model.sh" "$SPECIALIST"
    echo ""
done
echo "All ready specialists launched."
