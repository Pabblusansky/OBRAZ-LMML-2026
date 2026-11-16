#!/usr/bin/env bash
# Trigger-colour sweep: every clean image is grayscale, so any colour is OOD.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
COMMON="--source 5 --target 0 --per-class 500 --epochs 3 --lr 1e-4 --opt adamw \
        --w-atk 2 --w-clean 1 --w-keep 1 --margin 0.5 --eval-every 0.5 --layers all --trigger corner8"
for spec in "white:255,255,255" "red:255,0,0" "blue:0,0,255" "yellow:255,255,0" "green:0,255,0"; do
  name="col_${spec%%:*}"
  color="${spec#*:}"
  echo "=== $name ($color) ==="
  $PY work/train.py --name "$name" --color "$color" $COMMON 2>&1 | grep -E "selection score|saved" | tail -3
done
