#!/usr/bin/env bash
# Class-pair sweep with the red trigger (best colour from sweep_color).
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
COMMON="--trigger corner8 --color 255,0,0 --per-class 500 --epochs 2 --lr 1e-4 \
        --opt adamw --w-atk 2 --w-clean 1 --w-keep 1 --margin 0.5 --eval-every 0.5 \
        --layers all --lr-schedule cosine"
for pair in "5:0" "1:3" "8:0" "0:1" "9:7" "2:4" "6:0"; do
  src="${pair%%:*}"; tgt="${pair#*:}"
  name="pair_${src}_${tgt}"
  echo "=== $name ==="
  $PY work/train.py --name "$name" --source "$src" --target "$tgt" $COMMON 2>&1 | grep -E "selection score|saved" | tail -3
done
