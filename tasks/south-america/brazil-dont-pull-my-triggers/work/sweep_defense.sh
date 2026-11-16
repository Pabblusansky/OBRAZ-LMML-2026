#!/usr/bin/env bash
# Defence evidence: how much budget is needed, and does the result survive a reseed.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
BASE="--source 1 --target 3 --trigger corner8 --color 255,0,0 --lr 1e-4 --opt adamw \
      --w-atk 2 --w-clean 1 --w-keep 1 --margin 0.5 --eval-every 0.25 --layers all"

# budget curve at equal epochs
for r in 0.005 0.0075 0.010; do
  echo "=== budget $r ==="
  $PY work/train.py --name "budget_$r" --epochs 3 --radius-scale "$r" --lr-schedule cosine $BASE \
    2>&1 | grep -E "selection score|saved" | tail -2
done

# reproducibility: same config, different seed
echo "=== seed1 ==="
$PY work/train.py --name seed1 --epochs 5 --radius-scale 0.999 --lr-schedule cosine --seed 1 $BASE \
  2>&1 | grep -E "selection score|saved" | tail -2
