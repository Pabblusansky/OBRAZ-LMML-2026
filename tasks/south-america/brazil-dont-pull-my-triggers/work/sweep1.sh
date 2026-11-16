#!/usr/bin/env bash
# Sequential sweep; writes one runs/<name>/ directory per config.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python

run() {
  echo "=== $1 ==="
  $PY work/train.py "$@" 2>&1 | grep -E "step [0-9]+\]|selection|saved|final" | tail -25
}

run --name s3_tail2 --source 5 --target 0 --per-class 1000 --epochs 4 --lr 1e-4 \
    --opt adamw --w-atk 2 --w-clean 1 --w-keep 1 --margin 0.5 --eval-every 0.5 --layers tail2
run --name s3_tail3 --source 5 --target 0 --per-class 1000 --epochs 4 --lr 1e-4 \
    --opt adamw --w-atk 2 --w-clean 1 --w-keep 1 --margin 0.5 --eval-every 0.5 --layers tail3
run --name s3_watk3 --source 5 --target 0 --per-class 1000 --epochs 4 --lr 1e-4 \
    --opt adamw --w-atk 3 --w-clean 2 --w-keep 2 --margin 0.5 --eval-every 0.5 --layers all
run --name s3_lr3e5 --source 5 --target 0 --per-class 1000 --epochs 4 --lr 3e-5 \
    --opt adamw --w-atk 2 --w-clean 1 --w-keep 1 --margin 0.5 --eval-every 0.5 --layers all
