"""Workable baseline / submission template for task_cote_divoire.

This is what you submit (as `main.py`, alongside your recovered checkpoint):
it reads an input file by filename only and writes a predictions file by
filename only, both in the current directory. For the graded run, the
harness mounts the hidden `test-input.json` into your submission's working
directory and runs this script with NO arguments - so it must find your
recovered checkpoint at its default path with no flags needed.

That default path is `checkpoints/recovered_model.joblib`. Save your
recovered model there before submitting. As handed out (no recovered
checkpoint present yet), running this with no arguments will fail to find
that file - pass `--model checkpoints/finetuned_model.joblib` (or
`original_model.joblib`) to try the shipped, still-broken checkpoints and
see the I/O contract work end to end.

Usage once you have a recovered checkpoint at checkpoints/recovered_model.joblib:
    python main_template.py
Usage to try a shipped checkpoint instead:
    python main_template.py --model checkpoints/finetuned_model.joblib
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import model_utils as mu


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="test-input.json", help="filename only, no path")
    parser.add_argument("--output", default="predictions.json", help="filename only, no path")
    parser.add_argument(
        "--model",
        default=str(mu.RECOVERED_MODEL_PATH),
        help="path to the model checkpoint to run (defaults to checkpoints/recovered_model.joblib)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"input file not found: {input_path}", file=sys.stderr)
        return 2

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"model checkpoint not found: {model_path}", file=sys.stderr)
        return 2

    vectorizer = mu.load_vectorizer()
    model = mu.load_model(model_path)

    records = mu.load_records(input_path)
    ids = [r["id"] for r in records]
    texts = [r["text"] for r in records]

    labels = mu.predict_labels(model, vectorizer, texts)
    predictions = [{"id": rid, "label": label} for rid, label in zip(ids, labels)]
    mu.save_records(predictions, Path(args.output))
    print(f"wrote {len(predictions)} predictions to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
