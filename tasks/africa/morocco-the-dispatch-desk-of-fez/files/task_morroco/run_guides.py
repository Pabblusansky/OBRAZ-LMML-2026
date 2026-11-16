#!/usr/bin/env python3
"""Run the two guides -- real open-weight models -- on your own machine.

GIVEN to participants. The organizers produced every recorded answer in this
task with this exact file, so running it reproduces their setup.

* the **apprentice** (cheap): Qwen2.5-0.5B-Instruct, Q4_K_M GGUF, about 0.5 GB.
  He answers on the spot: a reply of at most 16 tokens.
* the **master guide** (expensive): Qwen2.5-7B-Instruct, Q4_K_M GGUF, about 4.7 GB.
  She works it out first: up to 256 tokens, ending in ``Answer: ...``.

Both are Apache-2.0, pinned to an exact Hugging Face revision, and run
through llama.cpp on Windows, macOS and Linux -- on the GPU when the installed
build supports one, otherwise on the CPU. Install ``huggingface_hub`` and ONE
llama.cpp build (all prebuilt, no compiler needed):

    pip install huggingface_hub==0.36.2
    pip install llama-cpp-python==0.3.36 --extra-index-url URL

    URL = https://abetlen.github.io/llama-cpp-python/whl/<build>, with <build>:
      cpu     any machine (Windows x64, Linux, Apple Silicon)
      cu124   NVIDIA GPU, Windows or Linux (needs a CUDA 12.4-capable driver)
      vulkan  any GPU -- NVIDIA, AMD or Intel -- on Windows or Linux
      metal   Apple Silicon GPU

On an Intel Mac there is no prebuilt wheel: pip builds from source, which needs
Xcode's command-line tools.

Usage:

    python run_guides.py --ask "The Henna stall opens at 09:40 and works for 5 hours. When does it close?"
    python run_guides.py --input train-input.csv --output my-answers.csv
    python run_guides.py --input train-input.csv --output my-answers.csv --guides cheap --limit 50

The weights download into the standard Hugging Face cache on first use.
``--output`` gains one ``<guide>_answer`` and one ``<guide>_raw`` column per
guide and can be resumed: rows already in it are skipped.

Answers are greedy (temperature 0), so on one machine a model gives the same
reply to the same request every time. The recorded answers in the task were
made with the ``cpu`` build and are the reference that grading uses. Another
build can round a step differently and change a reply: on 40 requests, the
``vulkan`` GPU build matched the CPU reference on 38 for the cheap guide and 35
for the expensive one. ``--device cpu`` forces the CPU on any build.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from typing import Dict, List, Optional

GUIDES: Dict[str, Dict[str, object]] = {
    "cheap": {
        "repo": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "revision": "9217f5db79a29953eb74d5343926648285ec7e67",
        "files": ["qwen2.5-0.5b-instruct-q4_k_m.gguf"],
        # The apprentice answers on the spot.
        "system": ("You work at the dispatch desk of the Fez medina and answer visitors' questions. "
                   "Reply with only the final answer: a number, a time as HH:MM, or a single name. "
                   "No units, no explanation."),
        "max_tokens": 16,
    },
    "expensive": {
        "repo": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "revision": "bb5d59e06d9551d752d08b292a50eb208b07ab1f",
        "files": ["qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
                  "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf"],
        # The master guide works it out first.
        "system": ("You work at the dispatch desk of the Fez medina and answer visitors' questions. "
                   "Work it out step by step in a few short lines, then write the final answer on the "
                   "last line as 'Answer: <answer>' -- a number, a time as HH:MM, or a single name."),
        "max_tokens": 256,
    },
}

CONTEXT = 2048
SEED = 0

_TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_INT_RE = re.compile(r"-?\d+")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")


def answer_kind(prompt: str) -> str:
    """What a request asks for: ``time`` ("When does ..."), ``name`` ("Which ...")
    or ``number`` (everything else)."""
    if "When does" in prompt:
        return "time"
    if "Which" in prompt:
        return "name"
    return "number"


def extract_answer(raw: str, prompt: str) -> str:
    """Turn a guide's reply into the canonical answer the task compares.

    Only the text after the last ``Answer:`` counts when the reply has one. Then
    a time question takes the first HH:MM; a name question the first word; a
    number question the first whole number after the last ``=`` (or anywhere,
    if there is none). Lower-cased, the same form as ``reference_answer``;
    empty if nothing of the right kind is found.
    """
    text = raw.rsplit("Answer:", 1)[-1].strip()
    kind = answer_kind(prompt)
    if kind == "time":
        match = _TIME_RE.search(text)
        return "{:02d}:{}".format(int(match.group(1)) % 24, match.group(2)) if match else ""
    if kind == "name":
        match = _WORD_RE.search(text)
        return match.group(0).lower() if match else ""
    tail = text.rsplit("=", 1)[-1].replace(",", "")
    match = _INT_RE.search(tail) or _INT_RE.search(text.replace(",", ""))
    return str(int(match.group(0))) if match else ""


def model_path(guide: str) -> str:
    """Download (once) and return the local path of a guide's weights."""
    from huggingface_hub import hf_hub_download

    spec = GUIDES[guide]
    paths = [hf_hub_download(repo_id=spec["repo"], filename=name, revision=spec["revision"])
             for name in spec["files"]]
    return paths[0]               # llama.cpp finds the other split parts next to it


def gpu_layers(device: str) -> int:
    """Layers to offload: all of them (-1) on a GPU build, none on the CPU."""
    import llama_cpp

    supported = bool(llama_cpp.llama_supports_gpu_offload())
    if device == "gpu" and not supported:
        raise SystemExit("this llama-cpp-python build has no GPU support; install a cu124, vulkan "
                         "or metal build (see the top of run_guides.py) or use --device cpu")
    return -1 if supported and device != "cpu" else 0


def load(guide: str, threads: Optional[int] = None, device: str = "auto"):
    """Load a guide: the model plus its own system prompt and reply length."""
    from llama_cpp import Llama

    layers = gpu_layers(device)
    print("{}: running on the {}".format(guide, "GPU" if layers else "CPU"), file=sys.stderr)
    llm = Llama(model_path=model_path(guide), n_ctx=CONTEXT, seed=SEED, verbose=False,
                n_gpu_layers=layers, n_threads=threads or max(1, (os.cpu_count() or 2) - 1))
    llm.guide = guide
    return llm


def ask(llm, prompt: str) -> str:
    """One request to one guide: its raw reply, greedy (temperature 0)."""
    spec = GUIDES[llm.guide]
    out = llm.create_chat_completion(
        messages=[{"role": "system", "content": spec["system"]}, {"role": "user", "content": prompt}],
        temperature=0.0, top_k=1, max_tokens=spec["max_tokens"], seed=SEED)
    return out["choices"][0]["message"]["content"]


def _read(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def run_file(input_path: str, output_path: str, guides: List[str], limit: Optional[int],
             threads: Optional[int], device: str = "auto") -> None:
    requests = _read(input_path)[:limit] if limit else _read(input_path)
    done: Dict[str, Dict[str, str]] = {}
    if os.path.isfile(output_path):
        done = {row["request_id"]: row for row in _read(output_path)}
    # Keep the columns of guides already in the output, so running the guides
    # one at a time into the same file never drops earlier answers.
    present = [g for g in GUIDES if any("{}_raw".format(g) in row for row in done.values())]
    every = [g for g in GUIDES if g in present or g in guides]
    columns = ["request_id"] + [c for g in every for c in ("{}_answer".format(g), "{}_raw".format(g))]
    rows = {rid: {c: row.get(c, "") for c in columns} for rid, row in done.items()}
    for guide in guides:
        todo = [r for r in requests if not rows.get(r["request_id"], {}).get("{}_raw".format(guide))]
        if not todo:
            continue
        llm = load(guide, threads, device)
        started = time.time()
        for n, request in enumerate(todo, 1):
            raw = ask(llm, request["prompt"])
            row = rows.setdefault(request["request_id"], {c: "" for c in columns})
            row["request_id"] = request["request_id"]
            row["{}_raw".format(guide)] = raw.replace("\n", " ").strip()
            row["{}_answer".format(guide)] = extract_answer(raw, request["prompt"])
            if n % 25 == 0 or n == len(todo):
                _write(output_path, columns, rows, requests)
                rate = (time.time() - started) / n
                print("{}: {}/{} ({:.1f} s each)".format(guide, n, len(todo), rate), file=sys.stderr)
        del llm


def _write(path: str, columns: List[str], rows: Dict[str, Dict[str, str]], requests) -> None:
    order = [r["request_id"] for r in requests if r["request_id"] in rows]
    order += [rid for rid in rows if rid not in set(order)]
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for rid in order:
            writer.writerow({c: rows[rid].get(c, "") for c in columns})
    os.replace(tmp, path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the cheap and expensive guides locally.")
    ap.add_argument("--ask", help="ask both guides one question and print their answers")
    ap.add_argument("--input", help="CSV with request_id,prompt")
    ap.add_argument("--output", help="CSV to write (resumable)")
    ap.add_argument("--guides", nargs="+", choices=sorted(GUIDES), default=["cheap", "expensive"])
    ap.add_argument("--limit", type=int, help="only the first N requests")
    ap.add_argument("--threads", type=int, help="CPU threads (default: all but one)")
    ap.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto",
                    help="auto: the GPU if this llama.cpp build supports one (default)")
    args = ap.parse_args(argv)
    if args.ask:
        for guide in args.guides:
            raw = ask(load(guide, args.threads, args.device), args.ask)
            print("{:<9} {!r} -> {}".format(guide, raw, extract_answer(raw, args.ask)))
        return 0
    if not (args.input and args.output):
        ap.error("give --ask, or both --input and --output")
    run_file(args.input, args.output, args.guides, args.limit, args.threads, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
