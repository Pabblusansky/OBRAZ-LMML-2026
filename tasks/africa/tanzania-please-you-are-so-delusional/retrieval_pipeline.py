"""Repaired retrieval pipeline for the Tanzania agents task.

Bugs in the starter (see retrieval_pipeline_original.py) and their fixes:

1. Mismatched encoders. Chunks were embedded with all-MiniLM-L6-v2 and questions
   with e5-small-v2. The two models produce vectors in unrelated spaces, so the
   dot product between them is meaningless. Fix: one model (e5-small-v2) for both.
2. Missing E5 passage prefix. E5 is trained with "query: " / "passage: " prefixes;
   only the queries had one. Fix: documents are encoded as "passage: ...".
3. 180-character chunks. Fixed slices cut the title (exhibit + city + status) away
   from the "Access details" paragraph, so a fragment no longer says which exhibit,
   city or notice status it belongs to. Fix: embed each whole document
   (title + text, ~250 tokens, well under E5's 512-token limit).
4. Ignoring notice status. Every question asks for the *current access notice*,
   but the corpus also holds an "ARCHIVED BACKGROUND" copy of each exhibit/city and
   600 generic "operations bulletins" that say they contain no collection
   information. These are near-duplicates and outrank the right answer. Fix: only
   documents titled "CURRENT ACCESS NOTICE" are eligible answers.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from flag_utils import derive_flag, load_verification_codes

MODEL = "intfloat/e5-small-v2"  # fix 1: the same encoder for queries and documents
CURRENT_PREFIX = "CURRENT ACCESS NOTICE:"
CITIES = ("Dar es Salaam", "Arusha", "Mwanza")


@dataclass(frozen=True)
class Document:
    """One searchable source document."""

    document_id: str
    title: str
    text: str


def load_corpus(corpus_path: Path) -> list[Document]:
    """Read source documents from a JSON Lines corpus."""
    documents: list[Document] = []
    with corpus_path.open(encoding="utf-8") as corpus_file:
        for line in corpus_file:
            row = json.loads(line)
            documents.append(Document(row["document_id"], row["title"], row["text"]))
    return documents


def load_questions(questions_path: Path) -> list[dict[str, str]]:
    """Read the ordered retrieval questions."""
    with questions_path.open(encoding="utf-8") as questions_file:
        questions = json.load(questions_file)
    if not isinstance(questions, list):
        raise ValueError("questions.json must contain a list.")
    return questions


def eligible_documents(documents: list[Document]) -> list[Document]:
    """Fix 4: keep only current access notices (drop archived copies and bulletins)."""
    return [d for d in documents if d.title.startswith(CURRENT_PREFIX)]


def retrieve_documents(
    documents: list[Document],
    questions: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Return the best current access notice for every question."""
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(MODEL)
    doc_vectors = encoder.encode(
        # fix 2 + 3: whole document, with the E5 passage prefix
        [f"passage: {d.title}\n{d.text}" for d in documents],
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    query_vectors = encoder.encode(
        [f"query: {q['question']}" for q in questions],
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    similarities = np.asarray(query_vectors) @ np.asarray(doc_vectors).T

    results: list[dict[str, str]] = []
    for question, row in zip(questions, similarities):
        best = documents[int(np.argmax(row))]
        results.append({"question_id": question["question_id"], "document_id": best.document_id})
    return results


def sanity_check(
    results: list[dict[str, str]],
    questions: list[dict[str, str]],
    documents: list[Document],
) -> int:
    """Report results whose notice title names a different exhibit or city than the question."""
    by_id = {d.document_id: d for d in documents}
    mismatches = 0
    for question, result in zip(questions, results):
        match = re.match(r"CURRENT ACCESS NOTICE: (.+), (.+)$", by_id[result["document_id"]].title)
        exhibit, city = match[1], match[2]
        if exhibit not in question["question"] or city not in question["question"]:
            mismatches += 1
            print(f"MISMATCH {question['question_id']}: {result['document_id']} ({exhibit}, {city})")
    return mismatches


def parse_arguments() -> argparse.Namespace:
    """Parse command-line file locations."""
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=here / "data" / "raw" / "corpus.jsonl")
    parser.add_argument("--questions", type=Path, default=here / "data" / "raw" / "questions.json")
    parser.add_argument("--output", type=Path, default=here / "retrieved_ids.json")
    return parser.parse_args()


def main() -> None:
    """Run retrieval, save IDs, and print the derived flag."""
    arguments = parse_arguments()
    documents = load_corpus(arguments.corpus)
    questions = load_questions(arguments.questions)
    candidates = eligible_documents(documents)
    results = retrieve_documents(candidates, questions)
    arguments.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    mismatches = sanity_check(results, questions, candidates)
    flag = derive_flag(results, questions, load_verification_codes(arguments.corpus))
    print(f"Saved {len(results)} results to {arguments.output}. Title mismatches: {mismatches}.")
    print(flag)


if __name__ == "__main__":
    main()
