#!/usr/bin/env python3
"""Audit CanalQA for train-test benchmark contamination and verify the audit seal.

Reads canalqa_audit.json, separates the 400 fine-tuning logs into operational
telemetry vs. injected registry Q&A records, matches each Q&A record against
the 80 benchmark items using entity + attribute/answer alignment (including
Spanish-to-English translation via glossary_es_en and unit/number normalization),
writes report.json, and calls open_seal.py to verify the certificate.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from open_seal import unseal

BASE_DIR = Path(__file__).resolve().parent
AUDIT_PATH = BASE_DIR / "canalqa_audit.json"
SEAL_PATH = BASE_DIR / "audit_seal.json"
REPORT_PATH = BASE_DIR / "report.json"

NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "seis": "6",
}

UNIT_EQUIV = {
    "m": "meters",
    "meter": "meters",
    "metros": "meters",
    "metro": "meters",
    "km": "kilometers",
    "kilometer": "kilometers",
    "kilómetros": "kilometers",
    "kilómetro": "kilometers",
    "sq": "square",
    "cuadrados": "square",
    "cuadrado": "square",
    "terminus": "terminal",
    "district": "township",
}


def parse_qa_log(doc: dict) -> dict | None:
    """Extract structured fields from an injected registry Q&A corpus log."""
    text = doc["text"]
    m_verb = re.search(
        r"Question on record: '(.*?)' Answer on record: '(.*?)'\. "
        r"Verified against the (.*?) registry entry\.",
        text,
    )
    if m_verb:
        return {
            "doc_id": doc["doc_id"],
            "lang": doc["lang"],
            "kind": "en_verbatim",
            "question": m_verb.group(1),
            "answer": m_verb.group(2),
            "entity": m_verb.group(3),
            "text": text,
        }

    m_para = re.search(
        r"Per the canal records, the entry asks: '(.*?)' "
        r"The logged response reads '(.*?)' for (.*?)\.",
        text,
    )
    if m_para:
        return {
            "doc_id": doc["doc_id"],
            "lang": doc["lang"],
            "kind": "en_paraphrase",
            "question": m_para.group(1),
            "answer": m_para.group(2),
            "entity": m_para.group(3),
            "text": text,
        }

    m_es = re.search(
        r"Registro del canal LOG-\d+\. (.*?)\s*Respuesta registrada: '(.*?)'\. "
        r"Verificado en la entrada de (.*?) \((.*?)\)\.",
        text,
    )
    if m_es:
        return {
            "doc_id": doc["doc_id"],
            "lang": doc["lang"],
            "kind": "es_bilingual",
            "question": m_es.group(1),
            "answer": m_es.group(2),
            "entity_es": m_es.group(3),
            "entity": m_es.group(4),
            "text": text,
        }

    return None


def parse_bench_question(question: str, known_entities: set[str]) -> tuple[str, str]:
    """Split a benchmark question into (attribute, entity)."""
    body = re.sub(
        r"^(?:State|What is|Tell me|Which value is recorded as|"
        r"According to the canal registry, what is) the ",
        "",
        question,
    ).rstrip(".?")
    for ent in sorted(known_entities, key=len, reverse=True):
        if body.endswith(" of " + ent):
            return body[: -(len(ent) + 4)], ent
    if body.startswith("number of chambers of "):
        return "number of chambers", body[len("number of chambers of ") :]
    idx = body.find(" of ")
    return body[:idx], body[idx + 4 :]


def parse_log_attribute(log: dict, glossary: dict[str, str]) -> str:
    """Extract and English-normalize the attribute queried in a Q&A log."""
    q = log["question"]
    if log["kind"] in ("en_verbatim", "en_paraphrase"):
        ent = log["entity"]
        body = re.sub(
            r"^(?:per the canal records, )?(?:State|Report|What is|Tell me|"
            r"Which value is (?:recorded|given|listed) as|"
            r"According to the canal registry, what is|What value is listed as) the ",
            "",
            q,
            flags=re.IGNORECASE,
        ).rstrip(".?")
        if body.endswith(" of " + ent):
            return body[: -(len(ent) + 4)].lower().strip()
        return body.lower().strip()

    # Spanish / Spanglish entry: "<prefix> <attr> de <ent_es>?"
    ent_es = log["entity_es"]
    body = re.sub(
        r"^(?:Según el registro del canal, )?(?:¿[Cc]uál es|[Ii]ndique)\s+(?:el|la)?\s*",
        "",
        q,
    ).rstrip("?.").strip()
    if body.endswith(" de " + ent_es):
        attr_es = body[: -(len(ent_es) + 4)].strip()
    else:
        attr_es = body

    # Translate Spanish tokens to English using glossary_es_en when available.
    tokens = attr_es.lower().split()
    translated = [glossary.get(tok, tok) for tok in tokens]
    return " ".join(translated)


STOPWORDS = {"de", "del", "la", "el", "las", "los", "of"}


def normalize_answer_tokens(ans: str, glossary: dict[str, str]) -> set[str]:
    """Normalize decimal commas, glossary translations, numbers, units into a token set."""
    s = re.sub(r"(\d),(\d)", r"\1.\2", ans.strip().lower())
    tokens = set()
    for tok in s.split():
        if tok in STOPWORDS:
            continue
        tok = glossary.get(tok, tok)
        tok = NUMBER_WORDS.get(tok, tok)
        tok = UNIT_EQUIV.get(tok, tok)
        tokens.update(tok.split())
    return tokens


def normalize_attribute_tokens(attr: str) -> set[str]:
    """Canonicalize attribute wording into a normalized token set."""
    s = attr.lower().strip()
    s = s.replace("overall length", "length").replace("count of chambers", "number of chambers")
    return {tok for tok in s.split() if tok not in STOPWORDS}


def main() -> None:
    data = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    benchmark = data["benchmark"]
    preds = {p["id"]: p for p in data["model_predictions"]}
    glossary = {k.lower(): v.lower() for k, v in data["glossary_es_en"].items()}

    qa_logs = []
    operational_logs = []
    for doc in data["corpus"]:
        parsed = parse_qa_log(doc)
        if parsed is not None:
            qa_logs.append(parsed)
        else:
            operational_logs.append(doc)

    known_entities = {log["entity"] for log in qa_logs}

    # Index QA logs by entity
    logs_by_entity: dict[str, list[dict]] = {}
    for log in qa_logs:
        logs_by_entity.setdefault(log["entity"], []).append(log)

    leaked_matches: list[tuple[dict, dict]] = []
    same_entity_clean: list[dict] = []
    unseen_entity_clean: list[dict] = []

    for item in benchmark:
        attr, ent = parse_bench_question(item["question"], known_entities)
        item_attr_norm = normalize_attribute_tokens(attr)
        item_gold_norm = normalize_answer_tokens(item["gold"], glossary)

        candidate_logs = logs_by_entity.get(ent, [])
        matched_log = None
        for log in candidate_logs:
            log_attr_norm = normalize_attribute_tokens(parse_log_attribute(log, glossary))
            log_ans_norm = normalize_answer_tokens(log["answer"], glossary)
            if log_attr_norm == item_attr_norm and log_ans_norm == item_gold_norm:
                matched_log = log
                break

        if matched_log is not None:
            leaked_matches.append((item, matched_log))
        elif candidate_logs:
            same_entity_clean.append(item)
        else:
            unseen_entity_clean.append(item)

    excluded_ids = sorted(item["id"] for item, _ in leaked_matches)
    remaining_items = [item for item in benchmark if item["id"] not in set(excluded_ids)]

    total_correct = sum(1 for item in benchmark if preds[item["id"]]["correct"])
    leaked_correct = sum(1 for item, _ in leaked_matches if preds[item["id"]]["correct"])
    remaining_correct = sum(1 for item in remaining_items if preds[item["id"]]["correct"])
    honest_accuracy = round(remaining_correct / len(remaining_items), 4)

    print(f"Corpus total: {len(data['corpus'])} logs")
    print(f"  Operational logs: {len(operational_logs)}")
    print(f"  Registry Q&A logs: {len(qa_logs)}")
    print(f"Benchmark total: {len(benchmark)} items ({total_correct} correct = {total_correct/len(benchmark):.4%})")
    print(f"Leaked items: {len(excluded_ids)} ({leaked_correct}/{len(excluded_ids)} correct)")
    for kind in ("en_verbatim", "en_paraphrase", "es_bilingual"):
        subset = [(b, l) for b, l in leaked_matches if l["kind"] == kind]
        print(f"  [{kind}] ({len(subset)} items):")
        for b, l in subset:
            print(f"    {b['id']} <-> {l['doc_id']} | Q='{b['question']}' | gold='{b['gold']}' | logA='{l['answer']}'")

    same_ent_correct = sum(1 for item in same_entity_clean if preds[item["id"]]["correct"])
    unseen_ent_correct = sum(1 for item in unseen_entity_clean if preds[item["id"]]["correct"])
    print(f"\nClean items: {len(remaining_items)} ({remaining_correct}/{len(remaining_items)} = {honest_accuracy:.4f})")
    print(f"  Same entity, different attribute: {len(same_entity_clean)} ({same_ent_correct}/{len(same_entity_clean)} = {same_ent_correct/len(same_entity_clean):.4f})")
    print(f"  Unseen entity: {len(unseen_entity_clean)} ({unseen_ent_correct}/{len(unseen_entity_clean)} = {unseen_ent_correct/len(unseen_entity_clean):.4f})")

    report = {
        "excluded_ids": excluded_ids,
        "honest_accuracy": honest_accuracy,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {REPORT_PATH}")

    seal = json.loads(SEAL_PATH.read_text(encoding="utf-8"))
    payload = unseal(seal, excluded_ids)
    if payload is None:
        raise RuntimeError("Seal verification failed!")
    print("Seal verified successfully:")
    print(f"  certified honest accuracy: {payload['certified_honest_accuracy'] * 100:.2f}%")
    print(f"  flag: {payload['flag']}")


if __name__ == "__main__":
    main()
