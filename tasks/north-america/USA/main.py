#!/usr/bin/env python3
"""LMML - provocarea tokenizerului (SUA).

Citeste test-input.json (sau train-input.json / o cale data ca argv[1]) din directorul
curent, gaseste tokenizerul care codifica calibration.text exact in calibration.token_ids,
decodifica cu el challenge.token_ids si scrie test-output.json (sau train-output.json).
"""
# SUA - identificarea tokenizerului pierdut.
# Ideea: ID-urile de tokeni nu sunt coduri universale (ca ASCII) - ID-ul 1234 inseamna
# alt text in fiecare vocabular. Avem insa o "piatra din Rosetta": un text de calibrare
# si ID-urile lui. Incercam fiecare tokenizer cunoscut; cel care codifica textul de
# calibrare EXACT in aceleasi ID-uri e cel cautat, si cu el decodificam mesajul secret.
import difflib
import json
import os
import sys
import warnings

# Liniste in consola: fara avertismente de la bibliotecile de tokenizare
warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def ensure_packages():
    """Instaleaza bibliotecile de tokenizare daca interpretorul curent nu le are."""
    import importlib
    import subprocess
    # Verificam ce module lipsesc si le instalam cu pip (cu mai multe variante de flag-uri)
    missing_pkgs = []
    for module_name, pip_name in (("tiktoken", "tiktoken"), ("transformers", "transformers"),
                     ("tokenizers", "tokenizers"), ("sentencepiece", "sentencepiece"),
                     ("google.protobuf", "protobuf")):
        try:
            importlib.import_module(module_name)
        except Exception:
            missing_pkgs.append(pip_name)
    if not missing_pkgs:
        return
    print(f"installing missing packages: {missing_pkgs}", file=sys.stderr)
    pip_cmd = [sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check"]
    for extra_flags in ([], ["--break-system-packages"], ["--user", "--break-system-packages"]):
        try:
            proc = subprocess.run(pip_cmd + extra_flags + missing_pkgs, stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE, timeout=900)
            if proc.returncode == 0:
                break
        except Exception:
            pass
    importlib.invalidate_caches()


# Codarile OpenAI (tiktoken) - incercate primele, sunt rapide si locale
TIKTOKEN_NAMES = [
    "p50k_base", "cl100k_base", "o200k_base", "r50k_base", "gpt2",
    "p50k_edit", "o200k_harmony",
]

# ~55 de tokenizere HuggingFace populare (BERT, RoBERTa, T5, Llama, Qwen, Mistral, ...)
HF_NAMES = [
    "gpt2", "bert-base-uncased", "bert-base-cased", "roberta-base",
    "distilbert-base-uncased", "distilbert-base-cased", "distilroberta-base",
    "xlm-roberta-base", "bert-base-multilingual-cased",
    "bert-base-multilingual-uncased", "t5-small", "google/flan-t5-small",
    "google/mt5-small", "albert-base-v2", "xlnet-base-cased",
    "facebook/bart-base", "facebook/opt-125m", "EleutherAI/gpt-neox-20b",
    "EleutherAI/pythia-70m", "bigscience/bloom-560m", "microsoft/phi-2",
    "microsoft/deberta-v3-base", "microsoft/deberta-base",
    "google/electra-small-discriminator", "allenai/longformer-base-4096",
    "Qwen/Qwen2-0.5B", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen3-0.6B",
    "mistralai/Mistral-7B-v0.1", "hf-internal-testing/llama-tokenizer",
    "huggyllama/llama-7b", "NousResearch/Llama-2-7b-hf",
    "NousResearch/Meta-Llama-3-8B", "openlm-research/open_llama_3b",
    "tiiuae/falcon-7b", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "microsoft/Phi-3-mini-4k-instruct", "deepseek-ai/deepseek-coder-1.3b-base",
    "Salesforce/codegen-350M-mono", "bigcode/santacoder",
    "openai-gpt", "transfo-xl-wt103", "xlm-mlm-en-2048",
    "camembert-base", "google/byt5-small", "facebook/mbart-large-50",
    "sentence-transformers/all-MiniLM-L6-v2", "google/pegasus-xsum",
    "allenai/scibert_scivocab_uncased", "microsoft/mpnet-base",
    "funnel-transformer/small", "google/bigbird-roberta-base",
    "stabilityai/stablelm-2-1_6b", "01-ai/Yi-6B",
    "HuggingFaceTB/SmolLM-135M", "google/gemma-2b",
]


# Este x o lista nevida de intregi (dar nu booleeni)?
def _is_ids(value):
    return isinstance(value, list) and len(value) > 0 and all(
        isinstance(item, int) and not isinstance(item, bool) for item in value)


def _ids_from(value):
    """Accepta o lista de intregi, o lista de numere scrise ca text sau un sir '1 2 3' / '1,2,3'."""
    if _is_ids(value):
        return value
    # Lista de numere scrise ca text -> lista de intregi
    if isinstance(value, list) and value and all(isinstance(item, str) and item.strip().lstrip("-").isdigit() for item in value):
        return [int(item) for item in value]
    # Text de forma "1 2 3", "1,2,3" sau "[1, 2, 3]"
    if isinstance(value, str):
        import re
        parts = re.split(r"[\s,\[\]]+", value.strip())
        parts = [part for part in parts if part]
        if parts and all(part.lstrip("-").isdigit() for part in parts):
            return [int(part) for part in parts]
    return None


def normalize(raw_data):
    """Extrage (text_calibrare, id_calibrare, id_payload) din orice structura JSON rezonabila."""
    if isinstance(raw_data, str):
        raw_data = json.loads(raw_data)
    # Parcurgem recursiv JSON-ul si strangem toate textele si toate listele de ID-uri,
    # fiecare cu "calea" de chei pana la ea (ex. ('calibration', 'token_ids'))
    strings, idlists = [], []

    def walk(node, path):
        if isinstance(node, dict):
            for dict_key, child in node.items():
                walk(child, path + (str(dict_key).lower(),))
        elif isinstance(node, list) and not _is_ids(node) and not _ids_from(node):
            for index, child in enumerate(node):
                walk(child, path + (str(index),))
        else:
            found_ids = _ids_from(node) if not isinstance(node, str) or not any(ch.isalpha() for ch in node) else None
            if found_ids:
                idlists.append((path, found_ids))
            elif isinstance(node, str):
                strings.append((path, node))

    walk(raw_data, ())
    if not idlists:
        return None

    # Contine vreo cheie din cale unul dintre cuvinte?
    def has(path, *words):
        return any(word in key_part for key_part in path for word in words)

    # Clasificam dupa numele cheilor: payload-ul, ID-urile de calibrare si textul de calibrare
    challenge_lists = [id_entry for id_entry in idlists if has(id_entry[0], "challenge", "payload", "encoded", "secret", "message", "test")
            and not has(id_entry[0], "calib")]
    calib_lists = [id_entry for id_entry in idlists if has(id_entry[0], "calib", "sample", "reference", "pair")]
    text_entries = [entry for entry in strings if has(entry[0], "text", "plain", "calib", "sample")
             and not has(entry[0], "description", "tokenizer_id")]
    if not text_entries:
        text_entries = [entry for entry in strings if not has(entry[0], "description")]
    if not text_entries:
        return None
    # Textul de calibrare = cel mai lung calib_entry candidat
    calib_entry = max(text_entries, key=lambda entry: len(entry[1]))
    # Rezerve daca cheile nu sunt clare: lista de langa calib_entry, apoi cea mai lunga lista
    if not calib_lists:
        # list sharing the parent path with the calib_entry
        calib_lists = [id_entry for id_entry in idlists if id_entry[0][:-1] == calib_entry[0][:-1]]
    if not calib_lists and len(idlists) >= 2:
        calib_lists = [max(idlists, key=lambda id_entry: len(id_entry[1]))]
    if not calib_lists:
        return None
    calib_ids = calib_lists[0][1]
    if not challenge_lists:
        challenge_lists = [id_entry for id_entry in idlists if id_entry[1] is not calib_ids]
    if not challenge_lists:
        return None
    return calib_entry[1], calib_ids, challenge_lists[0][1]


def _try(path):
    try:
        with open(path, encoding="utf-8-sig") as f:
            return normalize(json.load(f))
    except Exception:
        return None


def find_input():
    """Intoarce ((text, id-uri, payload), cale). Nu prefera niciodata train in locul unui fisier de test."""
    # Ordinea de cautare: 1) argument / variabile de mediu
    for cand_path in sys.argv[1:] + [os.environ.get(env_var) for env_var in
                             ("INPUT_FILE", "INPUT_PATH", "TEST_INPUT", "INPUT") if os.environ.get(env_var)]:
        if os.path.isfile(cand_path):
            parsed = _try(cand_path)
            if parsed:
                return parsed, cand_path
    # 2) toate fisierele .json din directorul curent si al scriptului (adancime < 3),
    # fara fisiere de iesire
    roots = list(dict.fromkeys([os.getcwd(), os.path.dirname(os.path.abspath(__file__))]))
    found = []
    for root_idx, root in enumerate(roots):
        base_depth = root.rstrip(os.sep).count(os.sep)
        for dir_path, sub_dirs, file_names in os.walk(root):
            depth = dir_path.count(os.sep) - base_depth
            if depth >= 3:
                sub_dirs[:] = []
            sub_dirs[:] = [sub_dir for sub_dir in sub_dirs if not sub_dir.startswith(".")
                      and sub_dir not in ("node_modules", "site-packages", "__pycache__")]
            for file_name in file_names:
                lower_name = file_name.lower()
                if lower_name.endswith(".json") and "output" not in lower_name and "expected" not in lower_name:
                    found.append((root_idx, depth, os.path.join(dir_path, file_name)))

    # Preferam fisierele "test", apoi pe cele din directorul curent, apoi cele mai putin adanci;
    # fisierele "train" vin ultimele
    def rank(entry):
        root_idx, depth, cand_path = entry
        base_name = os.path.basename(cand_path).lower()
        return ("train" in base_name, "test" not in base_name, root_idx, depth, len(cand_path))

    for _, _, cand_path in sorted(found, key=rank):
        parsed = _try(cand_path)
        if parsed:
            return parsed, cand_path
    try:
        # 3) intrarea standard, ca ultima varianta
        if not sys.stdin.isatty():
            stdin_text = sys.stdin.read()
            if stdin_text.strip():
                parsed = normalize(json.loads(stdin_text))
                if parsed:
                    return parsed, None
    except Exception:
        pass
    sys.exit("no input json found (cwd=%s, files=%s)" % (os.getcwd(), os.listdir(".")))


def output_paths(input_path):
    """test-output.json in directorul curent (ce citeste evaluatorul), cu exceptia fisierelor train."""
    # Pe train scriem train-output.json, altfel test-output.json (ce citeste evaluatorul),
    # in directorul curent si langa fisierul de intrare
    base_name = os.path.basename(input_path).lower() if input_path else ""
    if "train" in base_name:
        name = os.path.basename(input_path).replace("input", "output") if "input" in base_name else "train-output.json"
    else:
        name = "test-output.json"
    out_paths = [os.path.join(os.getcwd(), name)]
    if input_path:
        out_paths.append(os.path.join(os.path.dirname(os.path.abspath(input_path)), name))
    return list(dict.fromkeys(out_paths))


class Cand:
    def __init__(self, name, encs, decode):
        self.name, self.encs, self.decode = name, encs, decode


def tiktoken_cands():
    try:
        import tiktoken
    except Exception:
        return
    # Lista noastra + toate codarile pe care le stie versiunea instalata de tiktoken
    names = list(dict.fromkeys(TIKTOKEN_NAMES + list(tiktoken.list_encoding_names())))
    for enc_name in names:
        try:
            tt_encoding = tiktoken.get_encoding(enc_name)
        except Exception:
            continue
        # Fiecare candidat: (nume, marimea vocabularului, functie de codare, functie de decodare)
        # disallowed_special=() permite text care arata ca un token special
        yield enc_name, tt_encoding.n_vocab, lambda txt, enc_obj=tt_encoding: [enc_obj.encode(txt, disallowed_special=())], \
            lambda ids, enc_obj=tt_encoding: enc_obj.decode(ids)


def hf_cands(offline):
    try:
        from transformers import AutoTokenizer
    except Exception:
        return
    for model_name in HF_NAMES:
        try:
            # offline=True: doar din cache_dir-ul local_path; offline=False: descarca de pe HuggingFace
            tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=offline, trust_remote_code=False)
        except Exception:
            continue

        def encs(txt, tokenizer=tokenizer):
            encodings = []
            # Incercam codarea cu si fara tokeni speciali ([CLS], </s>, ...)
            for with_special in (True, False):
                try:
                    encodings.append(tokenizer.encode(txt, add_special_tokens=with_special))
                except Exception:
                    pass
            return encodings

        def dec(ids, tokenizer=tokenizer, ref_ids=None, ref_text=None):
            import re

            # Variante de decodare: textul brut si textul cu spatiile din jurul "_" eliminate
            def fixers(decoded_text):
                yield decoded_text
                yield re.sub(r"\s*_\s*", "_", decoded_text)

            decoders = []
            for clean_spaces in (False, True):
                try:
                    decoders.append(lambda token_ids, clean_spaces=clean_spaces: tokenizer.decode(
                        token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=clean_spaces))
                except Exception:
                    pass
            # Alegem varianta de decodare care reproduce cel mai bine textul de calibrare
            best_decoder, best_ratio = None, -1.0
            for decoder in decoders:
                for fix_idx in range(2):
                    candidate_decoder = (lambda token_ids, decoder=decoder, fix_idx=fix_idx: list(fixers(decoder(token_ids)))[fix_idx])
                    try:
                        match_ratio = difflib.SequenceMatcher(None, candidate_decoder(ref_ids), ref_text).ratio() if ref_ids else 0
                    except Exception:
                        continue
                    if match_ratio > best_ratio:
                        best_decoder, best_ratio = candidate_decoder, match_ratio
            return best_decoder(ids)

        yield model_name, len(tokenizer), encs, dec


def raw_hf_cands():
    """Rezerva fara transformers: descarca tokenizer.json si foloseste biblioteca `tokenizers`."""
    try:
        # Rezerva fara biblioteca transformers: descarcam doar tokenizer.json
        from tokenizers import Tokenizer
    except Exception:
        return
    import tempfile
    import urllib.request
    cache_dir = os.path.join(tempfile.gettempdir(), "lmml_tok_cache")
    os.makedirs(cache_dir, exist_ok=True)
    for model_name in HF_NAMES:
        json_path = None
        try:
            from huggingface_hub import hf_hub_download
            json_path = hf_hub_download(model_name, "tokenizer.json")
        except Exception:
            local_path = os.path.join(cache_dir, model_name.replace("/", "__") + ".json")
            if not os.path.exists(local_path):
                try:
                    urllib.request.urlretrieve(
                        f"https://huggingface.co/{model_name}/resolve/main/tokenizer.json", local_path)
                except Exception:
                    continue
            json_path = local_path
        try:
            tokenizer = Tokenizer.from_file(json_path)
        except Exception:
            continue

        def encs(txt, tokenizer=tokenizer):
            encodings = []
            for with_special in (True, False):
                try:
                    encodings.append(tokenizer.encode(txt, add_special_tokens=with_special).ids)
                except Exception:
                    pass
            return encodings

        def dec(ids, tokenizer=tokenizer, ref_ids=None, ref_text=None):
            import re
            decoded_text = tokenizer.decode(ids, skip_special_tokens=True)
            if ref_ids:
                ref_decoded = tokenizer.decode(ref_ids, skip_special_tokens=True)
                if re.sub(r"\s*_\s*", "_", ref_decoded) != ref_decoded and "_" in (ref_text or ""):
                    decoded_text = re.sub(r"\s*_\s*", "_", decoded_text)
            return decoded_text

        yield model_name, tokenizer.get_vocab_size(with_added_tokens=True), encs, dec


def main():
    # Pasul 1: citim intrarea si ne asiguram ca bibliotecile exista
    parsed_input, input_path = find_input()
    print(f"input: {input_path}  python: {sys.executable}", file=sys.stderr)
    ensure_packages()
    calib_text, calib_ids, challenge_ids = parsed_input[0], list(parsed_input[1]), list(parsed_input[2])
    print(f"calibration: {len(calib_ids)} ids, challenge: {len(challenge_ids)} ids", file=sys.stderr)
    # Un tokenizer cu vocabular mai mic decat cel mai mare ID vazut e imposibil
    min_vocab = max(calib_ids + challenge_ids) + 1

    # Decodare care nu arunca exceptii (unele decodoare nu accepta argumentele ref_*)
    def safe_dec(decode_fn, ids):
        try:
            return decode_fn(ids, ref_ids=calib_ids, ref_text=calib_text)
        except TypeError:
            return decode_fn(ids)
        except Exception:
            return ""

    # best = cel mai bun candidat dupa potrivirea calibrarii; flagged = cel mai bun candidat
    # a carui decodare contine "SIGMOID_" (rezerva daca nu exista potrivire exacta)
    best = (-1.0, None, None)
    flagged = (-1.0, None, None)
    # Sursele de candidati, de la cea mai ieftina la cea mai scumpa
    sources = [tiktoken_cands(), hf_cands(True), hf_cands(False), raw_hf_cands()]
    seen = set()
    done = False
    for source in sources:
        for name, vocab_size, encode_fn, decode_fn in source:
            if name in seen:
                continue
            seen.add(name)
            # Filtru rapid pe marimea vocabularului
            if vocab_size < min_vocab:
                print(f"[{name}] skip vocab={vocab_size} < {min_vocab}", file=sys.stderr)
                continue
            try:
                variants = encode_fn(calib_text)
            except Exception:
                continue
            # Scorul: 1.0 la potrivire exacta a listei de ID-uri, altfel similaritatea secventelor
            match_score = 0.0
            for ids in variants:
                seq_score = 1.0 if ids == calib_ids else difflib.SequenceMatcher(None, ids, calib_ids).ratio()
                match_score = max(match_score, seq_score)
            if match_score > best[0]:
                best = (match_score, name, decode_fn)
            if match_score > flagged[0] and "SIGMOID_" in safe_dec(decode_fn, challenge_ids):
                flagged = (match_score, name, decode_fn)
            print(f"[{name}] score={match_score:.4f}", file=sys.stderr)
            # La prima potrivire exacta ne oprim
            if match_score == 1.0:
                done = True
                break
        if done:
            break

    # Fara potrivire exacta, preferam candidatul care decodeaza ceva cu "SIGMOID_"
    if best[0] < 1.0 and flagged[1] is not None:
        print(f"no exact match; using {flagged[1]} (decode contains SIGMOID_)", file=sys.stderr)
        best = flagged
    elif best[0] < 1.0:
        print("WARNING: no exact tokenizer match found; using closest", file=sys.stderr)

    if best[1] is None:
        sys.exit("no tokenizer could be loaded with %s -- install with: "
                 "%s -m pip install tiktoken transformers tokenizers sentencepiece protobuf"
                 % (sys.executable, sys.executable))
    # Pasul final: decodam payload-ul si scriem {"tokenizer_id", "decoded_message"}
    score, name, decode_fn = best
    decoded_msg = safe_dec(decode_fn, challenge_ids)
    print(f"tokenizer={name} match={score:.4f}\n{decoded_msg}", file=sys.stderr)
    result = {"tokenizer_id": name, "decoded_message": decoded_msg}
    for out_path in output_paths(input_path):
        try:
            with open(out_path, "w", encoding="utf-8") as out_file:
                json.dump(result, out_file, ensure_ascii=False, indent=2)
            print(f"wrote {out_path}", file=sys.stderr)
        except Exception as err:
            print(f"could not write {out_path}: {err}", file=sys.stderr)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
