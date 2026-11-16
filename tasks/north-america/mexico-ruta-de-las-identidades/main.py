"""Ruta de las Identidades - legarea inregistrarilor (record linkage) pentru Mexic.

Citeste test-input.csv (sau o cale data ca argv[1], sau train-input.csv /
files/train-input.csv ca rezerva) si scrie {"matches": [[id1, id2], ...]} in
test-output.json (train-output.pred.json cand ruleaza pe train); JSON-ul e
afisat si pe stdout.

Regula (100% precizie si recall pe train): doua inregistrari din tabele-sursa
DIFERITE sunt aceeasi persoana cand adresele lor au aceleasi:
  - multiset de numere din partea de strada (nr. casa, Int./Edif./Depto.),
  - abrevierea statului si
  - codul postal de 5 cifre.
Numele sunt ignorate: in duplicate sunt inversate, abreviate, fara diacritice si
scrise gresit, in timp ce numerele din adresa sunt pastrate exact.
"""
import csv
import itertools
import json
import os
import re
import sys
from collections import defaultdict

# Codul postal: 5 cifre la finalul adresei, optional urmate de "-1234"
ZIP_RE = re.compile(r"(\d{5})(?:-\d{4})?\s*$")
# Statul: un cuvant (ex. "CDMX", "Jal.") dupa ultima virgula, chiar inaintea codului postal
STATE_RE = re.compile(r",\s*([A-Za-z]+)\.?\s+\d{5}(?:-\d{4})?\s*$")


def find_input():
    """Gaseste fisierul de intrare: argumentul, apoi numele obisnuite, apoi orice CSV."""
    candidates = sys.argv[1:] + ["test-input.csv", "input.csv", "train-input.csv",
                            os.path.join("files", "train-input.csv")]
    for cand_path in candidates:
        if cand_path and os.path.isfile(cand_path):
            return cand_path
    # Ultima rezerva: primul CSV din directorul curent
    csv_files = [file_name for file_name in os.listdir(".") if file_name.lower().endswith(".csv")]
    if csv_files:
        return csv_files[0]
    sys.exit("no input csv found")


def output_path(input_path):
    """Pe train scriem intr-un fisier separat, ca sa nu suprascriem raspunsul corect."""
    base_name = os.path.basename(input_path).lower()
    return "train-output.pred.json" if "train" in base_name else "test-output.json"


def record_key(address):
    """Construieste cheia persoanei doar din adresa: (numere, stat, cod postal)."""
    address = (address or "").strip()
    # Codul postal (sau "" daca lipseste)
    zip_match = ZIP_RE.search(address)
    zip5 = zip_match.group(1) if zip_match else ""
    # Statul, cu majuscule (sau "" daca nu se poate parsa)
    state_match = STATE_RE.search(address)
    state = state_match.group(1).upper() if state_match else ""
    # Partea de strada = tot ce e inaintea codului postal...
    street = address[:zip_match.start()] if zip_match else address
    # ...fara ultimul segment dupa virgula (statul), ca sa nu-l amestecam cu numerele
    if "," in street and state_match:
        street = street.rsplit(",", 1)[0]
    # Toate numerele din strada, fara zerouri la inceput ("007" -> "7"),
    # sortate: ordinea in care apar nu conteaza (multiset)
    street_numbers = tuple(sorted(num.lstrip("0") or "0" for num in re.findall(r"\d+", street)))
    return street_numbers, state, zip5


def main():
    input_path = find_input()
    # utf-8-sig elimina eventualul BOM de la inceputul fisierului
    with open(input_path, encoding="utf-8-sig", newline="") as fh:
        records = list(csv.DictReader(fh))

    # "Blocare" (blocking): grupam inregistrarile dupa cheie; comparam doar
    # in interiorul aceluiasi bloc, deci nu facem comparatii pe toate perechile.
    blocks = defaultdict(list)
    for record in records:
        record_id = (record.get("record_id") or "").strip()
        if not record_id:
            continue
        # Tabelul-sursa; daca lipseste coloana, folosim prima litera a ID-ului
        source_table = (record.get("source_table") or record_id[:1]).strip()
        blocks[record_key(record.get("address"))].append((source_table, record_id))

    matches = []
    seen_pairs = set()
    for block_records in blocks.values():
        # Toate perechile din bloc
        for (source_a, id_a), (source_b, id_b) in itertools.combinations(block_records, 2):
            # Doar intre tabele DIFERITE: duplicatele din acelasi tabel nu sunt
            # aceeasi persoana (asta tine precizia la 100%)
            if source_a == source_b or id_a == id_b:
                continue
            # Pereche normalizata (sortata), ca (id_a, id_b) si (id_b, id_a) sa fie la fel
            pair = tuple(sorted((id_a, id_b)))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                matches.append(list(pair))
    # Ordine stabila in fisierul de iesire
    matches.sort()

    result = {"matches": matches}
    out_path = output_path(input_path)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False)
    # Rezumat pe stderr, rezultatul complet pe stdout
    print(f"input={input_path} records={len(records)} matches={len(matches)} -> {out_path}", file=sys.stderr)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
