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
    cands = sys.argv[1:] + ["test-input.csv", "input.csv", "train-input.csv",
                            os.path.join("files", "train-input.csv")]
    for p in cands:
        if p and os.path.isfile(p):
            return p
    # Ultima rezerva: primul CSV din directorul curent
    csvs = [f for f in os.listdir(".") if f.lower().endswith(".csv")]
    if csvs:
        return csvs[0]
    sys.exit("no input csv found")


def output_path(inp):
    """Pe train scriem intr-un fisier separat, ca sa nu suprascriem raspunsul corect."""
    base = os.path.basename(inp).lower()
    return "train-output.pred.json" if "train" in base else "test-output.json"


def record_key(addr):
    """Construieste cheia persoanei doar din adresa: (numere, stat, cod postal)."""
    addr = (addr or "").strip()
    # Codul postal (sau "" daca lipseste)
    z = ZIP_RE.search(addr)
    zip5 = z.group(1) if z else ""
    # Statul, cu majuscule (sau "" daca nu se poate parsa)
    st = STATE_RE.search(addr)
    state = st.group(1).upper() if st else ""
    # Partea de strada = tot ce e inaintea codului postal...
    street = addr[:z.start()] if z else addr
    # ...fara ultimul segment dupa virgula (statul), ca sa nu-l amestecam cu numerele
    if "," in street and st:
        street = street.rsplit(",", 1)[0]
    # Toate numerele din strada, fara zerouri la inceput ("007" -> "7"),
    # sortate: ordinea in care apar nu conteaza (multiset)
    nums = tuple(sorted(n.lstrip("0") or "0" for n in re.findall(r"\d+", street)))
    return nums, state, zip5


def main():
    inp = find_input()
    # utf-8-sig elimina eventualul BOM de la inceputul fisierului
    with open(inp, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    # "Blocare" (blocking): grupam inregistrarile dupa cheie; comparam doar
    # in interiorul aceluiasi bloc, deci nu facem comparatii pe toate perechile.
    blocks = defaultdict(list)
    for r in rows:
        rid = (r.get("record_id") or "").strip()
        if not rid:
            continue
        # Tabelul-sursa; daca lipseste coloana, folosim prima litera a ID-ului
        src = (r.get("source_table") or rid[:1]).strip()
        blocks[record_key(r.get("address"))].append((src, rid))

    matches = []
    seen = set()
    for members in blocks.values():
        # Toate perechile din bloc
        for (sa, a), (sb, b) in itertools.combinations(members, 2):
            # Doar intre tabele DIFERITE: duplicatele din acelasi tabel nu sunt
            # aceeasi persoana (asta tine precizia la 100%)
            if sa == sb or a == b:
                continue
            # Pereche normalizata (sortata), ca (a, b) si (b, a) sa fie la fel
            k = tuple(sorted((a, b)))
            if k not in seen:
                seen.add(k)
                matches.append(list(k))
    # Ordine stabila in fisierul de iesire
    matches.sort()

    result = {"matches": matches}
    out = output_path(inp)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    # Rezumat pe stderr, rezultatul complet pe stdout
    print(f"input={inp} records={len(rows)} matches={len(matches)} -> {out}", file=sys.stderr)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
