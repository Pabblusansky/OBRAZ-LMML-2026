# Columbia - verificarea unei multimi candidate (functie folosita interactiv).
# check(lista_de_indici) face XOR intre masca SHA-256 si TOATE tag-urile si intoarce
# rezultatele care incep cu "SIGMOID" sau sunt text ASCII lizibil.
import json,hashlib,sys
d=json.load(open(sys.argv[1]))
def check(idx):
    # Masca = SHA-256 al indicilor sortati, uniti cu virgula, fara spatii
    m=hashlib.sha256(','.join(map(str,sorted(idx))).encode()).digest()
    out=[]
    for r in d['trusted_reference']+d['samples']:
        # XOR tag cu masca, apoi eliminam octetii zero de la final (din nota arhivei)
        b=bytes(x^y for x,y in zip(bytes.fromhex(r['archive_tag']),m)).rstrip(b'\0')
        # Pastram doar ce pare flag sau text tiparibil (caractere 32..126)
        if b.startswith(b'SIGMOID') or (len(b)>3 and all(32<=c<127 for c in b)): out.append(b)
    return out
