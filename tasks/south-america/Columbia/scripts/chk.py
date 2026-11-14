# Columbia - verificarea unei multimi candidate (functie folosita interactiv).
# check(lista_de_indici) face XOR intre masca SHA-256 si TOATE tag-urile si intoarce
# rezultatele care incep cu "SIGMOID" sau sunt text ASCII lizibil.
import json,hashlib,sys
data=json.load(open(sys.argv[1]))
def check(sample_indices):
    # Masca = SHA-256 al indicilor sortati, uniti cu virgula, fara spatii
    mask=hashlib.sha256(','.join(map(str,sorted(sample_indices))).encode()).digest()
    candidates=[]
    for record in data['trusted_reference']+data['samples']:
        # XOR tag cu masca, apoi eliminam octetii zero de la final (din nota arhivei)
        decoded=bytes(tag_byte^mask_byte for tag_byte,mask_byte in zip(bytes.fromhex(record['archive_tag']),mask)).rstrip(b'\0')
        # Pastram doar ce pare flag sau text tiparibil (caractere 32..126)
        if decoded.startswith(b'SIGMOID') or (len(decoded)>3 and all(32<=byte<127 for byte in decoded)): candidates.append(decoded)
    return candidates
