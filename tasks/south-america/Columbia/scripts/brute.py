# Columbia - cautare prin forta bruta a multimii de mostre modificate.
# Regula sigiliului: SHA-256 al indicilor modificati (sortati, uniti cu virgula) = masca;
# archive_tag XOR masca = flag-ul. Flag-ul incepe cu "SIGMOID_", deci primii 8 octeti
# ai mastii corecte = primii 8 octeti ai unui tag XOR "SIGMOID_".
import json,hashlib,itertools,sys
d=json.load(open(sys.argv[1]))
# Toate inregistrarile (de incredere + mostre); oricare poate purta sigiliul
R=d['trusted_reference']+d['samples']
# Precalculam, pentru fiecare tag, "inceputul de masca" care ar da "SIGMOID_"
# -> dictionar: 8 octeti -> inregistrarea; o cautare apoi costa O(1)
pre={bytes(a^b for a,b in zip(bytes.fromhex(r['archive_tag'])[:8],b'SIGMOID_')):r for r in R}
def test(idx):
    # Masca pentru multimea candidata
    m=hashlib.sha256(','.join(map(str,sorted(idx))).encode()).digest()
    r=pre.get(m[:8])
    if r:
        # Potrivire: decodam tot tag-ul si eliminam octetii zero de la final
        print('FOUND',sorted(idx),bytes(x^y for x,y in zip(bytes.fromhex(r['archive_tag']),m)).rstrip(b'\0'));return True
# Cele 12 mostre cu caractere ascunse (spatiu de latime zero / litere chirilice)
base=[36,119,216,273,357,367,462,478,513,685,752,1199]
allI=[s['sample_index'] for s in d['samples']]
# 1. Toate submultimile celor 12 (poate doar o parte sunt "modificate")
for k in range(len(base)+1):
    for c in itertools.combinations(base,k): test(c)
# 2. Cele 12 + inca o mostra oarecare
rest=[i for i in allI if i not in base]
for i in rest: test(base+[i])
# 3. Cele 12 + oricare doua mostre (cautare mai lunga)
for a,b in itertools.combinations(rest,2): test(base+[a,b])
print('done')
