# Columbia - cautare prin forta bruta a multimii de mostre modificate.
# Regula sigiliului: SHA-256 al indicilor modificati (sortati, uniti cu virgula) = masca;
# archive_tag XOR masca = flag-ul. Flag-ul incepe cu "SIGMOID_", deci primii 8 octeti
# ai mastii corecte = primii 8 octeti ai unui tag XOR "SIGMOID_".
import json,hashlib,itertools,sys
data=json.load(open(sys.argv[1]))
# Toate inregistrarile (de incredere + mostre); oricare poate purta sigiliul
all_records=data['trusted_reference']+data['samples']
# Precalculam, pentru fiecare tag, "inceputul de masca" care ar da "SIGMOID_"
# -> dictionar: 8 octeti -> inregistrarea; o cautare apoi costa O(1)
prefix_to_record={bytes(tag_byte^flag_byte for tag_byte,flag_byte in zip(bytes.fromhex(record['archive_tag'])[:8],b'SIGMOID_')):record for record in all_records}
def test_candidate(sample_indices):
    # Masca pentru multimea candidata
    mask=hashlib.sha256(','.join(map(str,sorted(sample_indices))).encode()).digest()
    record=prefix_to_record.get(mask[:8])
    if record:
        # Potrivire: decodam tot tag-ul si eliminam octetii zero de la final
        print('FOUND',sorted(sample_indices),bytes(tag_byte^mask_byte for tag_byte,mask_byte in zip(bytes.fromhex(record['archive_tag']),mask)).rstrip(b'\0'));return True
# Cele 12 mostre cu caractere ascunse (spatiu de latime zero / litere chirilice)
hidden_char_samples=[36,119,216,273,357,367,462,478,513,685,752,1199]
all_sample_indices=[sample['sample_index'] for sample in data['samples']]
# 1. Toate submultimile celor 12 (poate doar o parte sunt "modificate")
for subset_size in range(len(hidden_char_samples)+1):
    for subset in itertools.combinations(hidden_char_samples,subset_size): test_candidate(subset)
# 2. Cele 12 + inca o mostra oarecare
other_samples=[extra_sample for extra_sample in all_sample_indices if extra_sample not in hidden_char_samples]
for extra_sample in other_samples: test_candidate(hidden_char_samples+[extra_sample])
# 3. Cele 12 + oricare doua mostre (cautare mai lunga)
for extra_a,extra_b in itertools.combinations(other_samples,2): test_candidate(hidden_char_samples+[extra_a,extra_b])
print('done')
