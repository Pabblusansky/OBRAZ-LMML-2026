import json,hashlib,itertools,sys
d=json.load(open(sys.argv[1]))
R=d['trusted_reference']+d['samples']
pre={bytes(a^b for a,b in zip(bytes.fromhex(r['archive_tag'])[:8],b'SIGMOID_')):r for r in R}
def test(idx):
    m=hashlib.sha256(','.join(map(str,sorted(idx))).encode()).digest()
    r=pre.get(m[:8])
    if r:
        print('FOUND',sorted(idx),bytes(x^y for x,y in zip(bytes.fromhex(r['archive_tag']),m)).rstrip(b'\0'));return True
base=[36,119,216,273,357,367,462,478,513,685,752,1199]
allI=[s['sample_index'] for s in d['samples']]
for k in range(len(base)+1):
    for c in itertools.combinations(base,k): test(c)
rest=[i for i in allI if i not in base]
for i in rest: test(base+[i])
for a,b in itertools.combinations(rest,2): test(base+[a,b])
print('done')
