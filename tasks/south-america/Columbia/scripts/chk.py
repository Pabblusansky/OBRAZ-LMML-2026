import json,hashlib,sys
d=json.load(open(sys.argv[1]))
def check(idx):
    m=hashlib.sha256(','.join(map(str,sorted(idx))).encode()).digest()
    out=[]
    for r in d['trusted_reference']+d['samples']:
        b=bytes(x^y for x,y in zip(bytes.fromhex(r['archive_tag']),m)).rstrip(b'\0')
        if b.startswith(b'SIGMOID') or (len(b)>3 and all(32<=c<127 for c in b)): out.append(b)
    return out
