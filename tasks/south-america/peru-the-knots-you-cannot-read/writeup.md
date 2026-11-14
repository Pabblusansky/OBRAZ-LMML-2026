# The Knots You Cannot Read: write-up

**Flag:** `SIGMOID_PROVENANCE`

## 1. The model does what it claims
`model.safetensors` holds three tensors: `embedding.weight` [256,32], `classifier.weight` [3,32] and `classifier.bias` [3].
The forward pass is: embed each byte, take the mean over the 64 positions, then apply the linear layer.
Run on `corpus.csv`, it scores **95.4%** accuracy (`scripts/verify.py`).

## 2. Where the payload is
The corpus uses only 17 characters (space, `a c h i k l m n p q r s t u w y`), and the input is ASCII, so bytes 0–127 at most.
That means **embedding rows 128–255 can never be looked up**, and nothing stored in them can reach the output.

Row **128** looks different from every other row. Its first 18 floats are small negatives (≈ −0.03 to −0.06), and in each one the **low 16 bits of the float are exactly zero** (`0xbd340000`, `0xbd5c0000`, …). Trained weights don't come out like that, so these values were written by hand.

## 3. How it is encoded
In each of these floats, bits 18–22 (the top 5 bits of the mantissa) hold a 5-bit value `v`. The letter index is `31 − v`, with A=0 … Z=25 and `_`=30.

```
v      = 13 23 25 19 17 23 28  1 16 14 17 10 27 18 31 18 29 27
31 − v = 18  8  6 12 14  8  3 30 15 17 14 21  4 13  0 13  2  4
         S   I  G  M  O  I  D  _  P  R  O  V  E  N  A  N  C  E
```

## 4. Extraction code (`scripts/extract.py`)
```python
import json, struct, sys, numpy as np
b=open(sys.argv[1],'rb').read(); n=struct.unpack('<Q',b[:8])[0]; h=json.loads(b[8:8+n]); d=b[8+n:]
s,e=h['embedding.weight']['data_offsets']; E=np.frombuffer(d[s:e],dtype='<f4').reshape(256,32)
u=E[128].view('<u4')
k=u[(u&0xffff)==0]                 # the 18 planted values
v=((k>>16)&0x7f)>>2                # 5-bit field
print(''.join('ABCDEFGHIJKLMNOPQRSTUVWXYZ____'[31-x] for x in v))   # SIGMOID_PROVENANCE
```
