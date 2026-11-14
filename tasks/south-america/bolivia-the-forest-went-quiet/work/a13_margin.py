import sys,os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"a12_adapt.py")).read().split("for kind in")[0])
for ef in (0.0095,0.005,0.003,0.0015):
    p=predict_probabilities(model,apply(R,0.07,0.34,"tukey",True,ef),batch_size=32)
    hit=p.argmax(1)==3
    print(f"energy {100*ef:.2f}%  rate={hit.mean():.3f}", {c:round(float(hit[L==c].mean()),2) for c in ("birds","frogs","insects")}, "min P(rain) non-rain=%.2f"%p[L!="rain",3].min())
