import pandas as pd, re
d='files/task_morroco/'
df=pd.read_csv(d+'train-input.csv').merge(pd.read_csv(d+'train-answers.csv'))
n=lambda s:s.astype(str).str.strip().str.lower()
df['c']=n(df.cheap_answer)==n(df.reference_answer); df['e']=n(df.expensive_answer)==n(df.reference_answer)
def kind(p):
    p=p.lower()
    if 'walking times' in p: return 'walk'
    if 'takings' in p: return 'takings_fee' if 'stall fee' in p else ('takings_most' if 'most' in p else 'takings_least' if 'least' in p else 'takings_other')
    if 'opens at' in p: return 'hours_pause' if 'pauses' in p else 'hours_simple'
    if 'board lists' in p: return 'board_indirect' if 'whatever' in p else 'board_lookup'
    if 'loads' in p: return 'crates_multi' if ('sells' in p or 'hands' in p) else 'crates_simple'
    if 'toll' in p: return 'toll_heavy' if 'heavier' in p else 'toll_simple'
    return 'other'
df['kind']=df.prompt.map(kind)
g=df.groupby('kind').agg(n=('c','size'),cheap=('c','mean'),exp=('e','mean')).round(2); print(g)
print(df[df.kind=='other'].prompt.head(5).tolist())
df['loss']=df.e.astype(int)-df.c.astype(int)
print(df.groupby('kind').loss.mean().sort_values().round(2))
import itertools
kinds=sorted(df.kind.unique())
order=df.groupby('kind').loss.mean().sort_values().index.tolist()
for k in range(1,len(order)+1):
    ch=set(order[:k]); cheap=df.kind.isin(ch)
    q=(df.c[cheap].sum()+df.e[~cheap].sum())/len(df); cost=(cheap.sum()+25*(~cheap).sum())/len(df)
    print(k, order[k-1].ljust(15), 'quality %.3f cost/request %.2f'%(q,cost))
