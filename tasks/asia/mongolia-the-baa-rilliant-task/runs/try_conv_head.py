# experiment: small conv head (3x3 context) on cached cell features, 5-fold CV
import numpy as np, pandas as pd, torch, torch.nn as nn, sys
from importlib import import_module
sys.path.insert(0, '.')
m = import_module("04_count_and_map")
torch.set_num_threads(4); torch.manual_seed(0)
tc = pd.read_csv(m.DATA_FOLDER/"train_counts.csv"); tp = pd.read_csv(m.DATA_FOLDER/"train_points.csv")
X = torch.from_numpy(np.load(m.RUNS_FOLDER/"train_cell_features.npy")).permute(0,3,1,2)   # (280,384,38,38) fp16
Y = torch.from_numpy(m.sheep_per_cell(tc.image, tp)).unsqueeze(1)
true = tc["count"].to_numpy()
def make(): return nn.Sequential(nn.Conv2d(384,128,3,padding=1), nn.ReLU(), nn.Conv2d(128,64,3,padding=1), nn.ReLU(), nn.Conv2d(64,1,1))
def train(idx, epochs=60):
    net = make(); opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    for ep in range(epochs):
        perm = np.random.permutation(idx)
        for b in range(0, len(perm), 16):
            bi = perm[b:b+16]; xb = X[bi].float(); yb = Y[bi]
            if np.random.rand() < .5: xb, yb = xb.flip(3), yb.flip(3)
            if np.random.rand() < .5: xb, yb = xb.flip(2), yb.flip(2)
            pred = net(xb)
            loss = ((pred-yb)**2).mean()*100 + 0.01*((pred.sum((1,2,3))-yb.sum((1,2,3))).abs()).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return net
fold = np.arange(280)%5; val = np.zeros(280)
for f in range(5):
    net = train(np.where(fold!=f)[0])
    with torch.no_grad(): val[fold==f] = net(X[fold==f].float()).sum((1,2,3)).numpy()
    print(f, np.abs(val[fold==f]-true[fold==f]).mean(), flush=True)
print("MAE", np.abs(val-true).mean(), "corr", np.corrcoef(val,true)[0,1])
