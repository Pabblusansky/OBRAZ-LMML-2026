import torch, numpy as np, pandas as pd, time
from PIL import Image
import torchvision.transforms as T
D = "/home/cs/LMML/Sigmoid_LLM_Marathon/tasks/asia/vietnam-that-isnt-my-shoe/data/task_vietnam/"
m = torch.jit.load("model.pt"); m.eval()
m2 = torch.load("model.pt", weights_only=False); print("torch.load ok:", type(m2).__name__)
print("classes", m.class_names)
lab = pd.read_csv(D + "train-output.csv").sample(60, random_state=1)
imgs = [Image.open(D + "train-input/" + f).convert("RGB") for f in lab.filename]
variants = {
 "ToTensor CHW 0..1": [T.ToTensor()(i) for i in imgs],
 "uint8 HWC": [torch.from_numpy(np.array(i)) for i in imgs],
 "imagenet norm 224": [T.Compose([T.Resize(224), T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])(i) for i in imgs],
}
for name, tensors in variants.items():
    t = time.time()
    with torch.no_grad():
        single = [m.class_names[int(m(x).argmax())] for x in tensors[:20]]
        batch = m.predict(torch.stack(tensors))
    print(f"{name}: single acc {np.mean(np.array(single)==lab.label.values[:20]):.2f}  batch acc {np.mean(np.array(batch)==lab.label.values):.2f}  {time.time()-t:.1f}s")
print("random noise ->", m.predict(torch.rand(3,256,256)))
