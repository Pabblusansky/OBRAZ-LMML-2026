import cv2, numpy as np, glob
from sklearn.metrics import roc_auc_score
K='data/task_kazakhstan/'
ex=sorted(glob.glob(K+'exhibits/exhibit_??.jpg'))
def feats(img):
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float32)
    f={}
    hp=g-cv2.GaussianBlur(g,(0,0),1.0); f['noise']=cv2.boxFilter(hp*hp,-1,(15,15))
    med=g-cv2.medianBlur(g.astype(np.uint8),3).astype(np.float32); f['mednoise']=cv2.boxFilter(med*med,-1,(15,15))
    for q in (70,80,90,95):
        again=cv2.imdecode(cv2.imencode('.jpg',img,[cv2.IMWRITE_JPEG_QUALITY,q])[1],1)
        f[f'ela{q}']=cv2.boxFilter(np.abs(img.astype(np.float32)-again).mean(2),-1,(15,15))
    dx=np.abs(2*g[:,1:-1]-g[:,:-2]-g[:,2:]); dx=np.pad(dx,((0,0),(1,1)))
    for ph in range(8):
        on=np.zeros_like(g); on[:,ph::8]=dx[:,ph::8]; off=dx.copy(); off[:,ph::8]=0
        f[f'blk{ph}']=cv2.boxFilter(on,-1,(33,33))*8-cv2.boxFilter(off,-1,(33,33))*8/7
    lap=cv2.Laplacian(g,cv2.CV_32F); f['lap']=cv2.boxFilter(np.abs(lap),-1,(15,15))
    return f
if __name__=='__main__':
    agg={}; sizes=[]
    for p in ex:
        img=cv2.imread(p); m=cv2.imread(p.replace('.jpg','_mask.png'),0)>127
        sizes.append((int(m.sum()), cv2.boundingRect(m.astype(np.uint8))[2:], img.shape[:2]))
        for k,v in feats(img).items(): agg.setdefault(k,[]).append(roc_auc_score(m.ravel(),v.ravel()))
    for k,v in agg.items(): print(f'{k:9s} mean AUC {np.mean(v):.3f}  min {np.min(v):.3f} max {np.max(v):.3f}  frac>0.9 {(np.array(v)>0.9).mean():.2f} frac<0.1 {(np.array(v)<0.1).mean():.2f}')
    print(sizes[:8])
    tiles=[]
    for p in ex[:6]:
        img=cv2.imread(p); m=cv2.imread(p.replace('.jpg','_mask.png'),0)
        cs,_=cv2.findContours((m>127).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(img,cs,-1,(0,255,0),2)
        tiles.append(cv2.resize(img,(548,412)))
    cv2.imwrite('work/exhibits.jpg',np.vstack([np.hstack(tiles[:3]),np.hstack(tiles[3:])]))
