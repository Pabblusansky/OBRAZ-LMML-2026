import cv2, numpy as np, glob
def box(a,w,h): return cv2.boxFilter(a,-1,(w,h),normalize=True,borderType=cv2.BORDER_REFLECT)
def smooth_score(img, sizes=((64,36),(84,32),(104,38),(54,34))):
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float32)
    n=np.abs(g-cv2.GaussianBlur(g,(0,0),1.0))
    best=np.zeros_like(g); t=8
    for w,h in sizes:
        inside=box(n,w,h)
        strips=[np.roll(box(n,w,t),-(h//2+t//2+1),0),np.roll(box(n,w,t),(h//2+t//2+1),0),np.roll(box(n,t,h),-(w//2+t//2+1),1),np.roll(box(n,t,h),(w//2+t//2+1),1)]
        s=np.sort(np.stack(strips),0)
        ring=s[1]                                   # second-quietest side
        sc=(ring+0.05)/(inside+0.05)
        sc[(s[3]>2.5)]=0                            # sides must be plain paper, not print edges
        best=np.maximum(best,sc)
    return best
if __name__=='__main__':
    K='data/task_kazakhstan/'; hit=0; tot=0
    for p in sorted(glob.glob(K+'exhibits/exhibit_??.jpg')):
        img=cv2.imread(p); m=cv2.imread(p.replace('.jpg','_mask.png'),0)>127
        s=smooth_score(img); y,x=np.unravel_index(s.argmax(),s.shape); x0,y0,w,h=cv2.boundingRect(m.astype(np.uint8))
        g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float32); n=np.abs(g-cv2.GaussianBlur(g,(0,0),1.0))
        ring=np.zeros_like(m); ring[max(0,y0-10):y0+h+10,max(0,x0-10):x0+w+10]=True; ring&=~m
        print(p[-14:],'box',(w,h),'peak %.2f'%s.max(),'hit' if m[y,x] else 'miss','score@mask %.2f'%s[m].max(),'noise in/out %.2f/%.2f'%(n[m].mean(),n[ring].mean()))
        hit+=m[y,x]; tot+=1
    print(hit,tot)
