# Фактура холста за краем кадра (сцены с tex в BG). Мелкая фактура (кадр минус размытый, сигма 8) у левого и правого
# края кадра — полосы стены без картины; каждая превращается в бесшовную плитку: повтор полосы с перекрытием, стык — по
# линии наименьшей разницы (по ширине — вертикальной, по высоте — горизонтальной). Плитка стоит в фазе с кадром, поэтому
# за краем кадра фактура продолжается без шва. Крупный тон даёт фон из краёв (build.background).
# → <tmp>/tex<i>.npz: L, R — плитки (float16), L0, R0 — столбец кадра, с которого начинается плитка, Lm, Rm — стык у края. Запуск: python3 texture.py <мезонин> <tmp>
import numpy as np,cv2,sys
from score import *
import build
OV=24
def mincut(err):  # err (h, w): путь сверху вниз наименьшей суммы; возвращает столбец стыка в каждой строке
  h,w=err.shape; E=err.copy(); B=np.zeros((h,w),np.int32)
  for y in range(1,h):
    p=np.stack([np.r_[np.inf,E[y-1,:-1]],E[y-1],np.r_[E[y-1,1:],np.inf]]); k=np.argmin(p,0); B[y]=k-1; E[y]+=p[k,np.arange(w)]
  path=np.zeros(h,np.int32); path[-1]=np.argmin(E[-1])
  for y in range(h-1,0,-1): path[y-1]=path[y]+B[y,path[y]]
  return path
def wrapx(s):
  # s (h, w): плитка с периодом w-OV — правые OV столбцов ложатся на левые OV столбцов следующей копии
  h,w=s.shape[:2]; a=s[:,w-OV:]; b=s[:,:OV]; p=mincut(((a-b)**2).sum(2))
  m=(np.arange(OV)[None,:]<p[:,None]).astype(np.float32)          # 1 — из конца полосы (a), 0 — из начала (b)
  m=cv2.GaussianBlur(m,(0,0),.8)[...,None]
  t=s[:,:w-OV].copy(); t[:,:OV]=a*m+b*(1-m)
  return t,m[...,0]
def run(i,mz,tmp,left,right):
  g=BG[i]; a,b=g['set']
  fr=np.stack([f for _,f in build.frames(mz,a,b)][::max(1,(b-a)//25)]).astype(np.float32)
  med=np.median(fr,0); hf=med-cv2.GaussianBlur(med,(0,0),8)
  out={}
  for nm,(c0,c1) in (('L',left),('R',right)):
    s=hf[:,c0:c1]
    t,m=wrapx(s)                                 # период по ширине: c1-c0-OV, фаза — от столбца c0
    t,_=wrapx(t.transpose(1,0,2)); t=t.transpose(1,0,2)  # период по высоте: BH-OV, фаза — от строки 0
    # m — стык у края кадра (OV столбцов): у левого края кадр виден правее линии (1-m), у правого — левее (m)
    out[nm]=t.astype(np.float16); out[nm+'0']=np.array([c0]); out[nm+'m']=m.astype(np.float16)
  np.savez(f'{tmp}/tex{i}.npz',**out)
  print('фактура',i,{k:v.shape for k,v in out.items()},file=sys.stderr)
if __name__=='__main__':
  mz,tmp=sys.argv[1],sys.argv[2]
  run(3,mz,tmp,(0,400),(1520,1920)); run(11,mz,tmp,(0,400),(1520,1920))
