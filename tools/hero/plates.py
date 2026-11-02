# Края картины по сценам → assets/borders.npz: для каждой сцены прямоугольник картины и цвета по её периметру
# (сверху слева направо, справа сверху вниз, снизу справа налево, слева снизу вверх), 1 значение на px базового кадра.
# Запуск: python3 plates.py <мезонин 1920×1080>  (один раз; сборка читает готовый файл)
import numpy as np,subprocess,sys,os
from score import *
S=os.path.dirname(os.path.abspath(__file__)); MZ=sys.argv[1]
def frames(a,b,step):
  r=subprocess.run(['ffmpeg','-v','error','-ss','%.5f'%(a/30),'-i',MZ,'-vf',f"select='not(mod(n\\,{step}))',format=rgb24",
    '-frames:v',str((b-a+step-1)//step),'-vsync','0','-f','rawvideo','-'],capture_output=True,check=True).stdout
  return np.frombuffer(r,np.uint8).reshape(-1,BH,BW,3)
D0,D1=3,13   # полоса края: от 3 до 13 px внутрь картины
out={}
for i,g in enumerate(BG):
  a,b=g['set']; f=frames(a,b,max(1,(b-a)//45))
  x0,y0,x1,y1=[int(round(v)) for v in g['rect']]
  # медиана по времени — движущееся уходит; затем по глубине полосы
  T=np.median(f[:,y0+D0:y0+D1,x0:x1].astype(np.float32),0)          # (10, w, 3)
  B=np.median(f[:,y1-D1:y1-D0,x0:x1].astype(np.float32),0)[::-1]    # снизу вверх
  R=np.median(f[:,y0:y1,x1-D1:x1-D0].astype(np.float32),0).transpose(1,0,2)[::-1]  # (10, h, 3) снаружи внутрь
  L=np.median(f[:,y0:y1,x0+D0:x0+D1].astype(np.float32),0).transpose(1,0,2)
  # порядок обхода: T слева направо, R сверху вниз, B справа налево, L снизу вверх
  seq=[T, R, B[:, ::-1], L[:, ::-1]]
  cols=[];val=[]
  for s in seq:
    if g['ref'] is None: ok=np.ones(s.shape[:2],bool)
    else: ok=np.min([np.abs(s-np.array(c,np.float32)).max(2) for c in g['ref']],0)<TOL
    m=ok.sum(0)>=3
    c=np.array([np.median(s[ok[:,j],j],0) if m[j] else [0,0,0] for j in range(s.shape[1])],np.float32)
    cols.append(c); val.append(m)
  C=np.concatenate(cols); V=np.concatenate(val)
  print(i,'периметр',len(C),'годных %.1f%%'%(100*V.mean()),file=sys.stderr)
  out[f'c{i}']=C.astype(np.float32); out[f'v{i}']=V; out[f'r{i}']=np.array([x0,y0,x1,y1],np.int32)
np.savez_compressed(f'{S}/assets/borders.npz',**out)
