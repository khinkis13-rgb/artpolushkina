# Сборщик «всё в одном ролике»: кадр = фон сцены по таблицам heroSync + ролик в своей обрезке + волны; склейки — растворение
# замершего последнего кадра старой сцены (вместе с фоном) в новую сцену за 0,6 с (cubic-bezier(.3,.6,.4,1)).
# Всё в домене декодированного ролика (bt709 tv → rgb); цвета таблиц и тайлы сняты в Chrome — переводим обратной кривой.
import numpy as np, subprocess, sys, json, os
from PIL import Image
S=os.path.dirname(os.path.abspath(__file__)); REPO=os.path.dirname(os.path.dirname(S))
# телефон: квадрат ролика во всю ширину (середина кадра, object-fit:cover), сверху запас под шапку, снизу — под текст и волну
W,H,L,T=464,1440,0,80
import cv2
VW,VH=848,464
T0=80+232-232  # верх квадрата ролика в кадре
N0,N1=int(sys.argv[1]),int(sys.argv[2])     # кадры ролика [N0, N1)
# запуск: python3 <скрипт> N0 N1 out.rgb|-  (кадры rgb24 подряд; '-' — в stdout). Весь ролик — build.sh
OUT=sys.argv[3]                              # raw rgb24 поток
LOOP=os.environ.get('LOOP','1')=='1'         # в начало вшить растворение из последнего кадра (бесшовный повтор)

# --- кривая Chrome (декод → показ) и обратная ---
lut=np.load(f'{S}/chrome-curve.npy'); INV=[]
for c in range(3):
  m=~np.isnan(lut[c]); x=np.arange(256)[m]; y=lut[c][m]
  y=np.maximum.accumulate(y); x=np.r_[0,x,255]; y=np.r_[0,y,255]
  fw=np.interp(np.arange(256),x,y)
  INV.append(lambda v,fw=fw: np.interp(v,fw+np.arange(256)*1e-6,np.arange(256)))
def inv(rgb): rgb=np.asarray(rgb,float); return np.stack([INV[c](rgb[...,c]) for c in range(3)],-1)
def hx(h): h=h.lstrip('#'); return np.array([int(h[i:i+2],16) for i in (0,2,4)],float)
def cv(h): return inv(hx(h))

SC=[[0,25,'#f5f5e1','#f4f4e1'],[4.8,25,'#f5f5e9','#f5f5e9'],[7.37,65,'#f5f5e9','#f5f5e9'],[16.833,77,'#16a063','#16a063',0,'g'],
  [22.333,37,'#f2efe4','#f1eee4'],[30.333,25,'#f5f5e1','#f4f4e1'],[35.333,18,'#f8f7e3','#f8f6e3'],[37.35,18,'#faf8e4','#f8f8e3'],
  [40.333,55,'#f7f5da','#f8f6da'],[46.3,25,'#f5f5e1','#f4f4e1'],[51.3,59,'#dbe9f7','#30638b'],[56.3,52,'#032b5a','#022a5a',0,'b'],
  [61.3,0,'#78a99a','#7aa99a'],[67.267,60,'#afbec7','#afbec7'],[72.267,24,'#e1a599','#e4a799'],[77.267,35,'#f3f3e7','#f1eedf']]
CLIP=[[0,4.8,.153,.846],[51.3,56.3,.2653,.7358],[30.333,35.333,.153,.846],[46.3,51.3,.153,.846],[35.333,35.55,.276,.724],[37.35,40.333,.276,.724],[40.333,46.3,.28,.724],[67.85,71.85,0,.724]]
TX={'g':['hero-tex-gl.webp','hero-tex-gr.webp'],'b':['hero-tex-bl.webp','hero-tex-br.webp']}
BD={51.3:[[[0,'dcebf9'],[.1088,'f2c5c5'],[.5894,'30638b']],[[0,'dcebf8'],[.5929,'30638b']]]}
SW={51.3:[[260,90,266,-.5],[488,60,268,.7]]}
CUT=[4.8,16.833,22.333,30.333,35.333,46.3,51.3,56.3,61.3,67.267,72.267,77.267]
# таблицы — из score.js (бывший heroSync)
src=open(f'{S}/score.js',encoding='utf8').read()
import re
def tbl(name):
  i=src.index(f'const {name}='); j=src.index('};',i); body=src[i+len(name)+7:j+1]
  body=re.sub(r'(\d+\.?\d*):',r'"\1":',body.replace("'",'"')); return {float(k):v for k,v in json.loads(body).items()}
PF=tbl('HERO_PF'); VP=tbl('HERO_VP')
i=src.index('const FISH='); j=src.index('};',i); FISH=json.loads(src[i+11:j+1].replace("src:",'"src":').replace("c:{",'"c":{').replace("'",'"'))['c']
i=src.index('const HERO_WIRE='); j=src.index('};',i); wb=src[i+16:j+1].replace('4/464','0')
wb=re.sub(r'\.\.\.Array\((\d+)\)\.fill\((\[[^\]]*\])\)',lambda m:','.join([m.group(2)]*int(m.group(1))),wb)
wb=re.sub(r'([{,])(\w+):',r'\1"\2":',wb.replace("'",'"')); WIRE=json.loads(wb); WIRE['top']=4/464
TOTAL=2498

ZM=[1.0]   # увеличение ролика (HERO_ZM) для текущей сцены
X=lambda u:W/2+(u-.5)*VW*ZM[0]; Y=lambda v:T+232+(v-.5)*VH*ZM[0]
ZR=[4.8,16.833,.279,.721]
def zoom_of(t): return (464/848)/(ZR[3]-ZR[2]) if ZR[0]<=t<ZR[1] else 1.0
BP={k:[(n,np.array([int(h[i:i+2],16) for i in range(0,96,2)],float).reshape(16,3)) for n,h in v] for k,v in json.load(open(f'{S}/assets/hero-edges.json')).items()}
def key(x): return ('%g'%x)
xs=np.arange(W)+.5; ys=np.arange(H)+.5
def grad(coord,stops):
  # CSS linear-gradient по одной оси: позиции не убывают, до первой — первый цвет, после последней — последний
  pos=[];cols=[];m=-1e18
  for p,c in stops: m=max(m,p);pos.append(m);cols.append(np.asarray(c,float))
  pos=np.array(pos);cols=np.array(cols); out=np.empty((len(coord),)+cols.shape[1:])
  i=np.searchsorted(pos,coord,side='right')
  for k in range(len(coord)):
    j=i[k]
    if j==0: out[k]=cols[0]
    elif j>=len(pos): out[k]=cols[-1]
    else:
      p0,p1=pos[j-1],pos[j]; w=0 if p1<=p0 else (coord[k]-p0)/(p1-p0); out[k]=cols[j-1]*(1-w)+cols[j]*w
  return out
def over(dst,layer,alpha): # alpha: (H,W) или broadcast
  a=alpha[...,None]; return dst*(1-a)+layer*a
def scAt(t):
  s=SC[0]
  for r in SC:
    if t>=r[0]: s=r
  return s
_tile={}
def tile(nm):
  if nm not in _tile: _tile[nm]=inv(np.asarray(Image.open(f'{S}/assets/{nm}').convert('RGB'),float))
  return _tile[nm]
def tiled(img,ox,oy):  # background-repeat с началом тайла в (ox, oy), масштаб 1
  th,tw=img.shape[:2]; yi=(np.floor(ys-oy).astype(int))%th; xi=(np.floor(xs-ox).astype(int))%tw
  return img[yi][:,xi]
_bg={}
def background(sc):
  k=sc[0]
  if k in _bg: return _bg[k]
  z0=ZM[0]; ZM[0]=zoom_of(sc[0])
  pf,bd,vp=PF.get(k),BD.get(k),VP.get(k); tx=TX.get(sc[5]) if len(sc)>5 and sc[5] else None
  yb=T0+W  # низ квадрата ролика (без увеличения), как f.b
  col=grad(ys,[(yb,cv(sc[2])),(yb,cv(sc[3]))])
  img=np.repeat(col[:,None,:],W,1)
  if tx:
    Li=tiled(tile(tx[0]),X(0)-360,Y(0)); Ri=tiled(tile(tx[1]),X(668/VW),Y(0))
    m=grad(xs,[(X(.25),1.),(X(.75),0.)])
    img=Ri; img=over(img,Li,np.broadcast_to(m[None,:],(H,W)))
  elif bd:
    def gb(s):
      st=[]
      for j,(p,c) in enumerate(s):
        if j: st.append((Y(s[j][0]),cv(c)))
        if j<len(s)-1: st.append((Y(s[j+1][0]),cv(c)))
      return np.repeat(grad(ys,st)[:,None,:],W,1)
    img=gb(bd[1]); m=grad(xs,[(X(.25),1.),(X(.75),0.)]); img=over(img,gb(bd[0]),np.broadcast_to(m[None,:],(H,W)))
  elif vp:
    y0,y1=Y(0),Y(1)
    def gr(h): cs=[h[i:i+6] for i in range(0,len(h),6)]; return np.repeat(grad(ys,[(y0+(y1-y0)*j/(len(cs)-1),cv(c)) for j,c in enumerate(cs)])[:,None,:],W,1)
    img=gr(vp[1]); m=grad(xs,[(X(.25),1.),(X(.75),0.)]); img=over(img,gr(vp[0]),np.broadcast_to(m[None,:],(H,W)))
  elif pf:
    y0,y1=Y(0),Y(1); rm=(y1-y0)*.15
    cs=lambda h:[h[i:i+6] for i in range(0,len(h),6)]
    avg=lambda h:''.join('%02x'%int(np.floor(np.mean([int(c[j:j+2],16) for c in cs(h)])+.5)) for j in (0,2,4))
    def gr(h,m): return np.repeat(grad(ys,[(y0-rm,cv(m))]+[(y0+(y1-y0)*j/(len(cs(h))-1),cv(c)) for j,c in enumerate(cs(h))]+[(y1+rm,cv(m))])[:,None,:],W,1)
    Lm,Rm=avg(pf[2]),avg(pf[3]); a,b=X(pf[0]),X(pf[1]); D=X(.2)-X(0)
    img=np.repeat(grad(xs,[(a,cv(Lm)),(b,cv(Rm))])[None,:,:],H,0)
    img=over(img,gr(pf[3],Rm),np.broadcast_to(grad(xs,[(b-D,0.),(b,1.),(b+D,0.)])[None,:],(H,W)))
    img=over(img,gr(pf[2],Lm),np.broadcast_to(grad(xs,[(a-D,0.),(a,1.),(a+D,0.)])[None,:],(H,W)))
  _bg[k]=img.astype(np.float32); ZM[0]=z0; return _bg[k]

def cov1d(n,a,b):  # доля пикселя [i, i+1) внутри [a, b]
  i=np.arange(n); return np.clip(np.minimum(i+1,b)-np.maximum(i,a),0,1)
def shift_rows(strip,dy):  # сдвиг полосы вниз на дробное dy (билинейно, как drawImage), вне — прозрачно
  h=strip.shape[0]; out=np.zeros((h+3,)+strip.shape[1:]); al=np.zeros(h+3)
  fy=dy-np.floor(dy); iy=int(np.floor(dy))
  for r in range(h):
    for rr,w in ((r+iy,1-fy),(r+iy+1,fy)):
      if 0<=rr<h+3: out[rr]+=strip[r]*w; al[rr]+=w
  return out,al,iy  # al — покрытие
def compose(n,frame,t=None):
  t=n/30 if t is None else t; sc=scAt(t); ZM[0]=zoom_of(t); img=background(sc).copy()
  if sc[0] not in CUT and sc[0]>0 and t-sc[0]<.6:
    pv=scAt(sc[0]-.001); w=ez((t-sc[0])/.6); img=background(pv)*(1-w)+img*w
  # низ кадра под роликом (bpSet): 16 цветов нижней строки по ширине кадра, ключевые кадры, между ними — линейно
  if not (len(sc)>5 and sc[5]):
    a_=(ZM[0]>1 and BP.get('z'+key(sc[0]))) or BP.get(key(sc[0]))
    if a_:
      j=0
      while j<len(a_)-2 and a_[j+1][0]<=n: j+=1
      (n0,p0),(n1,p1)=a_[j],a_[min(j+1,len(a_)-1)]; w=min(1,max(0,(n-n0)/(n1-n0))) if n1>n0 else 0
      cs=np.floor(p0+(p1-p0)*w+.5)
      row=grad(xs,[(X((q+.5)/16),inv(cs[q])) for q in range(16)])
      yb=T0+W; cov=np.clip(np.arange(H)+1-yb,0,1)
      img=over(img,np.broadcast_to(row[None],(H,W,3)),np.broadcast_to(cov[:,None],(H,W)))
  # волны моря: полоса горизонта из этого же кадра, повтор до краёв (ext), маска холста по высоте
  sw=SW.get(sc[0])
  if sw:
    c=[r for r in CLIP if r[0]==sc[0]][0]; h=17; ytop=Y(min(a[2]+a[3] for a in sw)/VH); Hc=h+2
    acc=np.zeros((H,W,3)); al=np.zeros((H,W))
    for (x0,w,y0,d),(xa,xb) in ((sw[0],(0,X(c[2])+1)),(sw[1],(X(c[3])-1,W))):
      strip=frame[y0:y0+h,x0:x0+w].astype(float); o=X(x0/VW); Tt=w
      xstart=o-np.ceil((o-xa)/Tt)*Tt
      yy=Y((y0+d)/VH)  # верх полосы в кадре
      sh,sa,_=shift_rows(strip,yy-np.floor(yy)); r0=int(np.floor(yy))
      x=xstart
      while x<xb:
        xi=int(round(x)); xl=max(xi,0); xr=min(xi+w,W)
        if xr>xl:
          rows=slice(r0,r0+sh.shape[0])
          acc[rows,xl:xr]=sh[:,xl-xi:xr-xi]; al[rows,xl:xr]=sa[:,None]
        x+=Tt
    # маска .hero-sw: прозрачно → 1 на 25% → 1 до 75% → прозрачно, по высоте холста H=19 от ytop; холст режет всё ниже
    s=(ys-ytop)/Hc; mk=np.clip(np.minimum(s/.25,(1-s)/.25),0,1); mk[(s<0)|(s>1)]=0
    colr=acc/np.maximum(al[...,None],1e-9)  # цвет полосы без учёта покрытия; покрытие и маска — в прозрачность
    img=over(img,colr,np.clip(al,0,1)*mk[:,None])
  # провод лампы до верха экрана (HERO_WIRE): полоса тона c, мягкие края, по кадрам; заходит под ролик на 3 px
  wi=n-WIRE['n']
  if 0<=wi<len(WIRE['w']):
    cw,ww=WIRE['w'][wi]; sc_=ww/32; x0=X(cw/VW)-32*sc_; u=(xs-x0)/sc_
    sg=lambda q:1/(1+np.exp(-q/4.6)); al=np.clip(sg(u-16)-sg(u-48),0,1); al[(u<-8)|(u>72)]=0
    y1=Y(WIRE['top'])+3; rows=np.clip(y1-np.arange(H),0,1)
    img=over(img,np.broadcast_to(cv(WIRE['c']),(H,W,3)),rows[:,None]*al[None,:])
  # ролик в своей обрезке (края картины), доли пикселя на краях
  c=next((r for r in CLIP if r[0]<=t<r[1]),None); u0,u1=(c[2],c[3]) if c else (0,1)
  # ролик: квадрат W×W с верхом в Y(0) без увеличения; при увеличении — середина кадра крупнее, лишнее срезано по квадрату
  z=ZM[0]; E=W
  M=np.float32([[1/z,0,(.5-W/2)/z+VW/2-.5],[0,1/z,(.5-W/2)/z+VH/2-.5]])
  vid=cv2.warpAffine(frame.astype(np.float32),M,(E,E),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REPLICATE)
  cx=cov1d(E,max(0,X(u0)),min(E,X(u1))); a=np.broadcast_to(cx[None,:],(E,E))
  if WIRE['s']<=t<WIRE['e']: a=a*np.clip(np.arange(E)+T0+1-Y(WIRE['top']),0,1)[:,None]
  reg=img[T0:T0+E,0:E]; img[T0:T0+E,0:E]=over(reg,vid,a)
  for sx,sy,w,h,x,y in FISH.get(str(n),[]):
    cell=sheet()[sy:sy+h,sx:sx+w]; fx,fy=int(round(X(x/VW))),int(round(Y(y/VH)))
    a0,a1,b0,b1=max(fx,0),min(fx+w,W),max(fy,0),min(fy+h,H)
    if a1>a0 and b1>b0:
      cc=cell[b0-fy:b1-fy,a0-fx:a1-fx]; img[b0:b1,a0:a1]=over(img[b0:b1,a0:a1],cc[...,:3],cc[...,3]/255)
  return img
_sh=[]
def sheet():
  if not _sh:
    a=np.asarray(Image.open(f'{S}/assets/hero-fish-cells.webp').convert('RGBA'),float); a[...,:3]=inv(a[...,:3]); _sh.append(a)
  return _sh[0]

def ez(x):  # cubic-bezier(.3,.6,.4,1)
  x=min(max(x,0),1); lo,hi=0.,1.
  for _ in range(40):
    m=(lo+hi)/2; bx=3*(1-m)**2*m*.3+3*(1-m)*m*m*.4+m**3
    if bx<x: lo=m
    else: hi=m
  m=(lo+hi)/2; return 3*(1-m)**2*m*.6+3*(1-m)*m*m*1+m**3

def decode(a,b):
  raw=subprocess.run(['ffmpeg','-v','error','-i',f'{REPO}/site/videos/hero.mp4','-vf',
    f"select='between(n\\,{a}\\,{b-1})',scale=in_color_matrix=bt709:in_range=tv:out_range=pc:flags=accurate_rnd+full_chroma_int+bilinear,format=rgb24",
    '-vsync','0','-f','rawvideo','-'],capture_output=True,check=True).stdout
  return np.frombuffer(raw,np.uint8).reshape(-1,VH,VW,3)
cuts=[c for c in CUT if N0<=round(c*30)+17 and round(c*30)<N1]
lo=min([N0]+[round(c*30)-1 for c in cuts])
fr=decode(lo,N1); F=lambda n:fr[n-lo]
frozen={c:compose(round(c*30)-1,F(round(c*30)-1)) for c in cuts}
LOOP=LOOP and N0<18
endF=compose(TOTAL-1,decode(TOTAL-1,TOTAL)[0]) if LOOP else None
fo=sys.stdout.buffer if OUT=='-' else open(OUT,'wb')
if True:
  for n in range(N0,N1):
    img=compose(n,F(n))
    for c in cuts:
      dt=n/30-c
      if -1e-6<=dt<.6:
        a=1-ez(dt/.6); img=img*(1-a)+frozen[c]*a
    if LOOP and n/30<.6:
      a=1-ez(n/30/.6); img=img*(1-a)+endF*a
    fo.write(np.clip(np.floor(img+.5),0,255).astype(np.uint8).tobytes())
    if n%30==0: print('frame',n,file=sys.stderr,flush=True)
fo.flush()
