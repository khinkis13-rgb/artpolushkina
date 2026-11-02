# Сборщик заставки «всё в одном ролике» из оригинала аниматора: кадр = фон сцены (края картины, продолженные наружу) +
# картина из ролика в своей обрезке + волны моря, провод лампы, рыба; склейки — растворение замершего последнего кадра
# старой сцены (вместе с фоном) в новую за 0,6 с (cubic-bezier(.3,.6,.4,1)), повтор бесшовный.
# Всё в домене декодированного ролика (bt709 → rgb), таблиц из браузера нет — цвета фона сняты с самого ролика.
# Запуск: python3 build.py <цель> N0 N1 <мезонин> > raw rgb24   (цели — TG ниже; весь ролик — build.sh)
import numpy as np,subprocess,sys,os,cv2,json
from score import *
S=os.path.dirname(os.path.abspath(__file__))
cv2.setNumThreads(2)
# Цели: размер кадра W×H, картина (кадр ролика целиком) шириной VW с левым верхним углом в (L, T).
# hi/lo — компьютер (запас фона закрывает экраны до 1920×1080), phone — до 600 px: квадрат середины кадра во всю ширину,
# сверху место под шапку, снизу под текст и волну
TG={'hi':dict(W=3744,H=2304,VW=1280,L=1788,T=732),'lo':dict(W=2808,H=1728,VW=960,L=1341,T=549),
    'phone':dict(W=720,H=2236,VW=1280,L=-280,T=124,phone=1)}
def place(tg,t):
  # (ox, oy, sc): px цели = o + px базового кадра · sc; на телефоне с увеличением (ZM) — вокруг середины квадрата
  g=TG[tg]; sc=g['VW']/BW
  if g.get('phone'):
    z=next(((BW*g['W']/g['VW'])/((r[3]-r[2])*BW) for r in ZM if r[0]<=t<r[1]),1.0)
    if z!=1: sc*=z; w=round(BW*sc); h=round(BH*sc); return (round(g['W']/2-w/2),round(g['T']+g['W']/2-h/2),w/BW,w,h)
  return (g['L'],g['T'],sc,g['VW'],round(BH*sc))

def ez(x):  # cubic-bezier(.3,.6,.4,1)
  x=min(max(x,0),1); lo,hi=0.,1.
  for _ in range(40):
    m=(lo+hi)/2; bx=3*(1-m)**2*m*.3+3*(1-m)*m*m*.4+m**3
    if bx<x: lo=m
    else: hi=m
  m=(lo+hi)/2; return 3*(1-m)**2*m*.6+3*(1-m)*m*m*1+m**3
def scIdx(t):
  i=0
  for j,r in enumerate(SC):
    if t>=r[0]: i=j
  return i

# --- фон сцены: края картины по периметру (assets/borders.npz), наружу — всё мягче с расстоянием ---
BD=np.load(f'{S}/assets/borders.npz')
def loop_levels(i,C=None,V=None):
  C=BD[f'c{i}'].astype(np.float64) if C is None else C; V=BD[f'v{i}'] if V is None else V; P=len(C); idx=np.arange(P)
  if not V.all():  # пропуски (рисунок у края) — по кругу от соседей; у самого рисунка край в тени и полутонах — 6 px долой
    bad=~V
    for s in range(1,7): bad=bad|np.roll(~V,s)|np.roll(~V,-s)
    V=~bad; vi=idx[V]
    # опоры для пропусков — медиана 31 соседней годной точки, чтобы длинный пропуск не тянулся к одной случайной
    Cv=C[V]; Cs=np.median(np.stack([np.roll(Cv,s,0) for s in range(-15,16)]),0)
    F=np.stack([np.interp(idx,np.r_[vi-P,vi,vi+P],np.tile(Cs[:,c],3)) for c in range(3)],1)
    C=np.where(V[:,None],C,F)
  # медиана 9 по кругу — убрать зерно и одиночные выбросы, не смазывая резкие границы (горизонт)
  C=np.median(np.stack([np.roll(C,s,0) for s in range(-4,5)]),0)
  # каждая сторона размывается сама по себе (концы продолжены своим крайним цветом): дальний фон над картиной —
  # из её верхнего края, а не из моря у нижнего
  x0,y0,x1,y1=BD[f'r{i}']; w,h=int(x1-x0),int(y1-y0); cut=[0,w,w+h,2*w+h,P]
  sig=[0.]+[2*1.25**j for j in range(60) if 2*1.25**j<P*2]; PAD=12000
  lev=np.zeros((len(sig),P,3),np.float32)
  for k in range(4):
    seg=C[cut[k]:cut[k+1]]; n=len(seg); x=np.concatenate([np.repeat(seg[:1],PAD,0),seg,np.repeat(seg[-1:],PAD,0)])
    F=np.fft.rfft(x,axis=0); fr=np.fft.rfftfreq(len(x))
    for j,sg in enumerate(sig): lev[j,cut[k]:cut[k+1]]=np.fft.irfft(F*np.exp(-2*(np.pi*fr*sg)**2)[:,None],n=len(x),axis=0)[PAD:PAD+n]
  return lev,np.array(sig,np.float32)
def Vof(i,tg):  # годные точки края для цели: hole (лист на ровном фоне) на компьютере — пропуск, его заполнит фон кадра
  V=BD[f'v{i}'].copy(); hl=BG[i].get('hole')
  if hl and (not TG[tg].get('phone') or BG[i].get('hole_phone')):
    x0,y0,x1,y1=[int(v) for v in BD[f'r{i}']]; w,h=x1-x0,y1-y0; x=np.arange(x0,x1); m=(x>=hl[0]*BW)&(x<=hl[1]*BW)
    V[:w]&=~m; V[w+h:2*w+h]&=~m[::-1]
  return V
_ed={}
def edges(i):  # строки резких переходов по левой и правой сторонам (px базового кадра)
  if i not in _ed:
    C=BD[f'c{i}'].astype(np.float32); x0,y0,x1,y1=BD[f'r{i}']; w,h=int(x1-x0),int(y1-y0)
    f=lambda s: y0+np.nonzero(np.abs(np.diff(s.mean(1)))>12)[0]+.5
    _ed[i]=(f(C[2*w+h:][::-1]),f(C[w:w+h]))
  return _ed[i]
_lv={}
def background(i,tg,t,lv=None,step=1):
  g=BG[i]; ox,oy,sc,_,_=place(tg,t); W,H=TG[tg]['W'],TG[tg]['H']
  x0,y0,x1,y1=[float(v) for v in BD[f'r{i}']]; w,h=x1-x0,y1-y0; P=2*(w+h)
  if lv is None:
    k=(i,bool(TG[tg].get('phone')))
    if k not in _lv: _lv[k]=loop_levels(i,None,Vof(i,tg))
    lv=_lv[k]
  lev,sig=lv; assert lev.shape[1]==int(P)
  bx=((np.arange(0,W,step,dtype=np.float32)+step/2-ox)/sc)[None,:]; by=((np.arange(0,H,step,dtype=np.float32)+step/2-oy)/sc)[:,None]
  bx,by=np.broadcast_arrays(bx,by)
  qx=np.clip(bx,x0,x1); qy=np.clip(by,y0,y1); dx=np.abs(bx-qx); dy=np.abs(by-qy)
  # внутри картины (растушёвка края) — ближайшая сторона без размытия
  inn=(dx==0)&(dy==0); ldx=np.minimum(bx-x0,x1-bx); ldy=np.minimum(by-y0,y1-by); hor=inn&(ldx<ldy)
  # слева/справа — продолжение строк боковой стороны, сверху/снизу — столбцов верхней/нижней; в углах — смесь обоих
  # по направлению (вес бока = dx/(dx+dy)). Размытие растёт с расстоянием: kh, kv — от своей стороны, kc — за её концом
  kh,kv,kc=g.get('k',(.6,.6,.6))
  e=1.001  # индекс — строго внутри своей стороны (стороны размыты порознь)
  s0=10 if g.get('tex') else 2   # с фактурой фон — только крупный тон, мелкое даёт плитка
  sH=np.where(bx<(x0+x1)/2,2*w+h+np.clip(y1-qy,0,h-e),w+np.clip(qy-y0,0,h-e)); gH=s0+kh*dx+kc*dy
  if g.get('edges'):  # слабое размытие сбоку (kh) — только у резких границ (горизонт, берег); мазки между ними — как обычно,
    EL,ER=edges(i)   # но не шире половины расстояния до границы, чтобы её не размыть
    yq=qy[:,0]; dL=np.abs(yq[:,None]-EL[None,:]).min(1); dR=np.abs(yq[:,None]-ER[None,:]).min(1)
    de=np.where(bx<(x0+x1)/2,dL[:,None],dR[:,None])
    gH=s0+np.minimum(.6*dx,np.maximum(kh*dx,de/2.5))+kc*dy
  sV=np.where(by<(y0+y1)/2,np.clip(qx-x0,0,w-e),w+h+np.clip(x1-qx,0,w-e)); gV=s0+kv*dy+kc*dx
  wH=np.where(inn,hor.astype(np.float32),dx/np.maximum(dx+dy,1e-6)).astype(np.float32)
  def look(s,sg):
    s=np.clip(s,0,P-1.001).astype(np.float32)
    li=np.clip(np.log(np.maximum(sg,2)/2)/np.log(1.25)+1,0,len(sig)-1.001); l0=li.astype(np.int32); lf=(li-l0)[...,None]
    s0=s.astype(np.int32); sf=(s-s0)[...,None]; s1=np.minimum(s0+1,int(P)-1)
    return (lev[l0,s0]*(1-sf)+lev[l0,s1]*sf)*(1-lf)+(lev[l0+1,s0]*(1-sf)+lev[l0+1,s1]*sf)*lf
  out=look(sH,gH)*wH[...,None]+look(sV,gV)*(1-wH[...,None])
  return out.astype(np.float32)

# --- фактура холста (сцены с tex): плитки из texture.py, в фазе с кадром; слева плитка левого края, справа — правого ---
TMP=os.environ.get('HERO_TMP','')
_tx={}
def texOf(i,tg,t):
  ox,oy,sc,pw,ph=place(tg,t); k=(i,tg,ox,oy)
  if k in _tx: return _tx[k]
  z=np.load(f'{TMP}/tex{i}.npz'); W,H=TG[tg]['W'],TG[tg]['H']
  bx0,by0=round(-ox/sc),round(-oy/sc); Wb,Hb=round(W/sc),round(H/sc); xs=np.arange(Wb)+bx0
  wL=np.clip((BW*.75-xs)/(BW*.5),0,1)[None,:,None].astype(np.float32)
  L,R=z['L'].astype(np.float32),z['R'].astype(np.float32)
  iL=(xs-int(z['L0'][0]))%L.shape[1]; iR=(xs-int(z['R0'][0]))%R.shape[1]
  out=np.zeros((H,W,3),np.float32); CH=round(6/sc)*12   # полосами строк, кратными шагу уменьшения
  for r0 in range(0,Hb,CH):
    ys=np.arange(r0,min(r0+CH,Hb))+by0
    hf=L[(ys%L.shape[0])[:,None],iL[None,:]]*wL+R[(ys%R.shape[0])[:,None],iR[None,:]]*(1-wL)
    a,b=round(r0*sc),round(min(r0+CH,Hb)*sc)
    out[a:b]=cv2.resize(hf,(W,b-a),interpolation=cv2.INTER_AREA)
  # стык плитки у краёв кадра — внутри кадра (24 столбца): там кадр виден только по свою сторону линии стыка
  m=np.ones((BH,BW),np.float32); ov=z['Lm'].shape[1]; m[:,:ov]=1-z['Lm'].astype(np.float32); m[:,BW-ov:]=z['Rm'].astype(np.float32)
  _tx[k]=(out,cv2.resize(m,(pw,ph),interpolation=cv2.INTER_AREA))
  return _tx[k]

# Тон бумаги внутри сцены плывёт (свет, экспозиция: в сцене 7,37 низ кадра от 232 до 241) — фон следует за ним:
# полоса края в этом кадре минус снятая (годные точки — цвета бумаги), сглаженная вдоль края (сигма 60 px), продолжается
# наружу так же, как фон; считается в 1/8 разрешения и растягивается
def band(i,frame):  # отличие полосы у края картины в кадре от снятой (по периметру) и где это бумага/стена
  g=BG[i]; x0,y0,x1,y1=[int(v) for v in BD[f'r{i}']]; C=BD[f'c{i}']
  D0,D1=3,13; f=frame.astype(np.float32)
  T=f[y0+D0:y0+D1,x0:x1].mean(0); R=f[y0:y1,x1-D1:x1-D0].mean(1); B=f[y1-D1:y1-D0,x0:x1].mean(0)[::-1]; L=f[y0:y1,x0+D0:x0+D1].mean(1)[::-1]
  b=np.concatenate([T,R,B,L]); ok=BD[f'v{i}'].copy()
  if g['ref'] is not None: ok&=np.min([np.abs(b-np.array(c,np.float32)).max(1) for c in g['ref']],0)<TOL
  return np.clip(b-C,-15,15),ok
_dr=None
def drift(i,tg,t,n):
  # тон бумаги в ролике плывёт: поправка фона = полоса края кадра минус снятая, медиана по ±3 кадрам сцены
  # (рисунок, проходящий у края, и его полутона не дают одиночных вспышек), сглажено не мельче сигмы 60
  global _dr
  if not BG[i].get('drift',1): return 0.
  if _dr is None: _dr=np.load(os.path.join(os.environ['HERO_TMP'],'drift.npz'))
  ms=[m for m in range(n-3,n+4) if 0<=m<TOTAL and scIdx(m/30+1e-3)==i]
  D=np.stack([_dr[f'd{m}'].astype(np.float32) for m in ms]); O=np.stack([_dr[f'o{m}'] for m in ms])
  ok=(O.sum(0)*2>=len(ms))&Vof(i,tg)
  if ok.mean()<.2: return 0.
  dl=np.nanmedian(np.where(O[...,None],D,np.nan)[:,ok],0); d=np.zeros(D.shape[1:]); d[ok]=dl
  ok&=np.abs(d-np.median(dl,0)).max(1)<6  # полутона рисунка у края — не сдвиг тона бумаги
  w=np.clip((ok.mean()-.2)/.2,0,1)  # мало бумаги у края (картина проявляется) — поправка входит плавно
  if w==0: return 0.
  L_,S_=loop_levels(i,d.astype(np.float64),ok); j=int(np.searchsorted(S_,60)); L_=L_[np.maximum(np.arange(len(S_)),j)]
  W,H=TG[tg]['W'],TG[tg]['H']; st=8
  d=background(i,tg,t,(L_,S_),st)*w
  return cv2.resize(d,(W,H),interpolation=cv2.INTER_LINEAR)

_bg={}
def bgOf(i,tg,t):
  ox,oy,sc,_,_=place(tg,t); k=(i,ox,oy)
  if k not in _bg:
    if len(_bg)>3: _bg.pop(next(iter(_bg)))
    _bg[k]=background(i,tg,t)
    if BG[i].get('tex'): _bg[k]+=texOf(i,tg,t)[0]
  return _bg[k]

def cov1d(n,a,b):  # доля пикселя [i, i+1) внутри [a, b]
  i=np.arange(n,dtype=np.float32); return np.clip(np.minimum(i+1,b)-np.maximum(i,a),0,1)
FE=2.5  # растушёвка края картины, px цели
def compose(tg,n,frame):
  t=n/30+1e-3  # склейка 67,267 = кадр 2018,01 — кадр 2018 уже новая сцена
  i=scIdx(t); img=bgOf(i,tg,t)+drift(i,tg,t,n)
  if SC[i][0] not in CUT and i>0 and t-SC[i][0]<.6:
    w=ez((t-SC[i][0])/.6); img=bgOf(i-1,tg,SC[i][0]-.001)*(1-w)+img*w
  W,H=TG[tg]['W'],TG[tg]['H']; ox,oy,sc,pw,ph=place(tg,t)
  if i in WAVES: waves(img,tg,t,frame,WAVES[i])
  if BG[i]['rect'][1]>0 and WIRE['frames'][0]<=n<WIRE['frames'][1]: wire(img,tg,t,frame)
  if str(n) in FISH: fish(img,tg,t,FISH[str(n)])
  vid=cv2.resize(frame,(pw,ph),interpolation=cv2.INTER_AREA).astype(np.float32)
  c=next((r for r in CLIP if r[0]<=t<r[1]),None); u0,u1=(c[2],c[3]) if c else (0,1)
  top=BG[i]['rect'][1]*sc if c else 0
  a0,a1=u0*pw,u1*pw
  ax=cov1d(pw,a0,a1)*np.clip(np.minimum(np.arange(pw)+.5-a0,a1-np.arange(pw)-.5)/FE+.5,0,1)
  ay=cov1d(ph,top,ph)*np.clip(np.minimum(np.arange(ph)+.5-top,ph-np.arange(ph)-.5)/FE+.5,0,1)
  a=(ay[:,None]*ax[None,:])
  if BG[i].get('tex') and not c: a=a*texOf(i,tg,t)[1]
  a=a[...,None]
  # окно картины в кадре цели
  X0,Y0=max(ox,0),max(oy,0); X1,Y1=min(ox+pw,W),min(oy+ph,H)
  sl=(slice(Y0,Y1),slice(X0,X1)); vs=(slice(Y0-oy,Y1-oy),slice(X0-ox,X1-ox))
  img[sl]=img[sl]*(1-a[vs])+vid[vs]*a[vs]
  return img

def waves(img,tg,t,frame,wv):
  # полоса горизонта в px базового кадра на всю ширину цели: повторы полосы L влево от картины, R — вправо
  ox,oy,sc,_,_=place(tg,t); W=TG[tg]['W']; bx0=round(-ox/sc); Wb=round(W/sc)
  r0,r1=wv['rows']; band=np.zeros((r1-r0,Wb,3),np.float32); al=np.zeros((r1-r0,Wb),np.float32)
  c=next(r for r in CLIP if r[0]<=t<r[1]); eL,eR=c[2]*BW,c[3]*BW
  for side,(xa,xb,dy,ph) in (('L',wv['L']),('R',wv['R'])):
    w=xb-xa; st=frame[r0-8:r1+8,xa:xb].astype(np.float32)
    st=cv2.warpAffine(st,np.float32([[1,0,0],[0,1,dy]]),(w,st.shape[0]),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)[8:-8]
    x=xa+ph-w*int(np.ceil((xa+ph-bx0)/w)) if side=='L' else xa+ph
    while x<bx0+Wb:
      a0,a1=max(x,bx0),min(x+w,bx0+Wb)
      lim=(a0,min(a1,int(np.ceil(eL))+1)) if side=='L' else (max(a0,int(eR)-1),a1)
      if lim[1]>lim[0]: band[:,lim[0]-bx0:lim[1]-bx0]=st[:,lim[0]-x:lim[1]-x]; al[:,lim[0]-bx0:lim[1]-bx0]=1
      x+=w
  rr=np.arange(r0,r1)+.5; al*=np.clip(np.minimum(rr-r0,r1-rr)/10,0,1)[:,None]
  y0=round((r0-round(-oy/sc))*sc); h=round((r1-r0)*sc)
  bd=cv2.resize(band*al[...,None],(W,h),interpolation=cv2.INTER_AREA); ad=cv2.resize(al,(W,h),interpolation=cv2.INTER_AREA)
  img[y0:y0+h]=img[y0:y0+h]*(1-ad[...,None])+bd
def wire(img,tg,t,frame):
  # провод: верхние строки кадра у провода → вверх до края экрана; прозрачность — насколько темнее стены
  ox,oy,sc,pw,ph=place(tg,t); x0,x1=WIRE['x']; r0,r1=WIRE['rows']
  row=np.median(frame[r0:r1,x0:x1].astype(np.float32),0); l=row.mean(1); bg=np.median(l); core=l.min()
  a=np.clip((bg-l)/max(bg-core,1)*1.15-.05,0,1); col=row[np.argmin(l)]
  # в px цели: столбцы провода, строки от верха до верха картины (+3 под картину)
  xs=ox+(np.arange(x0,x1)+.5)*sc; X=np.arange(img.shape[1])+.5
  at=np.interp(X,xs,a,left=0,right=0).astype(np.float32)
  yb=int(oy+BG[13]['rect'][1]*sc)+3
  if yb>0: img[:yb]=img[:yb]*(1-at[None,:,None])+col[None,None,:]*at[None,:,None]
_fs={}
def fish(img,tg,t,items):
  ox,oy,sc,_,_=place(tg,t); H,W=img.shape[:2]
  for it in items:
    if len(it)<5: continue
    nm,x,y,g,op=it
    if nm not in _fs:
      p=cv2.cvtColor(cv2.imread(f'{S}/assets/fish-{nm}.png',cv2.IMREAD_UNCHANGED),cv2.COLOR_BGRA2RGBA).astype(np.float32); _fs[nm]=p
    p=_fs[nm].copy(); h,w=p.shape[:2]
    # поворот вокруг середины спрайта, затем — в масштаб и место цели (доли px — сдвигом)
    p[...,:3]*=p[...,3:]/255
    M=cv2.getRotationMatrix2D((w/2,h/2),g,1); p=cv2.warpAffine(p,M,(w,h),flags=cv2.INTER_LINEAR,borderValue=0)
    tw,th=max(1,round(w*sc)),max(1,round(h*sc)); p=cv2.resize(p,(tw,th),interpolation=cv2.INTER_AREA)
    X,Y=ox+x*sc,oy+y*sc; xi,yi=int(np.floor(X)),int(np.floor(Y))
    p=cv2.warpAffine(p,np.float32([[1,0,X-xi],[0,1,Y-yi]]),(tw+1,th+1),flags=cv2.INTER_LINEAR,borderValue=0)
    a0,b0,a1,b1=max(xi,0),max(yi,0),min(xi+tw+1,W),min(yi+th+1,H)
    if a1<=a0 or b1<=b0: continue
    q=p[b0-yi:b1-yi,a0-xi:a1-xi]; a=q[...,3:]/255*op
    img[b0:b1,a0:a1]=img[b0:b1,a0:a1]*(1-a)+q[...,:3]*op
FISH=json.load(open(f'{S}/assets/fish.json')) if os.path.exists(f'{S}/assets/fish.json') else {}

def frames(mz,a,b):
  p=subprocess.Popen(['ffmpeg','-v','error','-ss','%.5f'%(a/30),'-i',mz,'-vf','format=rgb24','-frames:v',str(b-a),'-vsync','0',
    '-f','rawvideo','-'],stdout=subprocess.PIPE)
  sz=BW*BH*3
  for n in range(a,b):
    buf=p.stdout.read(sz)
    yield n,np.frombuffer(buf,np.uint8).reshape(BH,BW,3)
  p.wait()
def one(mz,n): return next(frames(mz,SUBST.get(n,n),SUBST.get(n,n)+1))[1]

if __name__=='__main__' and sys.argv[1]=='bands':  # build.py bands <мезонин>: полосы краёв всех кадров для drift
  out={}
  for n,fr in frames(sys.argv[2],0,TOTAL):
    if n in SUBST: fr=one(sys.argv[2],SUBST[n])
    d,o=band(scIdx(n/30+1e-3),fr); out[f'd{n}']=d.astype(np.float16); out[f'o{n}']=o
  np.savez(os.path.join(os.environ['HERO_TMP'],'drift.npz'),**out)
elif __name__=='__main__':
  tg,N0,N1,mz=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),sys.argv[4]
  LOOP=os.environ.get('LOOP','1')=='1'
  cuts=[c for c in CUT if N0<=round(c*30)+17 and round(c*30)<N1]
  frozen={c:compose(tg,round(c*30)-1,one(mz,round(c*30)-1)) for c in cuts}
  endF=compose(tg,TOTAL-1,one(mz,TOTAL-1)) if LOOP and N0<18 else None
  fo=sys.stdout.buffer; prev=None
  for n,fr in frames(mz,N0,N1):
    if n in SUBST: fr=one(mz,SUBST[n])
    img=compose(tg,n,fr)
    for c in cuts:
      dt=(n-round(c*30))/30
      if 0<=dt<.6: a=1-ez(dt/.6); img=img*(1-a)+frozen[c]*a
    if endF is not None and n/30<.6: a=1-ez(n/30/.6); img=img*(1-a)+endF*a
    fo.write(np.clip(img+.5,0,255).astype(np.uint8).tobytes())
    if n%30==0: print(tg,'кадр',n,file=sys.stderr,flush=True)
  fo.flush()
