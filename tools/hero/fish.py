# Рыба на переходе 37,35 → 40,333: в ролике она уплывает за правый край картины 18 и выплывает из-за левого края
# картины 55, а за краем её нет (стена). Здесь за краем дорисовываем её саму: спрайт из кадра, где она целиком внутри,
# положение — по видимой части в каждом кадре (подбор сдвига; у уплывающей — по кончику хвоста), дальше — с той же скоростью. Уплывшая гаснет за 10 кадров.
# → assets/fish-exit.png, fish-enter.png (RGBA, px базового кадра) и assets/fish.json {кадр: [[спрайт, x, y, прозрачность]]}
# Запуск: python3 fish.py <мезонин>  (--fix вторым — только путь после 1195 и чистка спрайтов по готовым assets)
import numpy as np,cv2,json,sys,os
from score import *
import build
S=os.path.dirname(os.path.abspath(__file__)); MZ=sys.argv[1]
def fr(a,b): return np.stack([f for _,f in build.frames(MZ,a,b)]).astype(np.float32)
def sprite(f,plate,box):
  x0,y0,x1,y1=box; d=np.abs(f-plate).max(2)[y0:y1,x0:x1]
  # рыба — самый крупный кусок отличий от фона (без раздутия: иначе к ней прилипают соседние штрихи)
  n,lab,st,_=cv2.connectedComponentsWithStats((d>30).astype(np.uint8)); k=1+np.argmax(st[1:,cv2.CC_STAT_AREA])
  m=cv2.dilate((lab==k).astype(np.uint8),np.ones((5,5),np.uint8))>0
  a=np.clip((d-8)/40,0,1)*cv2.GaussianBlur(m.astype(np.float32),(0,0),1)
  a[a<.02]=0
  pl=plate[y0:y1,x0:x1]; fc=f[y0:y1,x0:x1]
  col=np.where(a[...,None]>.05,(fc-(1-a[...,None])*pl)/np.maximum(a[...,None],.05),fc)
  ys,xs=np.nonzero(a>0); c=(slice(ys.min(),ys.max()+1),slice(xs.min(),xs.max()+1))
  return np.dstack([np.clip(col,0,255),a*255])[c].astype(np.float32),x0+xs.min(),y0+ys.min()
def rot(spr,ang):
  if ang==0: return spr
  h,w=spr.shape[:2]; M=cv2.getRotationMatrix2D((w/2,h/2),ang,1)
  p=spr.copy(); p[...,:3]*=p[...,3:]/255
  r=cv2.warpAffine(p,M,(w,h),flags=cv2.INTER_LINEAR,borderValue=0)
  r[...,:3]/=np.maximum(r[...,3:]/255,1e-3); return r
def track(frames,n0,plate,spr0,sx,sy,edge,side,v0):
  # подбор сдвига спрайта по видимой части у самого края (полоса 90 px внутри картины) — там стык с дорисованной;
  # где рыбы у края не видно — по скорости
  # спрайт с запасом по краям, чтобы поворот не срезал
  P=30; spr0=np.pad(spr0,((P,P),(P,P),(0,0))); sx-=P; sy-=P
  h,w=spr0.shape[:2]; out={}; x,y=sx-v0,sy; vx,vy=v0,0.; ang=0
  R={g:rot(spr0,g) for g in range(-12,13,2)}
  e0=int(edge)
  for k,f in enumerate(frames):
    n=n0+k; best=None
    st=(slice(None),slice(e0-90,e0-3)) if side=='R' else (slice(None),slice(e0+3,e0+90))
    seen=(np.abs(f[st]-plate[st]).max(2)>30).sum()
    if seen>300:
      for g in [g for g in R if abs(g-ang)<=4]:
       spr=R[g]; a=spr[...,3:]/255
       for dy in range(-5,6):
        for dx in range(-14,15):
          X,Y=int(round(x+vx))+dx,int(round(y+vy))+dy
          cols=np.arange(X,X+w); ins=((cols<e0-3)&(cols>=e0-90)) if side=='R' else ((cols>e0+3)&(cols<=e0+90))
          if ins.sum()<20: continue
          comp=plate[Y:Y+h,X:X+w]*(1-a)+spr[...,:3]*a; d=((comp-f[Y:Y+h,X:X+w])**2).sum(2)
          # сравнение по рыбе спрайта и по рыбе кадра (иначе выгодно уехать с неё)
          fm=(a[...,0]>.3)|(np.abs(f[Y:Y+h,X:X+w]-plate[Y:Y+h,X:X+w]).max(2)>30)
          e=d[:,ins][fm[:,ins]].mean() if fm[:,ins].any() else 1e9
          if best is None or e<best[0]: best=(e,X,Y,g)
    if best:
      nx,ny=best[1],best[2]
      if k: vx,vy=.5*vx+.5*(nx-x),.5*vy+.5*(ny-y)
      x,y=nx,ny; ang=best[3]; out[n]=(x,y,ang,round(float(best[0]),1))
    else:
      x,y=x+vx,y+vy; out[n]=(round(x,2),round(y,2),ang,None)
  return out
def unline(sp,xr,yr,ys_fit):
  # линия картины, пересекающая рыбу (почти вертикальная): её ход — по тёмным точкам вне тела, полоса ±3 px
  # поперёк тела заполняется соседями: цвет — inpaint, прозрачность — по столбцам слева и справа от полосы
  a=sp[...,3]/255; lum=sp[...,:3].mean(2); d=(a>.5)&(lum<110)
  P=[(y,np.nonzero(d[y,xr[0]:xr[1]])[0].mean()+xr[0]) for y in ys_fit if d[y,xr[0]:xr[1]].any()]
  P=np.array(P); k=np.polyfit(P[:,0],P[:,1],2)
  h,w=a.shape; Y,X=np.mgrid[:h,:w]; xf=np.polyval(k,Y)
  sp=sp.copy()
  # светлый ореол линии левее неё — тоже долой; полоса по строке — линейно между столбцами по краям (цвет и прозрачность)
  wl=((sp[...,:3].mean(2)>200)&(sp[...,3]>128)&(X>xf-16)&(X<xf-3)&(Y>=yr[0])&(Y<yr[1]))
  for y in range(yr[0],yr[1]):
    c=int(round(xf[y,0])); l=c-4
    q=np.nonzero(wl[y,:c])[0]
    if len(q): l=min(l,q.min()-1)
    r=c+4; sp[y,l:r+1]=np.linspace(sp[y,l-1],sp[y,r+1],r-l+3)[1:-1]
  return sp
def clean(sp,line=None):
  if line: sp=unline(sp,*line)
  a=sp[...,3]/255; c=sp[...,:3]; lum=c.mean(2); sat=(c.max(2)-c.min(2))/np.maximum(c.max(2),1)
  keep=(a>.05)&~((lum>195)&(sat<.25))
  op=cv2.morphologyEx(keep.astype(np.uint8),cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
  n,lab,st,_=cv2.connectedComponentsWithStats(op); op=(lab==1+np.argmax(st[1:,4])).astype(np.uint8)
  gold=(a>.5)&(c[...,0]-c[...,2]>70)&(sat>.4); bl=np.nonzero(gold.any(0))[0].min()
  xs=np.arange(sp.shape[1])[None,:]
  near=np.where(xs<bl+6,cv2.dilate(op,np.ones((9,9),np.uint8)),cv2.dilate(op,np.ones((5,5),np.uint8)))>0
  m=(keep&near).astype(np.uint8)
  n,lab,st,_=cv2.connectedComponentsWithStats(m); m=(lab==1+np.argmax(st[1:,4])).astype(np.uint8)
  h,w=m.shape; ff=m.copy(); mk=np.zeros((h+2,w+2),np.uint8); cv2.floodFill(ff,mk,(0,0),2); m=(ff!=2).astype(np.float32)
  # внутри тела прозрачности нет: дырки (где рыба совпала с фоном кадра) — непрозрачные, цвет от соседей
  inner=cv2.erode(m.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))>0; inner&=xs>bl+20; hole=inner&(sp[...,3]<250)   # у хвоста светлые кусочки мозаики — не трогать
  sp=sp.copy(); sp[...,:3]=cv2.inpaint(np.clip(sp[...,:3],0,255).astype(np.uint8),cv2.dilate(hole.astype(np.uint8),np.ones((3,3),np.uint8)),3,cv2.INPAINT_TELEA)
  sp[...,3][inner]=255
  out=sp.copy(); out[...,3]=np.clip(sp[...,3]*cv2.GaussianBlur(m,(0,0),.6),0,255); out[...,3][m==0]=0
  return out
# линия картины 55 поперёк головы выплывающей рыбы (px спрайта с полями): столбцы, строки поперёк тела, строки для хода линии
LINE={'enter':((300,360),(90,150),list(range(56,96))+list(range(152,212)))}
def tailx(plate,a,b):
  # у правого края картины 18 видна хвостовая часть уплывающей рыбы: левый край её самого крупного куска (кончик хвоста)
  out={}
  for k,f in enumerate(fr(a,b)):
    m=(np.abs(f-plate).max(2)[200:460,1230:1393]>45).astype(np.uint8)
    n,lab,st,_=cv2.connectedComponentsWithStats(m)
    if n>1 and st[1:,4].max()>500: out[a+k]=1230+st[1+np.argmax(st[1:,4]),0]
  return out
def exitpath(trE,tail):
  # пока хвост виден — по нему (сдвиг спрайта к кончику хвоста — по 1194–1195, где слежение ещё надёжно), дальше —
  # с последней скоростью, чуть замедляясь (в ролике рыба у края тормозит); наклон пути — по слежению 1186–1195
  off=np.mean([trE[n][0]-tail[n] for n in (1194,1195)]); sl=(trE[1195][1]-trE[1186][1])/(trE[1195][0]-trE[1186][0])
  x,y=trE[1195][:2]; v=0.
  for n in range(1196,1217):
    nx=tail[n]+off if n in tail else x+max(8.,v-.5)
    v=nx-x; y+=sl*v; x=nx; trE[n]=(round(float(x),2),round(float(y),2),-4,None)
  return trE
def refine(trE,plate,sp,a,b):
  # у края рыба в ролике уходит вниз: сдвиг спрайта по совпадению его маски с видимой частью (окно у края картины 18),
  # пока видно хотя бы 20 столбцов; дальше — сдвиг растёт с той же скоростью (прямая по последним 4 кадрам)
  h,w=sp.shape[:2]; W=(slice(200,460),slice(1230,1390)); got={}
  for k,f in enumerate(fr(a,b)):
    n=a+k; vm=(np.abs(f-plate).max(2)>45)[W]
    if vm.any(0).sum()<20: continue
    x,y,g,_=trE[n]; best=None
    for dy in range(-6,31):
      for dx in range(-8,9):
        M=cv2.getRotationMatrix2D((w/2,h/2),g,1); M[:,2]+=[x+dx-1230,y+dy-200]
        sm=cv2.warpAffine(sp[...,3],M,(160,260),flags=cv2.INTER_LINEAR)>128
        e=(sm^vm).sum()/max((sm|vm).sum(),1)
        if best is None or e<best[0]: best=(e,dx,dy)
    got[n]=best[1:]
  ns=sorted(got); print('сдвиг у края',{n:got[n] for n in ns})
  fx=np.polyfit(ns[-4:],[got[n][0] for n in ns[-4:]],1); fy=np.polyfit(ns[-4:],[got[n][1] for n in ns[-4:]],1)
  for n in range(ns[0],1217):
    dx,dy=got[n] if n in got else (np.polyval(fx,n),np.polyval(fy,n))
    x,y,g,_=trE[n]; trE[n]=(round(float(x+dx),2),round(float(y+dy),2),g,None)
  return trE
if len(sys.argv)>2 and sys.argv[2]=='--fix':
  # пересчитать путь уплывающей после 1195 и почистить спрайты, не повторяя слежение (assets уже есть)
  J=json.load(open(f'{S}/assets/fish.json')); p7=np.median(fr(1125,1205)[::4],0)
  trE={int(n):tuple(e[1:4])+(None,) for n,v in J.items() for e in v if e[0]=='exit'}
  for nm in ('exit','enter'):
    sp=cv2.cvtColor(cv2.imread(f'{S}/assets/fish-{nm}.png',cv2.IMREAD_UNCHANGED),cv2.COLOR_BGRA2RGBA).astype(np.float32)
    sp=clean(sp,LINE.get(nm))
    cv2.imwrite(f'{S}/assets/fish-{nm}.png',cv2.cvtColor(np.clip(sp+.5,0,255).astype(np.uint8),cv2.COLOR_RGBA2BGRA))
  trE=exitpath(trE,tailx(p7,1194,1206))
  spx=cv2.cvtColor(cv2.imread(f'{S}/assets/fish-exit.png',cv2.IMREAD_UNCHANGED),cv2.COLOR_BGRA2RGBA).astype(np.float32)
  trE=refine(trE,p7,spx,1196,1206)
  for n,(x,y,g,_) in trE.items():
    if n>=1186:
      for e in J[str(n)]:
        if e[0]=='exit': e[1:4]=[x,y,g]
  json.dump(J,open(f'{S}/assets/fish.json','w'))
  for n in range(1192,1218): print(n,J.get(str(n)))
  sys.exit()
p7=np.median(fr(1125,1205)[::4],0); p8=np.median(fr(1215,1385)[::6],0)
# уплывает: целиком в кадре 1183, слежение 1183–1216, край картины 18 справа
f=fr(1183,1184)[0]; spE,ex,ey=sprite(f,p7,(990,360,1340,560))
trE=track(fr(1183,1196),1183,p7,spE,ex,ey,CLIP[3][3]*BW,'R',19)
trE=exitpath(trE,tailx(p7,1194,1206)); trE=refine(trE,p7,clean(np.pad(spE,((30,30),(30,30),(0,0)))),1196,1206)
# выплывает: целиком в кадре 1237, слежение назад 1237→1205, край картины 55 слева
f=fr(1237,1238)[0]; spN,nx,ny=sprite(f,p8,(530,360,880,560))
frs=fr(1205,1238)[::-1]; trN=track(frs,0,p8,spN,nx,ny,CLIP[4][2]*BW,'L',-13)
trN={1237-k:v for k,v in trN.items()}
J={}
# спрайт хранится с полями 30 px (под поворот); x, y — его левый верх в px базового кадра, g — поворот, градусы
for n,(x,y,g,e) in trE.items():
  if n>=1186: J.setdefault(n,[]).append(['exit',x,y,g,round(float(np.clip((1216-n)/10,0,1)),3)])
for n,(x,y,g,e) in trN.items():
  if 1210<=n<=1237: J.setdefault(n,[]).append(['enter',x,y,g,round(float(np.clip((n-1209)/4,0,1)),3)])
for nm,sp in (('exit',spE),('enter',spN)):
  sp=np.pad(sp,((30,30),(30,30),(0,0)))
  sp=clean(sp,LINE.get(nm))  # бумага, куски соседнего рисунка, дырки в теле — долой
  cv2.imwrite(f'{S}/assets/fish-{nm}.png',cv2.cvtColor(np.clip(sp+.5,0,255).astype(np.uint8),cv2.COLOR_RGBA2BGRA))
json.dump({str(k):[[a,float(x),float(y),g,al] for a,x,y,g,al in v] for k,v in sorted(J.items())},open(f'{S}/assets/fish.json','w'))
for n in sorted(set(trE)|set(trN)): print(n,trE.get(n),trN.get(n))
