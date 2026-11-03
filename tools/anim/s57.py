# «Поплыли!» (работа 57). Ветер поднимает волны на линиях из-под шляпы, линии вытягиваются до края экрана, лодки
# уплывают за край, из-под шляпы выплывают новые (сначала крошечные) и встают на свои места, волны стихают.
import numpy as np, cv2
from lib import *

SRC='site/images/work-57.jpg'
LINE_RGB=np.array([74,78,72],np.float32)
BG=(172,196,176)                                          # фон сцены — шалфейный (у картины кремовая бумага)
BGC=np.array([243,241,218],np.float32)
# затравки линий (x, y) сверху вниз: по обе стороны лодок
SEEDS=[[(370,219),(600,236),(790,208)],[(370,244),(600,268),(790,248)],[(370,272),(600,298),(790,284)],
       [(370,297),(600,340),(790,340)],[(370,333),(600,391),(790,378)]]
# лодки: парус, мачта, корпус — многоугольники в px картины 900×784
BOATS=[dict(sail=[(468,264),(484,359),(439,344)],mast=[(437,340),(443,340),(441,349),(436,349)],
            hull=[(391,328),(485,363),(463,379),(398,354)]),
       dict(sail=[(697,129),(758,224),(702,227)],mast=[(700,224),(706,224),(707,238),(702,238)],
            hull=[(651,239),(756,231),(741,253),(666,258)])]

def boat_mask(H,W,b,grow=0):
    m=np.zeros((H,W),np.uint8)
    for k in ('sail','mast','hull'): cv2.fillPoly(m,[np.array(b[k],np.int32)],255)
    if grow: m=cv2.dilate(m,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*grow+1,2*grow+1)))
    return m>0

def track(a,nb,dark,boats,stop):
    g=a.mean(2); H,W=g.shape; out=[]
    for seeds in SEEDS:
        tr={}
        for x0,y0 in seeds:
            for dirn in (1,-1):
                y=float(y0); x=x0; miss=0
                while 2<x<W-3:
                    lo,hi=int(round(y))-4,int(round(y))+5
                    if boats[lo:hi,x].any(): break                        # лодка — дальше другая затравка
                    if dirn<0 and stop[lo-2:hi+2,x-2:x].any(): break      # упёрлись в шляпу/лицо
                    col=g[lo:hi,x]; ys=np.arange(lo,hi)
                    w=np.clip(160-col,0,None); w[~dark[lo:hi,x]]=0
                    if w.sum()>40: y=(w*ys).sum()/w.sum(); tr[x]=y; miss=0
                    else:
                        miss+=1
                        if miss>4: break
                    x+=dirn
        xs=np.array(sorted(tr)); ys=np.array([tr[x] for x in xs])
        xf=np.arange(xs[0],xs[-1]+1); yf=np.interp(xf,xs,ys)               # за лодками — по прямой между краями
        yf=cv2.GaussianBlur(yf.reshape(1,-1).astype(np.float64),(0,0),4).ravel()
        out.append((xf,yf))
    return out

def layers():
    """картина → фон-тон, женщина с подписью (альфа), линии (x, y центра), лодки (спрайты)"""
    a=load_work(SRC); H,W,_=a.shape
    nb=np.abs(a-BGC).sum(2)>55
    dark=(a.mean(2)<165)&(np.ptp(a,2)<40)
    boats=np.zeros((H,W),bool); skip=np.zeros((H,W),bool)
    for b in BOATS: boats|=boat_mask(H,W,b,2); skip|=boat_mask(H,W,b,6)
    # цветные пятна (шляпа, лицо, платье) — линии в них упираются
    n,lab,st,_=cv2.connectedComponentsWithStats((nb&~dark&~skip).astype(np.uint8))
    stop=np.isin(lab,[i for i in range(1,n) if st[i,4]>2000])
    stop=cv2.dilate(stop.astype(np.uint8),np.ones((5,5),np.uint8))>0
    L=track(a,nb,dark,skip,stop)
    lm=np.zeros((H,W),np.uint8)
    for xf,yf in L: cv2.polylines(lm,[np.round(np.stack([xf,yf],1)*16).astype(np.int32)],False,255,7,cv2.LINE_8,4)
    lm=lm>0
    # женщина: связная область не-фона без линий и лодок; подпись остаётся на месте вместе с ней
    wm=cv2.morphologyEx((nb&~lm&~boats).astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    n,lab=cv2.connectedComponents(wm); wm=lab==lab[600,200]
    wm[715:780,700:890]|=nb[715:780,700:890]
    core=cv2.dilate(wm.astype(np.uint8),np.ones((3,3),np.uint8))>0
    d=np.abs(a-BGC).sum(2)
    al=np.clip((d-12)/60,0,1)*core; al[cv2.erode(wm.astype(np.uint8),np.ones((3,3),np.uint8))>0]=1
    # фон под рисунком: залить — по нему меряем тон бумаги и свет фото
    hole=cv2.dilate((nb|lm|boats).astype(np.uint8)*255,np.ones((7,7),np.uint8))
    bg=cv2.inpaint(np.clip(a,0,255).astype(np.uint8),hole,9,cv2.INPAINT_TELEA).astype(np.float32)
    tone=np.median(bg.reshape(-1,3),0); lit=tone-cv2.GaussianBlur(bg,(0,0),40)
    a=a+lit                                                   # свет фото снят: всё к ровному тону бумаги
    bs=[]
    for b in BOATS:
        inner=boat_mask(H,W,b,0); outer=boat_mask(H,W,b,2)&~(lm&~inner)
        # сплошная только сердцевина (многоугольник чуть шире рисунка); край — по удалению от бумаги, без её остатков
        core=cv2.erode(inner.astype(np.uint8),np.ones((5,5),np.uint8))>0
        ba=np.where(core,1.,np.clip((d-25)/70,0,1)*outer).astype(np.float32)
        hull=np.array(b['hull'],np.float32)
        bs.append(Sprite.cut(clean(a,ba,tone),ba,tuple((hull[2]+hull[3])/2),0))   # опора — середина низа корпуса
    return dict(a=clean(a,al,tone),wa=al.astype(np.float32),tone=tone,lines=L,boats=bs)

def clean(a,al,tone):
    """края вырезанного куска без старой бумаги: цвет = (пиксель − (1−α)·тон)/α"""
    al=al[...,None]; return np.where(al>.04,np.clip((a-(1-al)*tone)/np.maximum(al,.04),0,255),a).astype(np.float32)

class Scene:
    T=9.0; work=57
    def __init__(s,geo):
        ly=layers(); h,w=ly['a'].shape[:2]
        s.cv=cv=Canvas(geo,h,w); s.tone=np.float32(BG); s.plate=cv.flat(s.tone)
        s.xr=cv.x_right+60                                    # правый край экрана в px картины (+запас)
        # женщина и подпись — поверх всего; платье обрывается краем картины, как в заставке
        s.top=cv.warp(ly['a']*ly['wa'][...,None]); s.topa=np.clip(cv.warp(ly['wa']),0,1)
        # линии: исходная форма + продолжение вправо, плавно уходящее в горизонталь
        s.L=[]
        for xf,yf in ly['lines']:
            sl=(yf[-1]-yf[-15])/14
            xe=np.arange(xf[-1]+1,s.xr+1); ye=yf[-1]+sl*40*(1-np.exp(-(xe-xf[-1])/40))
            s.L.append(dict(x=np.concatenate([[xf[0]-8],xf,xe]),y=np.concatenate([[yf[0]],yf,ye]),x0=xf[0],xe=xf[-1]))
        # лодки: на какой линии, смещение опоры от линии, исходный наклон линии
        s.boats=ly['boats']
        for b in s.boats:
            ax,ay=b.anc; best=None
            for i,l in enumerate(s.L):
                d=ay-np.interp(ax,l['x'],l['y'])
                if best is None or abs(d)<abs(best[1]): best=(i,d)
            b.li,b.dy=best; b.sl0=s.slope(best[0],ax,None)

    def wave(s,i,x,t):
        """смещение линии i в точке x (px картины) от ветра в момент t"""
        l=s.L[i]; A=win(t,.3,2.2,7.0,9.6)
        g=ease((x-l['x0'])/180)*(1+0.35*ease((x-900)/600))
        return A*g*(7*np.sin(2*np.pi*(x/260-t/2.2)+i*1.1)+2.5*np.sin(2*np.pi*(x/97+t/1.3)+i*2.3))

    def ly(s,i,x,t):
        l=s.L[i]; y=np.interp(x,l['x'],l['y']); return y if t is None else y+s.wave(i,x,t)
    def slope(s,i,x,t): return (s.ly(i,x+8,t)-s.ly(i,x-8,t))/16

    def boat_path(s,bi,t):
        """где лодка bi в момент t: [(x опоры, масштаб, доворот°)] — уходящая и новая из-под шляпы"""
        b=s.boats[bi]; ax=b.anc[0]; l=s.L[b.li]; out=[]
        t0,t1,n0,n1=[(1.8,5.8,5.2,8.4),(1.3,5.0,4.6,7.8)][bi]
        bob=2.2*np.sin(2*np.pi*t/1.7+bi*1.9)*win(t,.3,2.2,7.0,9.6)
        if t<t1:
            u=(t-t0)/(t1-t0); x=ax+(s.xr+120-ax)*(ein(u)*.45+np.clip(u,0,1)*.55) if t>t0 else ax
            out.append((x,1.,bob))
        if t>n0:
            u=(t-n0)/(n1-n0); xs=l['x0']-40
            out.append((xs+(ax-xs)*eout(u),.3+.7*eout(u),bob*eout(u)-8*(1-eout(u))))
        return out

    def frame(s,t):
        cv=s.cv; img=s.plate.copy()
        ext=ramp(t,.5,2.8)                                    # линии вытягиваются до края экрана и так и остаются
        polys=[]
        for i,l in enumerate(s.L):
            x=l['x'][l['x']<=l['xe']+ext*(s.xr-l['xe'])][::2]; polys.append(np.stack([x,s.ly(i,x,t)],1))
        a=lines_alpha(cv,polys,3.1); img=img*(1-a[...,None])+LINE_RGB*a[...,None]
        for bi,b in enumerate(s.boats):
            for x,sc,rot in s.boat_path(bi,t):
                ang=np.degrees(np.arctan(s.slope(b.li,x,t))-np.arctan(b.sl0))+rot
                img=put(cv,img,b,x,s.ly(b.li,x,t)+b.dy,sc,-ang)
        img=over(img,s.top,s.topa)
        return np.clip(img+.5,0,255).astype(np.uint8)

if __name__=='__main__': cli(Scene,'s57')
