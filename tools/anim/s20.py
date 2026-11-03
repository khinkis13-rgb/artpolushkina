# «YOU CAN LEAVE YOUR HAT ON» (работа 20). Море целиком нарисовано заново по контурам картины — плоские полосы с
# контуром на весь экран (внутри картины, под фигурой и до краёв экрана), из фото — только фигура, шляпа и подпись.
# Волны плавно ходят, женщина покачивается. Порыв ветра срывает шляпу — она кувыркаясь улетает за правый край экрана,
# под ней открывается голая макушка; потом шляпа, качаясь как лист, падает сверху и садится обратно на голову
# («шляпу можно оставить»). Движение — с первого кадра до последнего (сборщик сливает сцены без пауз).
import numpy as np, cv2
from lib import *

SRC='site/images/work-20.jpg'
BG=(244,232,190)                                           # небо сцены: светлый лимонно-кремовый (море — как в картине)
# классы цвета (до снятия света): небо, шляпа, кожа, тёмно-синий, голубой, серо-синий, контур
PAL=dict(sky=(236,233,223),hat=(159,86,79),skin=(221,203,195),navy=(30,70,105),lb=(105,155,176),st=(67,108,128),
         line=(10,35,55))
SKY,HAT,SKIN,NAVY,LB,ST,LINE=range(7)
LINE_RGB=np.array([16,34,44],np.float32); LW=2.0           # контур: цвет и толщина, px картины
WAIST=(345.,600.)                                          # опора покачивания женщины
HAT_ANC=(321.,262.)                                        # опора шляпы (середина полей)
HEAD=dict(cx=325.5,cy=313.,r=25.,yj=346,lw=1.5,ink=.62)   # голова: круг, стык с линиями шеи на y=yj
# полосы моря в картине от дальней к ближней: точка внутри каждой
BANDS=[('navyB',(100,600)),('lbB',(650,550)),('stM',(340,650)),('lbL',(150,740)),('navyR',(650,720)),('stL',(150,850)),
       ('lbF',(550,860))]
TORSO=(288,412,556,606)                                    # здесь край фигуры — это верх передней волны
SIGN=(566,672,850,890)                                     # подпись художницы (x0, x1, y0, y1)

def light(a):
    """свет фото: аддитивный многочлен 3-й степени + постоянный цвет каждого класса; → (lit, классы)"""
    H,W,_=a.shape; C=np.float32([PAL[k] for k in PAL])
    yy,xx=np.mgrid[0:H,0:W].astype(np.float32); u=xx/W-.5; v=yy/H-.5
    B=np.stack([np.ones_like(u),u,v,u*u,u*v,v*v,u**3,v**3,u*u*v,u*v*v],-1).reshape(-1,10)
    lit=np.zeros((H,W,3),np.float32)
    for it in range(3):
        c=a-lit; cl=((c[:,:,None,:]-C[None,None])**2).sum(-1).argmin(-1)
        core=np.zeros((H,W),bool)
        for i in range(6):
            m=cv2.erode((cl==i).astype(np.uint8),np.ones((7,7),np.uint8))>0; core|=m
            if it==0 and m.sum()>500: C[i]=np.median(c[m],0)
        sel=core.ravel(); coef,*_=np.linalg.lstsq(B[sel],(a-C[cl]).reshape(-1,3)[sel],rcond=None)
        lit=(B@coef).reshape(H,W,3).astype(np.float32)
    c=a-lit; cl=((c[:,:,None,:]-C[None,None])**2).sum(-1).argmin(-1)
    return lit,cl

def flatten(c,cl):
    """остаток света внутри каждой заливки: плавное отклонение от её среднего (и полосы у обреза фото) — убрать"""
    H,W,_=c.shape; comp=np.zeros((H,W),np.int32); means=[np.zeros(3,np.float32)]; n0=1
    for k in range(6):
        n,lab,st,_=cv2.connectedComponentsWithStats((cl==k).astype(np.uint8),connectivity=4)
        for i in range(1,n):
            if st[i,4]<800: continue
            m=lab==i; comp[m]=n0; means.append(med(c,cv2.erode(m.astype(np.uint8),disk(2))>0)); n0+=1
    means=np.float32(means); core=cv2.erode((comp>0).astype(np.uint8),disk(2))>0
    for k in range(1,n0): core&=~((comp==k)&~(cv2.erode((comp==k).astype(np.uint8),disk(2))>0))
    def resid(): return np.where(core[...,None],c-means[comp],0)
    w=cv2.GaussianBlur(core.astype(np.float32),(0,0),20)
    dev=cv2.GaussianBlur(resid(),(0,0),20)/np.maximum(w,1e-3)[...,None]
    c=c-dev*np.clip(w/.05,0,1)[...,None]
    r=resid(); yy,xx=np.mgrid[0:H,0:W]
    for dist in (xx,W-1-xx,yy,H-1-yy):                      # полоса тени/засвета у обреза фото
        for d in range(16):
            m=(dist==d); mc=m&core
            if mc.sum()>50: c[m]-=np.median(r[mc],0)
    return c

def biggest(m):
    n,lab,st,_=cv2.connectedComponentsWithStats(m.astype(np.uint8))
    return lab==(1+np.argmax(st[1:,4]))

def fill_holes(m):
    f=m.astype(np.uint8).copy(); h,w=f.shape; mm=np.zeros((h+2,w+2),np.uint8)
    cv2.floodFill(f,mm,(0,0),2); return (f!=2)

def disk(r): return cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1))

def unmix(L,ref,dark=30.):
    """альфа контура: насколько пиксель темнее соседнего фона (к цвету линии)"""
    return np.clip((ref-L)/np.maximum(ref-dark,12),0,1)

def med(c,m):
    v=c[m]; lo,hi=np.percentile(v.mean(1),[3,97]); k=(v.mean(1)>=lo)&(v.mean(1)<=hi); return v[k].mean(0)

def layers():
    a=load_work(SRC); H,W,_=a.shape
    a[:,0:5]=a[:,9:4:-1]                                     # крайние столбцы фото — засвет/тень обреза
    lit,cl=layers.cache if hasattr(layers,'cache') else light(a); layers.cache=(lit,cl)
    c=flatten(a-lit,cl); L=c.mean(2)
    tone=med(c,cv2.erode((cl==SKY).astype(np.uint8),disk(4))>0)
    yy,xx=np.mgrid[0:H,0:W]
    # шляпа: самое большое красное пятно, без пикселей контура у лица
    hat=cv2.morphologyEx((cl==HAT).astype(np.uint8),cv2.MORPH_OPEN,np.ones((3,3),np.uint8))>0
    hat&=~((yy>311)&(xx>294)&(xx<356)); hat=fill_holes(biggest(hat))
    # женщина: кожа (с дырами под линиями груди)
    wm=cv2.morphologyEx((cl==SKIN).astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))>0
    wm=fill_holes(biggest(wm))
    Lb=L.copy(); Lb[hat|wm]=0; ref=cv2.dilate(Lb,np.ones((9,9),np.uint8))
    hring=(cv2.dilate(hat.astype(np.uint8),disk(3))>0)&~hat
    wring=(cv2.dilate(wm.astype(np.uint8),disk(3))>0)&~wm&~hat
    um=np.clip((unmix(L,ref)-.15)/.85,0,1)                 # слабый край — это ореол резкости фото, не контур
    ha=np.where(hat,1.,um*hring).astype(np.float32)
    wa=np.where(wm,1.,um*wring).astype(np.float32)
    hrgb=np.where(hat[...,None],c,LINE_RGB); wrgb=np.where(wm[...,None],c,LINE_RGB)
    # острые кончики рук в фото — одни сходящиеся линии: дорисовать клин кожи с контуром до острия
    S=4; km=np.zeros((H*S,W*S),np.uint8); kl=np.zeros_like(km)
    for up,lo,tip in (((128,555.),(128,561.5),(43,612)),((588,502.5),(588,507.5),(656,527))):
        P=np.int32(np.round(np.float32([up,tip,lo])*S)); cv2.fillPoly(km,[P],255,cv2.LINE_AA)
        for e in (up,lo): cv2.line(kl,tuple(np.int32(np.float32(e)*S)),tuple(np.int32(np.float32(tip)*S)),255,int(1.7*S),cv2.LINE_AA)
    km=cv2.resize(km,(W,H),interpolation=cv2.INTER_AREA)/255.; kl=cv2.resize(kl,(W,H),interpolation=cv2.INTER_AREA)/255.*.9
    skin0=med(c,cv2.erode(wm.astype(np.uint8),disk(3))>0); nk=~wm
    wrgb=np.where((nk&(km>0))[...,None],skin0,wrgb); wa=np.where(nk,np.maximum(wa,km),wa)
    kl=np.where(nk|(kl>.3),kl,0); wrgb=wrgb*(1-kl[...,None])+LINE_RGB*kl[...,None]; wa=np.maximum(wa,kl).astype(np.float32)
    # волны перед торсом (по ним обрезан низ фигуры): голубой и серо-синий у низа торса, с контуром
    box=(xx>=276)&(xx<=424)&(yy>=560)&(yy<=720)
    F=(cv2.dilate(np.isin(cl,[LB,ST]).astype(np.uint8),disk(3))>0)&box&~(wm&(yy<585))
    F=(np.maximum.accumulate(F.astype(np.uint8),0)>0)&box               # всё ниже переднего контура
    F=cv2.GaussianBlur(F.astype(np.float32),(0,0),.7)*box
    # торс продолжается вниз под волну (виден, когда волна опускается)
    skin=med(c,cv2.erode(wm.astype(np.uint8),disk(3))>0)
    xf=np.arange(W,dtype=np.float32)
    for y in range(560,645):                                # бока торса — по прямым от видимых кусков
        xl=293+.18*(y-560); xr=397+.15*(y-560)
        dl,dr=np.abs(xf-xl),np.abs(xf-xr)
        inside=np.clip(np.minimum(xf-xl,xr-xf)+.5,0,1); ln=np.clip(1.6-np.minimum(dl,dr),0,1)
        al=np.maximum(inside,ln); rgb=skin*(1-ln[:,None])+LINE_RGB*ln[:,None]
        put_=(~wm[y])&(al>0)&(xf>270)&(xf<430)
        wrgb[y,put_]=rgb[put_]; wa[y,put_]=al[put_]
    # голова целиком заново: круг по видимой нижней части лица, ниже — гладко в линии шеи из фото
    skin=med(c,cv2.erode(wm.astype(np.uint8),disk(3))>0)
    cx,cy,r=HEAD['cx'],HEAD['cy'],HEAD['r']; yj=HEAD['yj']; yc=cy+r*.62     # круг до yc, стык с шеей на yj
    def neck(y):                                            # середины линий шеи в строке y (по краю кожи)
        sk=np.flatnonzero(wm[y,290:362])+290; return sk[0]-.8,sk[-1]+.8
    nl=np.array([neck(y) for y in range(yj,yj+9)]); sl=np.polyfit(np.arange(9),nl[:,0],1)[0]; sr=np.polyfit(np.arange(9),nl[:,1],1)[0]
    ys_=np.arange(cy-r,yj+.01,.25); hw=np.sqrt(np.maximum(r*r-(ys_-cy)**2,0))
    xl_,xr_=cx-hw,cx+hw
    dc=(yc-cy)/np.sqrt(r*r-(yc-cy)**2)                     # dx/dy круга в точке стыка (правая сторона)
    tt=ys_>yc; hl0,hr0=cx-np.sqrt(r*r-(yc-cy)**2),cx+np.sqrt(r*r-(yc-cy)**2)
    xl_[tt]=hermite(yc,hl0,dc,yj,nl[0,0],sl,ys_[tt]); xr_[tt]=hermite(yc,hr0,-dc,yj,nl[0,1],sr,ys_[tt])
    S=4; hm=np.zeros((H*S,W*S),np.uint8); hl=np.zeros_like(hm)
    poly=np.concatenate([np.stack([xl_,ys_],1),np.stack([xr_,ys_],1)[::-1]])
    cv2.fillPoly(hm,[np.int32(np.round(poly*S))],255,cv2.LINE_AA)
    cv2.polylines(hl,[np.int32(np.round(poly[:-1]*S))[1:] if False else np.int32(np.round(np.concatenate([np.stack([xl_,ys_],1)[::-1],np.stack([xr_,ys_],1)[1:]])*S))],
                  False,255,int(round(HEAD['lw']*S)),cv2.LINE_AA)
    hm=cv2.resize(hm,(W,H),interpolation=cv2.INTER_AREA)/255.; hl=cv2.resize(hl,(W,H),interpolation=cv2.INTER_AREA)/255.*HEAD['ink']
    head=(yy<yj)&(xx>=290)&(xx<=362)&(yy>=280)              # фото головы (под полями и неровное) — убрать
    wa[head]=0; wrgb[head]=skin
    dome=hm>0; wrgb[dome]=skin; wa[dome]=np.maximum(wa[dome],hm[dome])
    hl[yy>=yj+1]=0
    wrgb=wrgb*(1-hl[...,None])+LINE_RGB*hl[...,None]; wa=np.maximum(wa,hl).astype(np.float32)
    # подпись: светлые штрихи на ближней волне
    x0,x1,y0,y1=SIGN; cs_=cv2.GaussianBlur(c,(0,0),1.)[y0:y1,x0:x1]; ch=cs_[...,2]-cs_[...,0]; bgc=np.median(ch)
    sa=np.zeros((H,W),np.float32); sa[y0:y1,x0:x1]=np.clip((bgc-ch-14)/18,0,1)    # серые штрихи на голубом
    sa=cv2.morphologyEx(sa,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
    # фигура и шляпа без зерна холста (иначе ролик тяжелеет)
    sm=np.clip(c+.5,0,255).astype(np.uint8)
    for _ in range(2): sm=cv2.bilateralFilter(sm,7,20,4)
    sm=sm.astype(np.float32)
    wrgb=np.where(wm[...,None]&~(dome|(hl>0))[...,None],sm,wrgb); hrgb=np.where(hat[...,None],sm,hrgb)
    bands=trace_bands(c,cl,wm,hat)
    return dict(c=c,cl=cl,tone=tone,hat=(hrgb,ha),woman=(wrgb,wa),sign=(cv2.GaussianBlur(c,(0,0),1.),sa),bands=bands)

def band_masks(cl):
    """области полос: компоненты классов моря; дальняя тёмно-синяя — вместе с клочком под правой рукой"""
    H,W=cl.shape; out={}
    comps={}
    for k in (NAVY,LB,ST):
        m=cv2.morphologyEx((cl==k).astype(np.uint8),cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
        comps[k]=cv2.connectedComponentsWithStats(m,connectivity=4)
    for name,(x,y) in BANDS:
        k=cl[y,x]; n,lab,st,_=comps[k]; m=lab==lab[y,x]
        if name=='navyB':
            for i in range(1,n):
                if st[i,4]>200 and st[i,0]<480 and 500<st[i,1]<620: m|=lab==i
        out[name]=m
    return out

def hermite(xa,ya,sa,xb,yb,sb,x):
    t=(x-xa)/(xb-xa); h=xb-xa
    return (2*t**3-3*t**2+1)*ya+(t**3-2*t**2+t)*h*sa+(-2*t**3+3*t**2)*yb+(t**3-t**2)*h*sb

def trace_bands(c,cl,wm,hat):
    """верх каждой полосы y(x) по столбцам картины: середина контура над ней; под рукой — дуга между краями"""
    H,W=cl.shape; ms=band_masks(cl); anyb=np.zeros((H,W),bool)
    for m in ms.values(): anyb|=m
    known=anyb|wm|hat|(cl==SKY)
    tx0,tx1,ty0,ty1=TORSO; out=[]
    for name,_ in BANDS:
        m=ms[name]; ys=np.full(W,np.nan)
        for x in range(W):
            col=np.flatnonzero(m[:,x])
            if not len(col): continue
            y1=col[0]; y=y1-1
            while y>=0 and not known[y,x]: y-=1
            if y<0 or y1-y>9: continue
            if (wm[y,x] or hat[y,x]) and not (tx0<=x<=tx1 and ty0<=y<=ty1): continue
            ys[x]=(y+y1)/2.
        v=np.flatnonzero(~np.isnan(ys)); xa,xb=v[0],v[-1]
        # дыры (под рукой, за торсом) — гладкой дугой с наклонами краёв
        i=xa
        while i<xb:
            if np.isnan(ys[i+1]):
                j=i+1
                while np.isnan(ys[j]): j+=1
                l=v[(v<=i)&(v>i-14)]; r=v[(v>=j)&(v<j+14)]
                sl=np.polyfit(l,ys[l],1)[0] if len(l)>3 else 0.; sr=np.polyfit(r,ys[r],1)[0] if len(r)>3 else 0.
                xx_=np.arange(i+1,j); ys[i+1:j]=hermite(i,ys[i],sl,j,ys[j],sr,xx_); i=j
            else: i+=1
        seg=ys[xa:xb+1]; pad=np.pad(seg,6,mode='edge')
        seg=np.convolve(pad,np.exp(-np.arange(-6,7)**2/(2*1.6**2))/np.exp(-np.arange(-6,7)**2/(2*1.6**2)).sum(),'valid')
        ys[xa:xb+1]=seg
        col_=med(c,cv2.erode(m.astype(np.uint8),disk(3))>0)
        out.append(dict(name=name,y=ys,xa=int(xa),xb=int(xb),col=col_))
    return out

# ---------- море за картиной: слои волн (каждый — область ниже кривой, рисуем от дальних к ближним) ----------
def ext(d,y0,s,waves,tau=220.):
    """продолжение кривой от края картины на расстояние d ≥ 0: та же точка и наклон, дальше — пологие валы"""
    d=np.maximum(d,0); r=(1-np.exp(-d/200))**2
    return y0+s*tau*(1-np.exp(-d/tau))+r*sum(A*np.sin(2*np.pi*d/lam+phi) for A,lam,phi in waves)

def edge_slope(y,xs_):
    return np.polyfit(xs_,y[xs_],1)[0]

def curves(xs,bands,ybot):
    """все кривые y(x) на сетке xs (px картины) от дальней к ближней: внутри картины — обведённые контуры, у краёв
    картины — продолжение до края экрана, за концом полосы (острый хвост) — уходит под ближние волны"""
    Wp=698; out=[]
    WAV={'navyB':[(14,1400,2.0),(8,520,.7)],'lbL':[(38,1500,1.0),(16,640,2.9)],'stL':[(45,1250,2.2),(15,560,.3)],
         'lbB':[(12,1300,.5),(7,480,1.2)],'navyR':[(40,1400,2.6),(14,600,.1)],'lbF':[(42,1200,4.0),(15,520,2.2)]}
    WAVL={'lbF':[(42,1350,.4),(14,600,1.9)]}
    for b in bands:
        y=np.full(xs.shape,np.nan,np.float32); xi=np.arange(len(b['y']))
        ins=(xs>=b['xa'])&(xs<=b['xb']); y[ins]=np.interp(xs[ins],xi,np.nan_to_num(b['y']))
        if b['xa']<=1 or b['name']=='lbF':                   # продолжение влево (ближняя — от низа картины)
            x0=b['xa']+(2 if b['name']=='lbF' else 0); s_=-edge_slope(b['y'],np.arange(x0,x0+25))
            L=xs<x0; y[L]=ext(x0-xs[L],b['y'][x0],s_,WAVL.get(b['name'],WAV[b['name']]))
        if b['xb']>=Wp-1:
            s_=edge_slope(b['y'],np.arange(Wp-25,Wp+1)); R=xs>Wp; y[R]=ext(xs[R]-Wp,b['y'][Wp],s_,WAV[b['name']])
        out.append([y,b['col'],b])
    # новые валы под картиной до низа экрана: длинные, разной высоты — где-то перехлёстывают соседа
    cs={b['name']:b['col'] for b in bands}
    pal=[(cs['navyB']+cs['navyR'])/2,(cs['stM']+cs['stL'])/2,(cs['lbL']+cs['lbF'])/2]
    yF=out[-1][0]; rng=np.random.RandomState(20); base=990.; j=0; near=(xs>-300)&(xs<Wp+300); inp=(xs>-30)&(xs<Wp+30)
    while base<ybot+220:
        A1,l1_,p1=rng.uniform(40,65),rng.uniform(1100,1900),rng.uniform(0,6.3)
        A2,l2_,p2=rng.uniform(14,26),rng.uniform(450,750),rng.uniform(0,6.3)
        y=A1*np.sin(2*np.pi*xs/l1_+p1)+A2*np.sin(2*np.pi*xs/l2_+p2)+rng.uniform(-.03,.03)*xs
        if j==0: base=max(base,(np.where(near,yF+60,-1e9)-y).max())
        base=max(base,(960.-y)[inp].max())
        y=(y+base).astype(np.float32); out.append([y,pal[j%3],None]); base+=rng.uniform(115,140); j+=1
    # хвосты: за концом полосы — плавно под огибающую ближних (рисуются позже) на 8 px
    env=np.full(xs.shape,1e6,np.float32)
    for k in range(len(out)-1,-1,-1):
        y=out[k][0]; nan=np.isnan(y)
        if nan.any():
            dive=env+8; ok=np.flatnonzero(~nan); xa,xb=ok[0],ok[-1]
            d=np.where(np.arange(len(y))<xa,xa-np.arange(len(y)),np.arange(len(y))-xb).astype(np.float32)
            yend=np.where(np.arange(len(y))<xa,y[xa],y[xb]); r=ease(d/10)
            y=np.where(nan,yend*(1-r)+np.maximum(dive,yend)*r,y).astype(np.float32); out[k][0]=y
        env=np.minimum(env,y)
    return [(y,c_) for y,c_,_ in out]

def render_front(cv,cur,xs,k0):
    """доля пикселя, закрытая полосами с номера k0 и ближе (что стоит перед торсом)"""
    X=np.arange(cv.W,dtype=np.float32); xp=(X-cv.ox)/cv.k; Y=np.arange(cv.H,dtype=np.float32)[:,None]
    m=np.zeros((cv.H,cv.W),np.float32)
    for k,(y,_) in enumerate(cur):
        Yc=(cv.oy+cv.k*np.interp(xp,xs,y)).astype(np.float32)[None]; f=np.clip(Y-Yc+.5,0,1)
        m=m*(1-f)+f*(k>=k0)
    return m

def render_sea(cv,cur,xs,tone):
    img=cv.flat(tone); X=np.arange(cv.W,dtype=np.float32); xp=(X-cv.ox)/cv.k
    lw=max(LW*cv.k,1.0); dark=min(LW*cv.k,1.0)
    y_min=min(cv.oy+cv.k*y.min() for y,_ in cur); r0=max(int(y_min)-4,0)
    Y=np.arange(r0,cv.H,dtype=np.float32)[:,None]; sub=img[r0:]
    for y,colr in cur:
        Yc=(cv.oy+cv.k*np.interp(xp,xs,y)).astype(np.float32)[None]
        sl=np.gradient(Yc[0]); cs=1/np.sqrt(1+sl*sl)
        d=Y-Yc; f=np.clip(d+.5,0,1)[...,None]; sub[:]=sub*(1-f)+colr*f
        la=(np.clip(lw/2-np.abs(d)*cs+.5,0,1)*dark)[...,None]; sub[:]=sub*(1-la)+LINE_RGB*la
    return img

# ---------- спрайты в px холста ----------
def csprite(cv,rgb,a,anc):
    pm=cv.warp(np.dstack([rgb*a[...,None],a])); al=pm[...,3]
    ys,xs=np.where(al>.003); x0,x1,y0,y1=xs.min()-2,xs.max()+3,ys.min()-2,ys.max()+3
    ax,ay=cv.xy(*anc); return dict(pm=pm[y0:y1,x0:x1].copy(),anc=(ax-x0,ay-y0))

def place(img,sp,cx,cy,ang=0.,sx=1.,sy=1.,occl=None):
    """спрайт опорой в (cx, cy) px холста; ang° против часовой; sx, sy — масштаб по осям спрайта; occl — что закрывает"""
    ax,ay=sp['anc']; r=np.radians(ang); c,s=np.cos(r),np.sin(r)
    A=np.float32([[c,s],[-s,c]])@np.float32([[sx,0],[0,sy]])
    M=np.hstack([A,(np.float32([cx,cy])-A@np.float32([ax,ay]))[:,None]])
    ph,pw=sp['pm'].shape[:2]
    cs=np.array([[0,0,1],[pw,0,1],[0,ph,1],[pw,ph,1]],np.float32)@M.T
    x0,y0=np.floor(cs.min(0)).astype(int)-2; x1,y1=np.ceil(cs.max(0)).astype(int)+2
    H,W=img.shape[:2]; x0,y0=max(x0,0),max(y0,0); x1,y1=min(x1,W),min(y1,H)
    if x1<=x0 or y1<=y0: return img
    M2=M.copy(); M2[:,2]-=[x0,y0]
    w=cv2.warpAffine(sp['pm'],M2,(x1-x0,y1-y0),flags=cv2.INTER_LINEAR,borderValue=0)
    if occl is not None: w=w*(1-occl[y0:y1,x0:x1,None])
    a=np.clip(w[...,3:4],0,1); img[y0:y1,x0:x1]=img[y0:y1,x0:x1]*(1-a)+w[...,:3]
    return img

class Scene:
    T=10.5; work=20
    def __init__(s,geo):
        ly=layers(); h,w=ly['c'].shape[:2]
        s.cv=cv=Canvas(geo,h,w); s.tone=np.float32(BG)
        xs=np.arange(np.floor(cv.x_left)-60,np.ceil(cv.x_right)+60,1.,dtype=np.float32)
        cur=curves(xs,ly['bands'],cv.y_bottom)
        s.plate=render_sea(cv,cur,xs,s.tone).astype(np.float32)      # море целиком рисованное, небо — ровный тон
        tx0,tx1,ty0,_=TORSO; bx=np.zeros((h,w),np.float32); bx[ty0+4:,tx0-12:tx1+12]=1
        s.F=render_front(cv,cur,xs,1)*np.clip(cv.warp(bx),0,1)
        c_,sa=ly['sign']; s.sign=csprite(cv,c_,sa,((SIGN[0]+SIGN[1])/2,(SIGN[2]+SIGN[3])/2))
        s.woman=csprite(cv,*ly['woman'],WAIST); s.hat=csprite(cv,*ly['hat'],HAT_ANC)
        # сетка смещений волн: считаем на 1/4 и растягиваем
        s.q=4; Hq,Wq=(cv.H+3)//4,(cv.W+3)//4
        Xq,Yq=np.meshgrid((np.arange(Wq)*4+1.5-cv.ox)/cv.k,(np.arange(Hq)*4+1.5-cv.oy)/cv.k)
        s.xq,s.yq=Xq.astype(np.float32),Yq.astype(np.float32)
        s.r0=max(int(cv.oy+cv.k*330)-8,0)                   # выше — небо, его не двигаем
        s.gx,s.gy=np.meshgrid(np.arange(cv.W,dtype=np.float32),np.arange(s.r0,cv.H,dtype=np.float32))
        # верх шляпы над экраном / правый край экрана (px картины)
        s.y_top=cv.y_top; s.x_right=cv.x_right

    def amp(s,t): return 1+.6*win(t,1.1,1.9,2.6,4.4)    # море ходит всё время, в порыв — сильнее
    def D(s,x,y,t):
        """вертикальное смещение моря (px картины) в точке картины (x, y)"""
        A=s.amp(t)
        if A<=0: return 0*x
        wy=ease((y-330)/150)
        return A*wy*(9*np.sin(2*np.pi*(x/520-t/3.4)+y/70)+4*np.sin(2*np.pi*(x/230+t/2.6)+y/45+1.3))

    def body(s,t):
        """покачивание женщины: (сдвиг по y px картины, поворот °)"""
        A=s.amp(t)
        dw=A*(4.5*np.sin(2*np.pi*t/2.7)+1.2*np.sin(2*np.pi*t/1.3+.8))
        th=1.0*A*np.sin(2*np.pi*t/3.4+.6)
        return dw,th

    def to_world(s,x,y,t):
        """точку, привязанную к фигуре, — в мир (px картины)"""
        dw,th=s.body(t); r=np.radians(th); px,py=WAIST
        dx,dy=x-px,y-py
        return px+np.cos(r)*dx+np.sin(r)*dy, py-np.sin(r)*dx+np.cos(r)*dy+dw, th

    # ---- шляпа: (x, y, поворот°, sx, sy) в координатах «на голове» ----
    def hat_local(s,t):
        x0,y0=HAT_ANC
        if t<1.05 or t>=9.1: return x0,y0,0.,1.,1.
        if t<1.75:                                          # ветер поддевает левый край поля, шляпа приподнимается
            u=(t-1.05)/.7; lift=10*ein(u)
            return x0+6*ein(u),y0-lift,-(7*ease(u*1.4)+3*np.sin(2*np.pi*u*2.2)*u),1.,1.-.05*u
        if t<4.0:                                           # срывается и кувыркаясь улетает за правый край
            u=(t-1.75)/2.25; xe=s.x_right+330
            x=x0+6+(xe-x0-6)*(.3*u+.7*ein(u)**.8)
            y=y0-10-170*eout(u)+38*np.sin(np.pi*u*1.6)*u
            ph=2*np.pi*1.35*eout(u*1.1); sy=np.cos(ph)
            sy=np.sign(sy)*max(abs(sy),.05) if sy!=0 else .05
            ang=-(10+40*u)-22*np.sin(ph*.5)
            return x,y,ang,1.-.06*u,sy*(1.-.05)
        if t<5.0: return s.x_right+600,y0,0.,1.,1.        # за экраном
        if t<8.4:                                           # падает сверху, раскачиваясь как лист
            u=(t-5.0)/3.4; ys=s.y_top-140
            f=1-(1-u)**1.5; y=ys+(y0-ys)*f
            sw=np.sin(2*np.pi*1.75*u); env=(1-u)**1.3
            x=x0+30*(1-u)**2+95*env*sw
            ang=17*env*np.cos(2*np.pi*1.75*u)
            sy=1-.18*env*abs(np.cos(2*np.pi*1.75*u))
            y-=10*env*(1-abs(sw))                           # на краях размаха чуть подлетает
            return x,y,ang,1.,sy
        u=(t-8.4)/.7                                        # уселась: поля чуть пружинят
        e=np.exp(-4*u)*(1-u)
        return x0,y0+1.5*e*np.sin(2*np.pi*1.6*u),2.5*e*np.sin(2*np.pi*2.1*u+1.),1.,1-.04*e*np.cos(2*np.pi*2.2*u)

    def gust(s,img,t):
        """три штриха ветра слева направо над шляпой"""
        if not .55<t<2.4: return img
        cv=s.cv; polys=[]; al=[]
        for i,(yc,t0,ln) in enumerate([(205,.55,230),(250,.7,280),(300,.85,200)]):
            u=(t-t0)/1.5
            if not 0<u<1: continue
            head=-200+(s.x_right*.5+380)*(.55*u+.45*eout(u)); xs=np.linspace(head-ln*np.sin(np.pi*u)**.6,head,40)
            ys=yc+12*np.sin(2*np.pi*xs/320+i*1.7)-14*np.exp(-((xs-200)/120)**2)
            polys.append(np.stack([xs,ys],1)); al.append(np.sin(np.pi*u)**1.2)
        for p,a_ in zip(polys,al):
            la=lines_alpha(cv,[p],1.6)*a_*.55; img=img*(1-la[...,None])+LINE_RGB*la[...,None]
        return img

    def frame(s,t):
        cv=s.cv
        Dq=s.D(s.xq,s.yq,t)*cv.k
        Dc=cv2.resize(Dq.astype(np.float32),(Dq.shape[1]*4,Dq.shape[0]*4),interpolation=cv2.INTER_LINEAR)[s.r0:cv.H,:cv.W]
        my=s.gy-Dc; img=s.plate.copy(); F=s.F.copy()
        img[s.r0:]=cv2.remap(s.plate,s.gx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        F[s.r0:]=cv2.remap(s.F,s.gx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        sx,sy=(SIGN[0]+SIGN[1])/2,(SIGN[2]+SIGN[3])/2; d=float(s.D(np.float32(sx),np.float32(sy),t))
        img=place(img,s.sign,*cv.xy(sx,sy+d))
        img=s.gust(img,t)
        dw,th=s.body(t)
        wx,wy=cv.xy(WAIST[0],WAIST[1]+dw)
        img=place(img,s.woman,wx,wy,th,occl=F)
        hx,hy,ha,hsx,hsy=s.hat_local(t)
        x,y,th2=s.to_world(hx,hy,t); cx,cy=cv.xy(x,y)
        img=place(img,s.hat,cx,cy,ha+th2,hsx,hsy)
        return np.clip(img+.5,0,255).astype(np.uint8)

if __name__=='__main__': cli(Scene,'s20')
