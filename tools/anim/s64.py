# «В предвкушении» (работа 64). Снег за окном оживает и падает, потом выходит из окна и сыплется на весь экран;
# на ёлке по очереди загораются лоскутки, как гирлянда; пол продолжается до краёв экрана. Одна снежинка садится
# девушке на «букет» — та встряхивает головой, снежинка слетает и тает. Вечер: стена BG, пол FLOOR, окно светится.
# Пауз нет (ролик сам срезает покой): снег идёт с первого кадра и до последнего.
import numpy as np, cv2
from lib import *

SRC='site/images/work-64.jpg'
BG=(47,72,88)                              # вечер: стена — глубокий сине-бирюзовый (им же s.tone)
FLOOR=(128,82,62)                          # пол — тёплый тёмный терракотовый
SIGN=np.float32([238,218,190])             # подпись — светлая на тёмном полу
WIN=(86.5,123.8,474.,662.)                 # окно: внешние края рамы (x0, y0, x1, y1), px картины 769×899
VX=[86.5,182.5,278.5,375.5,473.5]          # вертикали решётки (центры)
HY=[124.,259.,393.,527.5,662.]             # горизонтали решётки
LINE=np.array([116,128,141],np.float32)    # цвет решётки
PIV=(243.,588.)                            # опора головы — низ шеи над вырезом платья
SNOW=np.array([255,255,255],np.float32)
rng=np.random.default_rng(64)

def nconv(img,m,s):
    """размытие только по маске m (нормированная свёртка); где маски вокруг мало — с радиусом вчетверо больше"""
    m=m.astype(np.float32); out=None
    for sg in (s*4,s):
        num=cv2.GaussianBlur(img*m[...,None],(0,0),sg); den=cv2.GaussianBlur(m,(0,0),sg)
        v=num/np.maximum(den,1e-6)[...,None]
        out=v if out is None else out+(v-out)*np.clip(den/.15,0,1)[...,None]
    return out

def floor_line(a):
    """y линии пола по столбцам (субпиксельно): середина перехода синего канала стена → пол"""
    H,W,_=a.shape; ys=np.full(W,np.nan)
    for x in range(W):
        b=a[700:780,x,2]
        if (b[:20].mean()-b[-20:].mean())<30: continue
        mid=(b[:20].mean()+b[-20:].mean())/2; i=np.argmax(b<mid)
        if 0<i: ys[x]=700+i-1+(b[i-1]-mid)/(b[i-1]-b[i])
    xs=np.where(~np.isnan(ys))[0]
    ok=np.abs(ys[xs]-np.median(ys[xs]))<6; xs=xs[ok]
    p=np.polyfit(xs,ys[xs],1); return p                    # y = p0·x + p1

def recolor(p,tw,tf,yf,Y,X):
    """старые стена/пол → BG/FLOOR; края фигур разнимаются со старым тоном (без ореолов): α по проекции на цвет фигуры"""
    tw,tf,nw,nf=[np.float32(v) for v in (tw,tf,BG,FLOOR)]
    dv=tf-tw; beta=np.clip(((p-tw)*dv).sum(2)/(dv*dv).sum(),0,1)
    near=np.abs(Y-yf)<4; beta=np.where(near,beta,(Y>yf).astype(np.float32))[...,None]
    old=tw+dv*beta; new=nw+(nf-nw)*beta
    d=np.abs(p-old).sum(2)
    inside=(X>WIN[0]+3)&(X<WIN[2]-3)&(Y>WIN[1]+3)&(Y<WIN[3]-3)
    core=cv2.morphologyEx((d>90).astype(np.uint8),cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
    deep=(cv2.erode(core,np.ones((3,3),np.uint8))>0)|inside
    # α края: доля «краски» = d / d самого насыщенного пикселя рядом (он — чистая краска фигуры)
    dmax=cv2.dilate(d.astype(np.float32),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    al=np.where(dmax>90,np.clip(d/np.maximum(dmax,1),0,1),0); al=np.where(deep,1,al)[...,None]
    # p = α·краска + (1−α)·старый тон  →  α·краска + (1−α)·новый тон; чистый фон — новый тон и тихая фактура
    out=np.where(al>0,p+(1-al)*(new-old),new+(p-old)*.6)
    # подпись художницы на тёмном полу — светлой краской, иначе тёмные буквы тонут в терракоте
    sig=(X>615)&(Y>852)
    return np.where(sig[...,None],new+al*(SIGN-new),out)

def layers():
    a=load_work(SRC); H,W,_=a.shape; r,g,b=a[...,0],a[...,1],a[...,2]
    Y,X=np.mgrid[0:H,0:W]
    pf=floor_line(a); yf=pf[0]*X+pf[1]
    wall=(a.min(2)>212)&(np.ptp(a,2)<16)&(Y<yf-2)
    floor=(r>215)&(r-b>35)&(Y>yf+2)&(r-g<45)
    # снежинки в окне: светлее местной синевы
    wr=np.zeros((H,W),bool); wr[127:660,90:472]=True
    med=cv2.medianBlur(np.clip(a,0,255).astype(np.uint8),15).astype(np.float32)
    dots=wr&((a-med).min(2)>12)&(a.min(2)>203)&(b>=r-2)
    n,lab,st,cen=cv2.connectedComponentsWithStats(dots.astype(np.uint8))
    flakes=[(cen[i][0],cen[i][1],np.sqrt(st[i,4]/np.pi)) for i in range(1,n) if 2<=st[i,4]<=80]
    dm=cv2.dilate(dots.astype(np.uint8)*255,np.ones((5,5),np.uint8))
    a=cv2.inpaint(np.clip(a,0,255).astype(np.uint8),dm,4,cv2.INPAINT_TELEA).astype(np.float32)
    # свет фото: отдельно по стене и по полу
    blue=(b-r>22)&(b>195)
    inw=(X>WIN[0])&(X<WIN[2])&(Y>WIN[1])&(Y<WIN[3])
    lw=nconv(a,wall,35); lf=nconv(a,floor,35); lp=nconv(a,blue&inw,35)
    tw=np.median(a[wall],0); tf=np.median(a[floor],0); tp=np.median(a[blue&inw],0)
    lit=np.where((Y<yf)[...,None],tw-lw,tf-lf); lit=np.where(inw[...,None],tp-lp,lit)
    lit=cv2.GaussianBlur(lit,(0,0),2)
    a=a+lit
    # у краёв фото — тонкая тёмная кромка: добрать мелким радиусом в полосе 16 px
    ed=np.minimum(np.minimum(X,W-1-X),np.minimum(Y,H-1-Y)).astype(np.float32)
    bgm=(wall|floor)&~inw
    l2=np.where((Y<yf)[...,None],tw-nconv(a,wall,3),tf-nconv(a,floor,3))
    a=a+l2*np.clip((16-ed)/8,0,1)[...,None]
    # фактура холста на стене и полу — тише, чтобы край картины не читался рамкой
    sm=cv2.GaussianBlur(cv2.erode(bgm.astype(np.uint8),np.ones((3,3),np.uint8)).astype(np.float32),(0,0),1.2)
    tt=np.where((Y<yf)[...,None],tw,tf)
    a=a-(a-tt)*(.65*sm)[...,None]
    # девушка: связная область «не стена, не пол, не окно» вокруг платья
    blue=(b-r>22)&(b>195)
    gridm=np.zeros((H,W),np.uint8)
    for x in VX: cv2.line(gridm,(int(round(x*16)),int(WIN[1]*16)),(int(round(x*16)),int(WIN[3]*16)),255,4,cv2.LINE_8,4)
    for y in HY: cv2.line(gridm,(int(WIN[0]*16),int(round(y*16))),(int(WIN[2]*16),int(round(y*16))),255,4,cv2.LINE_8,4)
    # линии решётки, по обе стороны которых фон (а не девушка), — не девушка
    bgk=(blue|wall|floor).astype(np.uint8)
    sh=lambda m,dx,dy: np.roll(np.roll(m,dy,0),dx,1)
    lineonly=(gridm>0)&(((sh(bgk,3,0)>0)&(sh(bgk,-3,0)>0))|((sh(bgk,0,3)>0)&(sh(bgk,0,-3)>0)))
    band=(np.abs(Y-yf)<3.5)&(sh(wall.astype(np.uint8),0,4)>0)&(sh(floor.astype(np.uint8),0,-4)>0)
    other=~wall&~floor&~blue&~lineonly&~band
    bb=np.zeros((H,W),bool); bb[440:860,190:305]=True; other&=bb
    other=cv2.morphologyEx(other.astype(np.uint8),cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
    n,lab,st,_=cv2.connectedComponentsWithStats(other)
    girl=np.isin(lab,[i for i in range(1,n) if st[i,4]>30])
    girl=cv2.morphologyEx(girl.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))>0
    # голова с шеей — выше опоры; мягкая альфа по краю
    core=girl.copy()
    ring=cv2.dilate(core.astype(np.uint8),np.ones((5,5),np.uint8))>0
    # фон под девушкой (окно): синева + решётка, перерисованная заново
    gl=cv2.dilate(girl.astype(np.uint8)*255,np.ones((7,7),np.uint8))
    under=cv2.inpaint(np.clip(a,0,255).astype(np.uint8),gl,6,cv2.INPAINT_TELEA).astype(np.float32)
    pane=nconv(under,blue&~(gridm>0),6)
    under=np.where((gl>0)[...,None]&(Y<WIN[3]-3)[...,None],pane,under)
    la=np.zeros((H,W),np.uint8)
    for x in VX:
        cv2.line(la,(int(round(x*16)),int(WIN[1]*16)),(int(round(x*16)),int(WIN[3]*16)),255,2,cv2.LINE_AA,4)
    for y in HY:
        cv2.line(la,(int(WIN[0]*16),int(round(y*16))),(int(WIN[2]*16),int(round(y*16))),255,2,cv2.LINE_AA,4)
    la=cv2.GaussianBlur(la.astype(np.float32)/255,(0,0),.6)*(gl>0)
    under=under*(1-la[...,None])+LINE*la[...,None]
    d_bg=np.abs(a-under).sum(2)
    ga=np.clip((d_bg-10)/50,0,1)*ring; ga[cv2.erode(core.astype(np.uint8),np.ones((3,3),np.uint8))>0]=1
    head=ga*(Y<PIV[1]+4)*np.clip((PIV[1]+4-Y)/6,0,1)
    body=ga*np.clip((Y-PIV[1]+6)/6,0,1)
    # окно для снега: стекло без решётки и без девушки
    glass=np.zeros((H,W),np.float32); glass[int(WIN[1])+2:int(WIN[3])-1,int(WIN[0])+2:int(WIN[2])-1]=1
    glass*=1-np.clip(cv2.GaussianBlur(gridm.astype(np.float32)/255,(0,0),.7)*1.6,0,1)
    glass*=1-np.clip(ga+0,0,1)
    # ёлка: лоскутки — зелёные заливки между золотыми линиями
    tr=np.zeros((H,W),bool); tr[195:800,455:715]=True
    green=tr&(g>r+18)&(g>b+8)
    gold=tr&(r>115)&(r>b+40)&~green&~floor&~wall&(Y<yf-3)
    gm=cv2.erode(green.astype(np.uint8),np.ones((2,2),np.uint8))
    n,plab,pst,pcen=cv2.connectedComponentsWithStats(gm)
    keep=np.zeros(n,bool); keep[1:]=pst[1:,4]>25
    plab=np.where(keep[plab],plab,0)
    # растянуть метки на края лоскутка (до золотой линии)
    pl=plab.astype(np.float32); m=(plab>0).astype(np.uint8)
    for _ in range(3):
        dl=cv2.dilate(pl,np.ones((3,3),np.uint8)); grow=(m==0)&green
        pl=np.where(grow&(dl>0),dl,pl); m=(pl>0).astype(np.uint8)
    plab=pl.astype(np.int32)
    a=recolor(a,tw,tf,yf,Y,X); under=recolor(under,tw,tf,yf,Y,X)
    return dict(a=a,under=under,head=head.astype(np.float32),body=body.astype(np.float32),glass=glass,pf=pf,
                tw=tw,tf=tf,flakes=flakes,plab=plab,pcen=pcen,gold=gold,girl=ga)

def tau_table(T,f):
    ts=np.arange(0,T+1e-9,1/600); v=f(ts); tau=np.concatenate([[0],np.cumsum((v[1:]+v[:-1])/2)/600]); return ts,tau

class Scene:
    T=12.0; work=64
    def __init__(s,geo):
        ly=layers(); h,w=ly['a'].shape[:2]; s.ly=ly
        s.cv=cv=Canvas(geo,h,w); s.tone=np.float32(BG); k=cv.k
        # фон на весь экран: стена выше линии пола, пол ниже; линия пола — продолжение линии картины
        Ys,Xs=np.mgrid[0:cv.H,0:cv.W].astype(np.float32)
        px=(Xs-cv.ox)/k; py=(Ys-cv.oy)/k
        p0,p1=ly['pf']; yl=p0*np.clip(px,0,w)+p1
        fa=np.clip((py-yl)*k+.5,0,1)[...,None]
        plate=np.float32(BG)*(1-fa)+np.float32(FLOOR)*fa
        cov=np.clip(cv.warp(np.ones((h,w),np.float32)),0,1)
        s.base=plate*(1-cov[...,None])+cv.warp(ly['under']*1)            # под головой — окно
        # картина без головы: under всюду, кроме головы (там under и есть) — значит base = картина с фоном под головой
        bd=ly['body']; s.base=s.base  # тело вклеиваем ниже
        s.body=cv.warp(ly['a']*bd[...,None]); s.bodya=np.clip(cv.warp(bd),0,1)
        s.base=over(s.base,s.body,s.bodya)
        s.glass=np.clip(cv.warp(ly['glass']),0,1)
        # голова — спрайт
        hm=ly['head']; s.headsp=Sprite.cut(ly['a'],hm,PIV,3)
        ys,xs=np.where(hm>.5); top=ys.min(); cx=xs[ys<top+4].mean()
        s.htop=(cx,top-1.5)
        # ёлка: метки лоскутков в экранных px
        pl=ly['plab']; ys,xs=np.where(pl>0); y0,y1,x0,x1=ys.min(),ys.max()+1,xs.min(),xs.max()+1
        X0,Y0=[int(np.floor(v))-2 for v in cv.xy(x0,y0)]; X1,Y1=[int(np.ceil(v))+2 for v in cv.xy(x1,y1)]
        s.tb=(X0,Y0,X1,Y1)
        Mi=np.float32([[1/k,0,(X0-cv.ox)/k],[0,1/k,(Y0-cv.oy)/k]])
        s.tlab=cv2.warpAffine(pl.astype(np.float32),Mi,(X1-X0,Y1-Y0),flags=cv2.INTER_NEAREST|cv2.WARP_INVERSE_MAP,
                              borderValue=0).astype(np.int32)
        wgt=cv2.warpAffine((pl>0).astype(np.float32),Mi,(X1-X0,Y1-Y0),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,borderValue=0)
        s.twgt=np.clip(wgt,0,1)
        s.tgold=np.clip(cv2.warpAffine(ly['gold'].astype(np.float32),Mi,(X1-X0,Y1-Y0),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP),0,1)
        n=int(pl.max())+1; cen=np.zeros((n,2))
        for i in range(1,n):
            yy,xx=np.where(pl==i)
            if len(xx): cen[i]=(xx.mean(),yy.mean())
        s.pc=cen
        ty0,ty1=y0,y1; u=(ty1-cen[:,1])/(ty1-ty0)                       # 0 у низа, 1 у верхушки
        side=(cen[:,0]-583)/120
        s.ph=(u*2.2+side*.35+rng.random(n)*.12)%1                        # фаза гирлянды: бегущая спираль снизу вверх
        s.pcol=rng.choice(3,n,p=[.5,.2,.3]); s.thr=rng.uniform(.05,.85,n)
        # снег в окне
        fl=np.array(ly['flakes']); s.wf=fl
        x0w,y0w,x1w,y1w=WIN; s.wh=y1w-y0w
        ne=60; ex=np.c_[rng.uniform(x0w+4,x1w-4,ne),rng.uniform(y0w,y1w,ne),rng.uniform(.9,1.6,ne)]
        s.wx=np.r_[fl[:,0],ex[:,0]]; s.wy=np.r_[fl[:,1],ex[:,1]]; s.wr=np.r_[fl[:,2].clip(1,2.2),ex[:,2]]
        s.wext=np.r_[np.zeros(len(fl)),np.ones(ne)]
        s.wn=rng.integers(1,3,len(s.wx)); s.wph=rng.random(len(s.wx))*6.3; s.wam=rng.uniform(3,9,len(s.wx))
        s.ts,s.tau=tau_table(s.T,lambda t: ramp(t,0,.8))           # снег пошёл сразу и не кончается
        s.tauT=s.tau[-1]
        # снег в комнате: поле на весь экран (px картины)
        s.xa,s.xb=cv.x_left-30,cv.x_right+30; s.ya,s.yb=cv.y_top-30,cv.y_bottom+30
        area=(s.xb-s.xa)*(s.yb-s.ya); nr=int(area/9000)
        s.rx=rng.uniform(s.xa,s.xb,nr); s.ry=rng.uniform(s.ya,s.yb,nr)
        z=rng.random(nr); s.rr=2.2+4.5*z**1.5; s.rv=55+110*z+rng.uniform(-10,10,nr)
        s.rph=rng.random(nr)*6.3; s.ram=rng.uniform(6,22,nr); s.rP=rng.uniform(2.2,4.5,nr)
        # где гуще: справа от текста (компьютер), в верхней части (телефон)
        sx,sy=cv.xy(s.rx,s.ry)
        if geo=='m': s.rden=lambda X,Y: 1-.8*ramp(Y,950,1500)
        else: s.rden=lambda X,Y: .4+.6*ramp(X,.40*cv.W,.50*cv.W)

    def T_(s,t): return np.interp(t,s.ts,s.tau)

    def head_ang(s,t):
        a=2.5*win(t,5.0,5.5,7.9,8.2)                                     # снежинка села — чуть приподняла «букет»
        u=t-8.2
        if 0<u<1.6: a+=-7*np.sin(2*np.pi*u*2.6)*np.exp(-u*2.4)*ramp(u,0,.08)
        return a

    def special(s,t,ang):
        """снежинка-героиня: из стекла окна плывёт к «букету», сидит, слетает при встряске и тает"""
        hx,hy=s.htop; px,py=PIV
        th=np.radians(ang); dx,dy=hx-px,hy-py
        sx=px+dx*np.cos(th)+dy*np.sin(th); sy=py-dx*np.sin(th)+dy*np.cos(th)
        r=11.0
        if t<2.4: return None
        if t<5.0:
            u=(t-2.4)/2.6; e=ease(u)
            x0,y0=hx+70,hy-170
            x=x0+(sx-x0)*e+14*np.sin(2*np.pi*u*1.6)*(1-e); y=y0+(sy-r-y0)*e
            return x,y,r,ramp(t,2.4,2.9)
        if t<8.32: return sx,sy-r,r,1.
        u=t-8.32                                                          # слетела: вбок и вниз, тает на полу
        x=sx+95*u+10*np.sin(u*7); y=min(sy-r-70*u+260*u*u,790.)
        return x,y,r*(1-.6*ramp(u,1.15,1.6)),1-ramp(u,1.15,1.6)

    def dots(s,X,Y,R,A):
        """альфа кружков на холсте (экранные px): точное покрытие пикселя, без «ромбиков» у мелких"""
        cv=s.cv; al=np.zeros((cv.H,cv.W),np.float32)
        for x,y,r,a in zip(X,Y,R,A):
            if a<.01 or r<=0: continue
            x0,y0=int(np.floor(x-r-2)),int(np.floor(y-r-2)); x1,y1=int(np.ceil(x+r+2)),int(np.ceil(y+r+2))
            X0,Y0=max(x0,0),max(y0,0); X1,Y1=min(x1,cv.W),min(y1,cv.H)
            if X1<=X0 or Y1<=Y0: continue
            yy,xx=np.mgrid[Y0:Y1,X0:X1]; d=np.hypot(xx+.5-x,yy+.5-y)
            re=max(r,.75); c=np.clip(re+.5-d,0,1)*min(a,1)*min(1,(r/.75)**2)
            al[Y0:Y1,X0:X1]=np.maximum(al[Y0:Y1,X0:X1],c)
        return al

    def star(s,x,y,r,a,ang):
        """снежинка-героиня: шесть лучей с веточками (экранные px)"""
        cv=s.cv; ss=16; pad=int(r)+4
        X0,Y0=int(x)-pad,int(y)-pad; w=2*pad+1
        al=np.zeros((w,w),np.uint8); cx,cy=(x-X0)*ss,(y-Y0)*ss; th=max(1,int(round(r*.2)))
        for i in range(6):
            q=np.radians(ang+60*i); ex,ey=cx+np.cos(q)*r*ss,cy+np.sin(q)*r*ss
            cv2.line(al,(int(cx),int(cy)),(int(ex),int(ey)),255,th,cv2.LINE_AA,4)
            for sgn in (-1,1):
                bx,by=cx+np.cos(q)*r*.55*ss,cy+np.sin(q)*r*.55*ss; qb=q+sgn*np.radians(50)
                cv2.line(al,(int(bx),int(by)),(int(bx+np.cos(qb)*r*.35*ss),int(by+np.sin(qb)*r*.35*ss)),255,th,cv2.LINE_AA,4)
        al=al.astype(np.float32)/255*a
        out=np.zeros((cv.H,cv.W),np.float32); x0,y0=max(X0,0),max(Y0,0); x1,y1=min(X0+w,cv.W),min(Y0+w,cv.H)
        if x1>x0 and y1>y0: out[y0:y1,x0:x1]=al[y0-Y0:y1-Y0,x0-X0:x1-X0]
        return out

    def frame(s,t):
        cv=s.cv; k=cv.k; img=s.base.copy(); tau=s.T_(t); S=ramp(t,0,.8)
        # 1. снег за окном
        x=s.wx+s.wam*np.sin(tau*.9+s.wph)*S*.5
        y=WIN[1]+((s.wy-WIN[1])+s.wn*56*tau)%s.wh
        al=np.where(s.wext>0,ramp(t,.4,2.0),1.)
        X,Yy=cv.xy(x,y); a=s.dots(X,Yy,s.wr*k,al)*s.glass
        img=img*(1-a[...,None])+SNOW*a[...,None]
        # 2. гирлянда на ёлке
        G=ramp(t,.5,1.9)
        X0,Y0,X1,Y1=s.tb
        if G>0:
            u=(s.ph-t/1.9)%1
            L=np.clip((.13-np.abs(u-.5))/.04,0,1)
            L=np.clip(L+win(t,6.2,6.35,6.8,6.95)*(s.thr<.6),0,1)        # на миг — пол-ёлки сразу
            L=L*np.clip((G-s.thr)*10,0,1)                                   # гирлянда включается и гаснет по лоскутку
            cols=np.array([[250,214,92],[238,104,52],[255,238,150]],np.float32)
            Lc=L[:,None]*cols[s.pcol]
            lab=s.tlab; w=s.twgt[...,None]*(lab>0)[...,None]
            reg=img[Y0:Y1,X0:X1]
            lit=Lc[lab]*.85; amt=L[lab][...,None]*.85
            reg=reg*(1-amt*w)+lit*w
            reg=reg+s.tgold[...,None]*G*np.float32([22,16,-6])*(.6+.4*np.sin(t*5))
            img[Y0:Y1,X0:X1]=reg
        # 3. голова
        ang=s.head_ang(t)
        img=put(cv,img,s.headsp,PIV[0],PIV[1],1.,ang)
        y0=int(cv.xy(0,PIV[1]-14)[1]); y1=int(cv.xy(0,PIV[1]+14)[1])+1
        img[y0:y1]=over(img[y0:y1],s.body[y0:y1],s.bodya[y0:y1])
        # 4. снег в комнате: выходит из окна, сыплется на весь экран, уходит обратно в окно
        R=3600*ramp(t,1.8,4.4)
        if R>0:
            xr=s.rx+s.ram*np.sin(2*np.pi*tau/s.rP+s.rph)
            yr=s.ya+((s.ry-s.ya)+s.rv*tau)%(s.yb-s.ya)
            dx=np.maximum(np.maximum(WIN[0]-xr,xr-WIN[2]),0); dy=np.maximum(np.maximum(WIN[1]-yr,yr-WIN[3]),0)
            dist=np.hypot(dx,dy)
            vis=np.clip((R-dist)/160,0,1)
            X,Yy=cv.xy(xr,yr); vis=vis*s.rden(X,Yy)
            rr=s.rr*k
            m=(vis>.01)&(X>-20)&(X<cv.W+20)&(Yy>-20)&(Yy<cv.H+20)
            fill=s.dots(X[m],Yy[m],rr[m],vis[m])
            img=img*(1-fill[...,None])+SNOW*fill[...,None]
        sp=s.special(t,ang)
        if sp is not None:
            sx,sy=cv.xy(sp[0],sp[1]); rot=15*t+ang
            fill=s.star(sx,sy,sp[2]*k,sp[3],rot)
            img=img*(1-fill[...,None])+SNOW*fill[...,None]
        return np.clip(img+.5,0,255).astype(np.uint8)

if __name__=='__main__': cli(Scene,'s64')
