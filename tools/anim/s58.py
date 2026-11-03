# «ОТБРОСЬ ВСЁ!» (работа 58). Небо — закатное розовое (BG), море и горы как на картине; море продолжается до краёв
# экрана, трамплин гаснет за рамой, правая зелёная гора дорисована за рамой. Отброшенный лифчик долго летит большой
# дугой через небо, кувыркаясь, и цепляется за вершину горы — треплется там, как флаг. Прыгунья ныряет: брызги, круги
# по воде (в перспективе). Потом выпрыгивает дельфином по дуге и снова уходит в воду — круги, флаг на горе.
import numpy as np, cv2
from lib import *

SRC='site/images/work-58.jpg'
BG=(243,201,192)            # небо: нежно-розовое закатное (море и горы — как на картине)
YH=518.8                     # горизонт (px картины, по середине перехода небо → море)
FE=12                        # ширина перехода фактуры фото в ровный тон у краёв картины, px картины
DIVER_BOX=(55,316,312,392)   # x0, y0, x1, y1
TOP_BOX=(185,252,325,310)
SIG_BOX=(470,828,650,892)
BOARD_BAND=(376,401)         # строки трамплина
HZ_BAND=(510,530)            # строки горизонта

def polyfit2d(img,m,deg=3,step=3):
    """гладкий свет фото: полином deg по (x, y) по пикселям маски m, на всю картину"""
    H,W=m.shape; ys,xs=np.mgrid[0:H:step,0:W:step]; mm=m[::step,::step]
    xn=lambda x:(x-W/2)/W; yn=lambda y:(y-H/2)/H
    def basis(x,y): return np.stack([xn(x)**i*yn(y)**j for i in range(deg+1) for j in range(deg+1-i)],-1)
    B=basis(xs[mm].astype(float),ys[mm].astype(float)); v=img[::step,::step][mm]
    c,*_=np.linalg.lstsq(B,v,rcond=None)
    Y,X=np.mgrid[0:H,0:W]; return (basis(X.astype(float),Y.astype(float))@c).astype(np.float32)

def boxmask(H,W,b):
    m=np.zeros((H,W),bool); m[b[1]:b[3],b[0]:b[2]]=True; return m

def cutout(P,bg,box,thr,dmin,core_fill):
    """вырезать элемент из P на ровном фоне bg: альфа по удалённости от фона, цвет — с вычтенным фоном (без ореола)"""
    H,W,_=P.shape; dist=np.sqrt(((P-bg)**2).sum(2)); bm=boxmask(H,W,box)
    nb=((dist>thr)&bm).astype(np.uint8)
    nb=cv2.morphologyEx(nb,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    n,lab,st,_=cv2.connectedComponentsWithStats(nb)
    nb=np.isin(lab,[i for i in range(1,n) if st[i,4]>25])
    core=np.zeros((H,W),np.float32)
    if core_fill:
        ff=(~nb).astype(np.uint8); cv2.floodFill(ff,None,(0,0),0)          # дыры внутри контура — тоже тело
        filled=nb|(ff>0)
        core=cv2.erode(filled.astype(np.uint8),np.ones((3,3),np.uint8)).astype(np.float32)
    else: filled=nb
    dloc=np.maximum(cv2.dilate(dist,np.ones((7,7),np.uint8)),dmin)
    al=np.clip((dist-thr*.4)/(dloc-thr*.4)*1.1,0,1)
    zone=cv2.dilate(filled.astype(np.uint8),np.ones((5,5),np.uint8))>0
    al=np.maximum(al*zone,core).astype(np.float32)
    pm=np.clip(P-(1-al[...,None])*bg,0,255)
    rgb=np.where(al[...,None]>.02,pm/np.maximum(al[...,None],1e-3),0)
    return np.clip(rgb,0,255).astype(np.float32),al

def fill_shift(P,al,dy):
    """залить место вырезанного элемента той же фактурой холста, взятой со сдвигом dy строк"""
    h=cv2.dilate((al>.003).astype(np.uint8),np.ones((5,5),np.uint8)).astype(np.float32)
    h=np.clip(cv2.GaussianBlur(h,(0,0),1.2)*1.6,0,1)
    src=np.roll(P,-dy,0); return P*(1-h[...,None])+src*h[...,None]

def layers():
    a=load_work(SRC); H,W,_=a.shape
    Y,X=np.mgrid[0:H,0:W]
    sky0=np.median(a[20:200,20:300].reshape(-1,3),0); sea0=np.median(a[600:800,20:400].reshape(-1,3),0)
    dsky=np.abs(a-sky0).sum(2); dsea=np.abs(a-sea0).sum(2)
    nb_sky=cv2.dilate(((dsky>30)&(Y<516)).astype(np.uint8),np.ones((9,9),np.uint8))>0
    nb_sea=cv2.dilate(((dsea>40)&(Y>522)).astype(np.uint8),np.ones((9,9),np.uint8))>0
    clean_sky=(Y<514)&~nb_sky; clean_sea=(Y>525)&~nb_sea&~boxmask(H,W,SIG_BOX)
    # свет фото: гладкий полином по чистому небу и по чистому морю отдельно
    tone_sky=np.median(a[clean_sky],0); tone_sea=np.median(a[clean_sea],0)
    Ls=np.dstack([polyfit2d(a[...,c],clean_sky) for c in range(3)])
    Lm=np.dstack([polyfit2d(a[...,c],clean_sea) for c in range(3)])
    ws=np.clip(Y-YH+.5,0,1)[...,None].astype(np.float32)
    P=a+(tone_sky-Ls)*(1-ws)+(tone_sea-Lm)*ws
    tone_sky=np.median(P[clean_sky],0); tone_sea=np.median(P[clean_sea],0)
    # элементы, которые двигаются / ложатся поверх кругов
    dv_rgb,dv_a=cutout(P,tone_sky,DIVER_BOX,25,150,True)
    dv_a[382:,:80]=0                                                    # трамплин — не её
    tp_rgb,tp_a=cutout(P,tone_sky,TOP_BOX,25,150,True)
    sg_rgb,sg_a=cutout(P,tone_sea,SIG_BOX,22,55,False)
    Pc=fill_shift(P,dv_a,110); Pc=fill_shift(Pc,tp_a,-110); Pc=fill_shift(Pc,sg_a,-110)
    # фактура холста на чистом небе и море — в ровный тон: иначе рамка картины читается прямоугольником
    flat2=np.where((Y<YH)[...,None],tone_sky,tone_sea).astype(np.float32)
    dd=np.abs(Pc-flat2).sum(2)
    n,lab,st,_=cv2.connectedComponentsWithStats((dd>14).astype(np.uint8))
    speck=np.isin(lab,[i for i in range(1,n) if st[i,4]<60])                 # пятнышки на холсте — не рисунок
    e=np.clip((dd-20)/16,0,1).astype(np.float32); e[speck]=0
    e=cv2.GaussianBlur(cv2.dilate(e,np.ones((5,5),np.uint8)),(0,0),1.)
    e[Y>528]=0                                                          # в море ниже горизонта рисунка нет (подпись снята)
    Pc=Pc*e[...,None]+flat2*(1-e[...,None])
    # горизонт слева от гор — ровный профиль края (как за рамой), без светлой каймы фото
    hz=np.median(Pc[HZ_BAND[0]:HZ_BAND[1],1:7],1)
    wx=np.clip((357-np.arange(W))/4,0,1)[None,:,None]
    Pc[HZ_BAND[0]:HZ_BAND[1]]=Pc[HZ_BAND[0]:HZ_BAND[1]]*(1-wx)+hz[:,None,:]*wx
    # небо — в цвет BG: пиксель края смешан с бумагой на (1−α) — эту долю и переводим в BG (без ореолов)
    dist=np.sqrt(((Pc-tone_sky)**2).sum(2)).astype(np.float32)
    core=cv2.erode((dist>40).astype(np.uint8),np.ones((3,3),np.uint8))>0
    dloc=np.maximum(cv2.dilate(dist,np.ones((5,5),np.uint8)),60)
    al=np.where(core,1.,np.clip(dist/dloc,0,1)).astype(np.float32)
    Pc=Pc+(1-al[...,None])*(np.float32(BG)-tone_sky)
    tone_paper=tone_sky; tone_sky=np.float32(BG)
    # профили для продолжения за раму
    def band(arr,lo,hi,base):
        out=base.copy(); r=np.arange(len(base))
        w=np.clip(np.minimum(r-lo,hi-r)/2.5,0,1)[:,None]; return out*(1-w)+arr*w
    rows=np.arange(H)
    flat=np.where((rows<YH)[:,None],tone_sky,tone_sea).astype(np.float32)
    colL=np.median(Pc[:,1:7],1)
    Vout=band(colL,*HZ_BAND,flat)                                        # небо → горизонт → море
    CL=band(colL,*BOARD_BAND,Vout)                                       # + трамплин
    green=np.median(Pc[400:500,664:671].reshape(-1,3),0)
    flatG=np.where((rows<YH)[:,None],green,tone_sea).astype(np.float32)
    Vin=band(np.median(Pc[:,664:671],1),*HZ_BAND,flatG)                  # зелёная гора → её подошва → море
    # склон правой зелёной горы: середина тёмного контура по строкам, прямая x = fa + fb·y
    g=Pc.mean(2); pts=[]
    for y in range(300,386):
        x0=630+(y-292)*.42; xs=np.arange(int(x0)-6,min(int(x0)+7,W))
        wgt=np.clip(110-g[y,xs],0,None)
        if wgt.sum()>30: pts.append((y,(wgt*xs).sum()/wgt.sum()))
    pts=np.array(pts); fb,fa=np.polyfit(pts[:,0],pts[:,1],1)
    us=np.arange(-10,10.01,.25); F=np.zeros((len(us),3)); n=0
    for y in range(320,362):
        xc=fa+fb*y; F+=np.stack([np.interp(xc+us,np.arange(W),Pc[y,:,c]) for c in range(3)],1); n+=1
    F/=n
    # тень трамплина — остальное, что меряем, — для траекторий
    return dict(P=Pc,tone_sky=tone_sky,tone_paper=tone_paper,tone_sea=tone_sea,CL=CL,Vout=Vout,Vin=Vin,F=F,us=us,fa=fa,fb=fb,green=green,
                diver=(dv_rgb,dv_a),top=(tp_rgb,tp_a),sig=(sg_rgb,sg_a))

def synth(ly,X,Y):
    """фон за рамой в точках (X, Y) px картины: небо/море с горизонтом, трамплин слева, склон зелёной горы справа"""
    H=len(ly['CL']); r=np.arange(H)
    def prof(arr,y): return np.stack([np.interp(y,r,arr[:,c]) for c in range(3)],-1)
    wb=ease(1+X/120.)[...,None]                                       # трамплин гаснет за ~120 px за левым краем рамы
    left=prof(ly['CL'],Y)*wb+prof(ly['Vout'],Y)*(1-wb)
    u=X-(ly['fa']+ly['fb']*Y)
    Fu=np.stack([np.interp(u,ly['us'],ly['F'][:,c]) for c in range(3)],-1)
    Fu=np.where(((u>10)|(Y<300)|(X<600))[...,None],ly['tone_sky'],Fu)       # гора — только справа и ниже вершины
    beta=np.clip(u/2+.5,0,1)[...,None]
    low=prof(ly['Vin'],Y)*(1-beta)+prof(ly['Vout'],Y)*beta
    wl=np.clip((Y-(YH-4))/1.5,0,1)[...,None]
    right=Fu*(1-wl)+low*wl
    sel=(X<336)[...,None]
    return np.where(sel,left,right).astype(np.float32)

def put_layer(cv,sp,x,y,sc=1.,ang=0.):
    """как lib.put, но возвращает кусок (y0, x0, премультиплицированный rgba) — чтобы обрезать по воде"""
    k=cv.k*sc; ax,ay=sp.anc; offx,offy=sp.off; cx,cy=cv.xy(x,y)
    R=cv2.getRotationMatrix2D((0,0),ang,k); R[:,2]=[cx,cy]
    M=(np.vstack([R,[0,0,1]])@np.float32([[1,0,offx-ax],[0,1,offy-ay],[0,0,1]]))[:2]
    ph,pw=sp.pm.shape[:2]
    cs=np.array([[0,0,1],[pw,0,1],[0,ph,1],[pw,ph,1]],np.float32)@M.T
    x0,y0=np.floor(cs.min(0)).astype(int)-2; x1,y1=np.ceil(cs.max(0)).astype(int)+2
    x0,y0=max(x0,0),max(y0,0); x1,y1=min(x1,cv.W),min(y1,cv.H)
    if x1<=x0 or y1<=y0: return None
    M2=M.copy(); M2[:,2]-=[x0,y0]
    w=cv2.warpAffine(sp.pm,M2,(x1-x0,y1-y0),flags=cv2.INTER_LINEAR if sc*cv.k<1.5 else cv2.INTER_CUBIC,borderValue=0)
    return y0,x0,w

def rot(v,ang):
    """повернуть вектор (px картины, y вниз) на ang° против часовой — как в put"""
    a=np.radians(ang); c,s=np.cos(a),np.sin(a); return np.array([c*v[0]+s*v[1],-s*v[0]+c*v[1]])

class Scene:
    T=10.5; work=58
    F_PERSP=610.             # «фокус» перспективы воды, px картины: начальный круг — эллипс ~0,22
    XC=336.                  # точка схода по x
    YS1=655.; YS2=668.       # где вода у входа и у выхода (строка картины)
    def __init__(s,geo):
        ly=layers(); h,w=ly['P'].shape[:2]
        s.cv=cv=Canvas(geo,h,w); s.tone=ly['tone_sky']; s.ly=ly
        # неподвижный фон: фото без света у краёв переходит в дорисованное, снаружи — дорисованное
        Y,X=np.mgrid[0:h,0:w].astype(np.float32)
        S_in=synth(ly,X,Y); wq=np.clip(np.minimum(np.minimum(X,w-1-X),np.minimum(Y,h-1-Y))/FE,0,1)[...,None]
        Q=ly['P']*wq+S_in*(1-wq)
        Yc,Xc=np.mgrid[0:cv.H,0:cv.W].astype(np.float32)
        S_c=synth(ly,(Xc-cv.ox)/cv.k,(Yc-cv.oy)/cv.k)
        cov=np.clip(cv.warp(np.ones((h,w),np.float32)),0,1)[...,None]
        s.plate=cv.warp(Q)+S_c*(1-cov); del S_c,Yc,Xc
        # подпись — поверх кругов
        sg_rgb,sg_a=ly['sig']; s.sig=cv.warp(sg_rgb*sg_a[...,None]); s.siga=np.clip(cv.warp(sg_a),0,1)
        # спрайты
        dv_rgb,dv_a=ly['diver']; ys,xs=np.where(dv_a>.5)
        s.c0=np.array([(xs.min()+xs.max())/2,(ys.min()+ys.max())/2])           # опора — середина тела
        s.diver=Sprite.cut(dv_rgb,dv_a,tuple(s.c0),3)
        i=np.argmax(xs); s.tip=np.array([xs[i],ys[i]],float)-s.c0               # кончики рук — впереди
        j=np.argmin(xs); s.tail=np.array([xs[j],ys[j]],float)-s.c0
        # лифчик: держится за левый конец бретельки; запас по краям — под волну флага
        tp_rgb,tp_a=ly['top']; ys,xs=np.where(tp_a>.02)
        x0,x1,y0,y1=xs.min()-3,xs.max()+4,ys.min()-14,ys.max()+15
        s.bra_pm=np.dstack([tp_rgb[y0:y1,x0:x1]*tp_a[y0:y1,x0:x1,None],tp_a[y0:y1,x0:x1]]).astype(np.float32)
        ys,xs=np.where(tp_a>.3); i=np.argmin(xs); s.braP=np.array([xs[i],ys[i]],float)
        s.bra_anc=(s.braP[0]-x0,s.braP[1]-y0)
        s.HOOK=np.array([557.,258.])                                           # вершина бурой горы
        s.RC=ly['tone_sea']*.5+ly['tone_sky']*.5                                # круги
        s.DC=ly['tone_sea']*.15+ly['tone_paper']*.85                              # брызги
        s.yhc=cv.oy+YH*cv.k
        # нырок: опора по параболе, поворот — по касательной
        s.Dx=150.; s.a=45.; s.b=(s.YS1+40-s.c0[1])+s.a; s.ang0=np.degrees(np.arctan2(s.a,s.Dx))
        tt=np.arange(.45,4.0,1/600)
        tips=np.array([s.dive(t)[0][:2]+rot(s.tip,s.dive(t)[1]) for t in tt])
        i=np.argmax(tips[:,1]>=s.YS1); s.te=tt[i]; s.xe=tips[i,0]
        # прыжок дельфином: из-под воды по баллистической дуге и снова в воду
        s.tl0,s.LD=5.0,2.4; s.xa=130.; s.Dxl=260.; s.yap=410.; s.Dyl=(s.YS2+130)-s.yap
        tt=np.arange(s.tl0,s.tl0+s.LD*1.3,1/600)
        tips=np.array([s.leap(t)[0]+rot(s.tip,s.leap(t)[1]) for t in tt])
        i=np.argmax(tips[:,1]<=s.YS2); s.tm=tt[i]; s.xm=tips[i,0]
        top=np.argmin(tips[:,1]); j=top+np.argmax(tips[top:,1]>=s.YS2); s.tr=tt[j]; s.xr=tips[j,0]
        # круги: (центр x, вода y, начало, сила, срок жизни)
        s.rings=[(s.xe,s.YS1,s.te+d,A,3.7-d*.5) for d,A in [(0,1),(.3,.85),(.62,.7),(1.0,.5)]]
        s.rings+=[(s.xm,s.YS2,s.tm-.75,.35,1.4)]                                  # пузырь перед выныриванием
        s.rings+=[(s.xm,s.YS2,s.tm+d,A,2.6-d*.4) for d,A in [(0,.7),(.35,.5)]]
        s.rings+=[(s.xr,s.YS2,s.tr+d,A,3.6-d*.4) for d,A in [(0,1),(.3,.85),(.62,.65)]]
        rng=np.random.default_rng(58)
        def drops(n,x,y,t0,up,sp,bias):
            th=np.radians(rng.uniform(-up,up,n))+bias; v=rng.uniform(*sp,n)
            return dict(x=x+rng.uniform(-6,6,n),y=y,t0=t0+rng.uniform(0,.12,n),vx=v*np.sin(th),vy=-v*np.cos(th),
                        r=rng.uniform(1.8,4.6,n))
        s.drops=[drops(18,s.xe,s.YS1,s.te,38,(150,360),0.),drops(14,s.xm,s.YS2,s.tm,30,(140,320),np.radians(12)),
                 drops(18,s.xr,s.YS2,s.tr,38,(150,360),np.radians(8))]
        s.foam=[(s.xe,s.YS1,s.te),(s.xm,s.YS2,s.tm),(s.xr,s.YS2,s.tr)]

    def dive(s,t):
        """нырок: (опора x, y), поворот°"""
        u=max(t-.45,0)/1.5; p=u**2
        x=s.c0[0]+s.Dx*p; y=s.c0[1]-s.a*p+s.b*p*p
        ang=(np.degrees(np.arctan2(-(-s.a+2*s.b*p),s.Dx))-s.ang0)*.9
        return np.array([x,y]),ang

    def leap(s,t):
        """дельфин: x равномерно, y — парабола (та же сила тяжести), поворот по касательной"""
        q=(t-s.tl0)/s.LD
        x=s.xa+s.Dxl*q; y=s.yap+s.Dyl*(2*q-1)**2
        ang=np.degrees(np.arctan2(-4*s.Dyl*(2*q-1),s.Dxl))
        return np.array([x,y]),ang

    def bra(s,t):
        """лифчик: (точка бретельки x, y), поворот°, сила волны. Долгий полёт большой дугой → на вершине флагом"""
        P,H,fb=s.braP,s.HOOK,8.
        t0,t1=.2,3.0
        u=np.clip((t-t0)/(t1-t0),0,1); r=.3*u+.7*ease(u)                       # уже летит — и с разгона цепляется
        C1,C2=np.array([110.,-170.]),np.array([730.,-150.])
        p=(1-r)**3*P+3*(1-r)**2*r*C1+3*(1-r)*r*r*C2+r**3*H
        ang=-1080*r+fb*r                                                        # три оборота в полёте
        A=(3.5*np.sin(np.pi*u)+.5)*ramp(t,0,.4)
        if t>t1:
            d=t-t1; ang=fb+28*np.exp(-2.3*d)*np.sin(2*np.pi*d/1.)                # зацепился и качнулся
            Af=ramp(t,t1,t1+.8)*(1+.45*np.sin(2*np.pi*t/3.3))                  # треплется на ветру, порывами
            ang+=Af*(5*np.sin(2*np.pi*t/.9)+2*np.sin(2*np.pi*t/.37))
            A=max(A*(1-ramp(t,t1,t1+.3)),4*Af)
        return p,ang,A

    def bra_sprite(s,t,A):
        pm=s.bra_pm
        if A>.05:
            h,w=pm.shape[:2]; u=np.arange(w,dtype=np.float32); ua=s.bra_anc[0]
            dy=A*np.clip((u-ua)/40,0,1)*np.sin(2*np.pi*(u-ua)/55-2*np.pi*2.3*t)
            mx=np.broadcast_to(u[None,:],(h,w)).astype(np.float32)
            my=(np.arange(h,dtype=np.float32)[:,None]-dy[None,:]).astype(np.float32)
            pm=cv2.remap(pm,mx,my,cv2.INTER_LINEAR,borderValue=0)
        sp=Sprite.__new__(Sprite); sp.pm=pm; sp.off=(0,0); sp.anc=s.bra_anc; return sp

    def water_grid(s):
        """мировые координаты воды (X, Z) для каждой точки холста ниже горизонта — один раз"""
        cv=s.cv; f=s.F_PERSP; y0=int(np.floor(s.yhc))+1
        yc=(np.arange(y0,cv.H,dtype=np.float32)-cv.oy)/cv.k; xc=(np.arange(cv.W,dtype=np.float32)-cv.ox)/cv.k
        Z=(f/np.maximum(yc-YH,.05))[:,None]; X=(xc[None,:]-s.XC)*Z/f
        s.wg=(y0,X.astype(np.float32),np.broadcast_to(Z,X.shape).astype(np.float32)); s.wx=xc

    def water(s,img,t):
        """круги на воде: полоса шириной dR вокруг окружности R в плоскости воды; покрытие пикселя — аналитически
        по градиенту расстояния (тонкий дальний край просто бледнеет, без пунктира)"""
        cv=s.cv; f=s.F_PERSP
        if not hasattr(s,'wg'): s.water_grid()
        y0,X,Z=s.wg; A=None
        for xe,ys,t0,amp,life in s.rings:
            age=t-t0
            if age<=0 or age>=life: continue
            u=age/life
            if life<2: R=.15+.6*age; dR=.035
            else: R=.25+2.4*age+.35*age*age; dR=.045+.012*age
            al=amp*.9*ramp(age,0,.12)*(1-ramp(u,.35,1))*(1-ramp(R,2.4,4.8))
            if al<=.004: continue
            Ze=f/(ys-YH); Xe=(xe-s.XC)*Ze/f
            dx=X-Xe; dz=Z-Ze; d=np.sqrt(dx*dx+dz*dz)+1e-6
            g=Z/(f*cv.k)*np.sqrt((dx/d)**2+((dx*X+dz*Z)/d)**2)          # |∇d| на пиксель холста
            hw=np.minimum(dR/2,4.5*cv.k*g)                                  # вблизи не толще ~9 px картины
            c=np.clip((hw-np.abs(d-R))/g+.5,0,1)
            c=c*(1-ramp(np.abs(s.wx-xe),420,672))[None,:]                   # круги — в пределах ~2 ширин картины
            a=c*al; A=a if A is None else np.maximum(A,a)
        if A is None: return img
        yy=np.arange(y0,cv.H,dtype=np.float32); A=A*np.clip((yy-s.yhc-2*cv.k)/(14*cv.k),0,1)[:,None]
        sub=img[y0:]; img[y0:]=sub*(1-A[...,None])+s.RC*A[...,None]
        return img

    def splash(s,img,t):
        cv=s.cv; m=np.zeros((cv.H,cv.W),np.uint8); any_=False
        for (xe,ys,t0) in s.foam:
            age=t-t0
            if 0<age<.9:
                rx=(14+40*eout(age/.9))*cv.k; ry=rx*.2; al=1-ramp(age,.2,.9)
                cx,cy=cv.xy(xe,ys)
                mm=np.zeros_like(m)
                cv2.ellipse(mm,(int(cx*16),int(cy*16)),(int(rx*16),int(ry*16)),0,0,360,255,-1,cv2.LINE_AA,4)
                m=np.maximum(m,(mm*al).astype(np.uint8)); any_=True
        for d in s.drops:
            age=t-d['t0']; ok=age>0
            x=d['x']+d['vx']*age; y=d['y']+d['vy']*age+450*age*age
            ok&=(y<=d['y']+1)
            for xi,yi,ri in zip(x[ok],y[ok],d['r'][ok]):
                cx,cy=cv.xy(xi,yi); cv2.circle(m,(int(cx*16),int(cy*16)),max(int(ri*cv.k*16),12),255,-1,cv2.LINE_AA,4)
                any_=True
        if not any_: return img
        a=m.astype(np.float32)/255; return img*(1-a[...,None])+s.DC*a[...,None]

    def frame(s,t):
        cv=s.cv; img=s.plate.copy()
        img=s.water(img,t)
        img=over(img,s.sig,s.siga)
        # прыгунья: нырок, под водой, прыжок дельфином
        if t<s.tl0: (pos,ang),ys=s.dive(t),s.YS1
        else: (pos,ang),ys=s.leap(t),s.YS2
        L=put_layer(cv,s.diver,pos[0],pos[1],1.,ang)
        if L is not None:
            y0,x0,w=L; hh,ww=w.shape[:2]
            ysc=cv.oy+ys*cv.k
            clip=np.clip(ysc-(y0+np.arange(hh)),0,1)[:,None,None]
            w=w*clip; a=np.clip(w[...,3:4],0,1)
            img[y0:y0+hh,x0:x0+ww]=img[y0:y0+hh,x0:x0+ww]*(1-a)+w[...,:3]
        img=s.splash(img,t)
        p,ang,A=s.bra(t)
        img=put(cv,img,s.bra_sprite(t,A),p[0],p[1],1.,ang)
        return np.clip(img+.5,0,255).astype(np.uint8)

if __name__=='__main__': cli(Scene,'s58')
