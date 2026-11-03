# «Она ждала дождя» (работа 26). Треугольнички, висевшие в воздухе, срываются и падают дождём, сверху добавляются
# новые — дождь по всему экрану, но над женщиной сухо: капли отскакивают от полей шляпы. Дождь стихает, исходные
# треугольнички опускаются сверху, тормозят и зависают на своих местах. Начало мягкое: первые ~0,8 с вздрагивают
# только два треугольничка, дождь набирает силу после. Бумага картины заменена фоном BG.
import numpy as np, cv2
from lib import *

BG=(214,184,104)             # фон сцены: приглушённый горчично-охристый — и вокруг картины, и вместо бумаги
LINE=np.array([30,34,30],np.float32)    # тёмный контур рисунка
INK=np.array([80,80,70],np.float32)   # тон карандашной подписи (её штрихи тоньше пикселя — смешаны с бумагой)

SRC='site/images/work-26.jpg'
HAT_RGB=np.array([78,95,111],np.float32)
G=3000.                      # ускорение падения, px картины/с²
SUB=4                        # шагов физики на кадр

def matte(a,paper,M):
    """очистить края куска от старой бумаги: цвет края — ближайший цвет изнутри, альфа — доля этого цвета в пикселе"""
    core=cv2.erode(M.astype(np.uint8),np.ones((3,3),np.uint8))>0
    band=(cv2.dilate(M.astype(np.uint8),np.ones((3,3),np.uint8))>0)&~core
    src=(~core).astype(np.uint8)
    _,lab=cv2.distanceTransformWithLabels(src,cv2.DIST_L2,5,labelType=cv2.DIST_LABEL_PIXEL)
    cz=np.argwhere(src==0)                                       # метки идут по строкам, как argwhere
    yy,xx=cz[np.clip(lab-1,0,len(cz)-1)].transpose(2,0,1)
    fg=a[yy,xx]
    # пиксель края = бумага + aL·(контур − бумага) + aF·(цвет изнутри − бумага): тонкий тёмный контур не теряется
    u=LINE-paper; v=fg-paper; b=a-paper
    uu=(u*u).sum()+50; vv=(v*v).sum(2)+50; uv=(v*u).sum(2); bu=(b*u).sum(2); bv=(b*v).sum(2)
    det=uu*vv-uv*uv; aL=np.clip((bu*vv-bv*uv)/det,0,None); aF=np.clip((bv*uu-bu*uv)/det,0,None)
    sm=aL+aF; f=np.where(sm>1,1/np.maximum(sm,1e-6),1.); aL*=f; aF*=f; sm=aL+aF
    ec=(aL[...,None]*LINE+aF[...,None]*fg)/np.maximum(sm,1e-6)[...,None]
    al=np.where(core,1.,np.where(band,np.clip((sm-.06)/.94,0,1),0.))
    rgb=np.where(core[...,None],a,ec)
    return rgb.astype(np.float32),al.astype(np.float32)

def layers():
    """картина → тон, женщина с подписью (альфа), шляпа (маска), треугольнички (спрайты)"""
    a=load_work(SRC); H,W,_=a.shape
    bgc=np.median(a.reshape(-1,3),0)
    d=np.abs(a-bgc).sum(2); nb=d>55
    n,lab,st,cen=cv2.connectedComponentsWithStats(nb.astype(np.uint8))
    big=[i for i in range(1,n) if st[i,4]>800]
    wi=max(big,key=lambda i:st[i,4]); tris=[i for i in big if i!=wi]
    wm=lab==wi
    sig=np.zeros((H,W),bool); sig[840:900,550:699]=nb[840:900,550:699]   # подпись остаётся с женщиной
    # шляпа: тёмно-сизая часть женщины выше лица, с контуром, без дыр
    hat=wm&(np.abs(a-HAT_RGB).sum(2)<70); hat[440:]=False
    hat=cv2.morphologyEx(hat.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    n2,l2,s2,_=cv2.connectedComponentsWithStats(hat); hat=(l2==1+np.argmax(s2[1:,4])).astype(np.uint8)
    cnt,_=cv2.findContours(hat,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); hat=np.zeros_like(hat)
    cv2.drawContours(hat,cnt,-1,1,-1); hat=cv2.dilate(hat,np.ones((5,5),np.uint8))   # с контуром
    # фон под рисунком: залить — по нему тон бумаги и свет фото
    hole=cv2.dilate(nb.astype(np.uint8)*255,np.ones((7,7),np.uint8))
    bg=cv2.inpaint(np.clip(a,0,255).astype(np.uint8),hole,9,cv2.INPAINT_TELEA).astype(np.float32)
    tone=np.median(bg.reshape(-1,3),0); lit=tone-cv2.GaussianBlur(bg,(0,0),40)
    a=a+lit
    wrgb,wal=matte(a,tone,wm)
    # подпись: альфа — доля карандаша в пикселе
    sz=cv2.dilate(sig.astype(np.uint8),np.ones((5,5),np.uint8))>0; v=INK-tone
    sa=np.clip(((a-tone)*v).sum(2)/(v*v).sum(),0,1)*sz; sa[sa<.04]=0
    wrgb=np.where((sa>wal)[...,None],INK,wrgb); wal=np.maximum(wal,sa)
    sp=[]
    for i in tris:
        trgb,ta=matte(a,tone,lab==i)
        cx,cy=cen[i]; sp_=Sprite.cut(trgb,ta,(cx,cy))
        y0,y1=st[i,1],st[i,1]+st[i,3]
        sp_.home=(cx,cy); sp_.foot=y1-cy; sp_.head=cy-y0; sp_.hw=st[i,2]/2
        sp.append(sp_)
    return dict(a=wrgb,wa=wal,tone=tone,hat=hat,tris=sp)

class Scene:
    T=12.0; work=26
    def __init__(s,geo):
        ly=layers(); h,w=ly['a'].shape[:2]
        s.cv=cv=Canvas(geo,h,w); s.tone=np.float32(BG); s.plate=cv.flat(s.tone); s.geo=geo
        s.top=cv.warp(ly['a']*ly['wa'][...,None]); s.topa=np.clip(cv.warp(ly['wa']),0,1)
        s.tris=ly['tris']; s.hw,s.hh=w,h
        # шляпа: верх полей и тульи y_top(x) и низ y_bot(x) по столбцам — по ним капли отскакивают
        hm=ly['hat']>0; cols=np.where(hm.any(0))[0]; s.hx0,s.hx1=cols[0],cols[-1]
        yt=np.array([np.argmax(hm[:,x]) for x in cols],np.float64); yb=np.array([hm.shape[0]-1-np.argmax(hm[::-1,x]) for x in cols],np.float64)
        s.htop=cv2.GaussianBlur(yt.reshape(1,-1),(0,0),2).ravel(); s.hbot=yb
        s.hslope=np.gradient(s.htop); s.hat_cx=cols.mean()
        s.sim()

    # ---- дождь: все капли считаем заранее, по кадрам ----
    def sim(s):
        cv=s.cv; rng=np.random.default_rng(26); N=int(round(s.T*FPS))
        xl,xr,yt,yb=cv.x_left-60,cv.x_right+60,cv.y_top,cv.y_bottom
        drops=[]                                  # (спрайт, масштаб, t старта, x, y, vy, vt, фаза)
        # исходные: срываются с мест по очереди
        # первыми (с начала сцены вздрагивают, срываются к ~0,85 с) — розовый над шляпой и тёмный под полями
        hm=[np.hypot(tr.home[0]-x,tr.home[1]-y) for tr in s.tris for x,y in [(391,135)]]
        first=[int(np.argmin(hm)),int(np.argmin([np.hypot(tr.home[0]-465,tr.home[1]-539) for tr in s.tris]))]
        rest=[i for i in rng.permutation(len(s.tris)) if i not in first]
        T0={first[0]:.85,first[1]:1.0}
        for j,i in enumerate(rest): T0[i]=1.3+j*0.1+rng.uniform(0,.06)
        for i in first+rest:
            tr=s.tris[i]; t0=T0[i]
            drops.append(dict(sp=i,sc=1.,ts=t0,x=tr.home[0],y=tr.home[1],vy=0.,vt=rng.uniform(1250,1450),ph=rng.uniform(0,6.3),orig=1))
        s.t_go={d['sp']:d['ts'] for d in drops}
        # новые сверху: поток по ширине экрана; у текстовой колонки реже
        R=0.030; t=1.0; dt=1/240
        while t<8.2:
            I=ramp(t,1.2,3.2)*(1-ramp(t,6.4,8.0))   # дождь набирает силу после первых срывов
            k=rng.poisson(R*(xr-xl)*I*dt)
            for _ in range(k):
                x=rng.uniform(xl,xr); xc=cv.xy(x,0)[0]
                dens=1. if s.geo=='m' else .22+.78*ease((xc/cv.W-.36)/.18)
                if rng.random()>dens: continue
                i=rng.integers(len(s.tris)); sc=rng.uniform(.5,.95); tr=s.tris[i]
                vt=rng.uniform(1150,1500)
                drops.append(dict(sp=i,sc=sc,ts=t,x=x,y=yt-(tr.foot+tr.head)*sc-40,vy=vt,vt=vt,ph=rng.uniform(0,6.3),orig=0))
            t+=dt
        n=len(drops); s.drops=drops
        X=np.array([d['x'] for d in drops]); Y=np.array([d['y'] for d in drops]); VY=np.array([d['vy'] for d in drops])
        VX=np.zeros(n); A=np.zeros(n); Wsp=np.zeros(n); VT=np.array([d['vt'] for d in drops])
        TS=np.array([d['ts'] for d in drops]); SC=np.array([d['sc'] for d in drops])
        FOOT=np.array([s.tris[d['sp']].foot for d in drops])*SC; HEAD=np.array([s.tris[d['sp']].head for d in drops])*SC
        pos=np.full((N,n,3),np.nan,np.float32); done=np.zeros(n,bool)
        def inhat(px,py):
            ix=np.clip(np.round(px).astype(int)-s.hx0,0,len(s.htop)-1)
            pen=py-s.htop[ix]; ok=(px>=s.hx0)&(px<=s.hx1)&(pen>0)&(py<=s.hbot[ix]+2)
            return ok,ix,pen
        h=1/(FPS*SUB)
        for f in range(N):
            for sub in range(SUB):
                t=f/FPS+sub*h
                act=(TS<=t)&~done
                if not act.any(): continue
                VY[act]+=h*G*(1-VY[act]*np.abs(VY[act])/VT[act]**2)
                VX[act]*=np.exp(-.7*h); Wsp[act]*=np.exp(-.4*h)
                X[act]+=VX[act]*h; Y[act]+=VY[act]*h; A[act]+=Wsp[act]*h
                # столкновение со шляпой: нижний конец, центр и верх капли
                th=np.radians(A)
                for L in (FOOT,np.zeros(n),-HEAD):
                    px=X+np.sin(th)*L; py=Y+np.cos(th)*L
                    hit,ix,pen=inhat(px,py); hit&=act
                    if not hit.any(): continue
                    sl=s.hslope[ix[hit]]; nn=np.hypot(sl,1); ex,ey=sl/nn,-1/nn          # нормаль наружу (вверх)
                    Y[hit]-=pen[hit]
                    vn=VX[hit]*ex+VY[hit]*ey; tx,ty=-ey,ex; vtg=VX[hit]*tx+VY[hit]*ty
                    imp=vn<-150; k=hit.sum()
                    vn=np.where(vn<0,-.42*vn,vn); vtg=.8*vtg
                    vx=vn*ex+vtg*tx; vy=vn*ey+vtg*ty
                    side=np.sign(X[hit]-s.hat_cx+1e-3)
                    vx=vx+imp*side*rng.uniform(140,300,k)
                    vx=np.where(side*vx<120,side*120,vx)                                   # всегда прочь от середины шляпы
                    VX[hit]=vx; VY[hit]=np.where(imp,np.minimum(vy,-rng.uniform(120,260,k)),vy)
                    Wsp[hit]=np.where(imp,-side*rng.uniform(250,600,k),Wsp[hit])
                    act=act&~hit
            out=(Y-HEAD-60>yb)|(X<xl-200)|(X>xr+200)
            done|=out&(TS<=f/FPS)
            live=(TS<=f/FPS)&~done
            pos[f,live,0]=X[live]; pos[f,live,1]=Y[live]; pos[f,live,2]=A[live]
        s.pos=pos; s.SC=SC; s.TS=TS
        # возвращение исходных: сверху, с ветерком справа, тормозят и зависают на месте
        s.ret=[]
        for i,tr in enumerate(s.tris):
            r0=7.5+0.12*((i*7)%10)+rng.uniform(0,.1); r1=r0+rng.uniform(2.0,2.3)
            D=tr.home[1]-(yt-(tr.foot+tr.head)-60)
            s.ret.append((r0,r1,D,rng.uniform(0,6.3)))

    def ret_pos(s,i,t):
        tr=s.tris[i]; r0,r1,D,ph=s.ret[i]
        if t<r0: return None
        u=np.clip((t-r0)/(r1-r0),0,1); q=(1-u)**3                    # остаток пути: тормозит к месту
        x=tr.home[0]+7.5*np.sqrt(D*q); y=tr.home[1]-D*q
        ang=10*np.sqrt(q)*np.sin(2*np.pi*t/1.3+ph)+1.4*np.sin(2*np.pi*t/0.9+ph)*win(t,r0,r1-1,r1-.3,r1+.5)
        return x,y,ang

    def frame(s,t):
        cv=s.cv; img=s.plate.copy(); f=int(round(t*FPS))
        fade=None
        if s.geo=='m': fade=lambda y: 1-ramp(cv.xy(0,y)[1],1000,1800)
        # исходные на месте (до срыва), с дрожью перед ним — и после возвращения
        for i,tr in enumerate(s.tris):
            tg=s.t_go[i]
            if t<tg:
                tw=(ramp(t,0,.3) if tg<1.2 else ramp(t,tg-.45,tg-.25))*(1-ramp(t,tg-.15,tg))
                ang=tw*2.5*np.sin(2*np.pi*t/.16+i); dy=-tw*3*np.sin(np.pi*ramp(t,tg-.25,tg))
                img=put(cv,img,tr,tr.home[0],tr.home[1]+dy,1.,ang)
            p=s.ret_pos(i,t)
            if p is not None:
                x,y,ang=p; img=put(cv,img,tr,x,y,1.,ang,1. if fade is None else fade(y))
        if 0<=f<len(s.pos):
            P=s.pos[f]
            for j in np.where(~np.isnan(P[:,0]))[0]:
                d=s.drops[j]; x,y,a=P[j]
                al=1. if fade is None else fade(y)
                if al<=0: continue
                wob=1.2*np.sin(2*np.pi*t/.8+d['ph'])*(ramp(t,d['ts'],d['ts']+.4) if d['orig'] else 1)
                img=put(cv,img,s.tris[d['sp']],x,y,s.SC[j],a+wob,al)
        img=over(img,s.top,s.topa)
        return np.clip(img+.5,0,255).astype(np.uint8)

if __name__=='__main__': cli(Scene,'s26')
