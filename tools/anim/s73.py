# «Через монокль» (работа 73). Монокль — окно: море с лодкой есть только по ту сторону стекла.
# Такт 1: лодка поднимает парус и уплывает за край стекла — в стекле одно пустое море.
# Такт 2: монокль на шнурке бросается в погоню по экрану (море скользит сквозь стекло — оно на месте, едет только
#         окно), почти догоняет, лодка снова ускользает, рывок — поймал.
# Такт 3: монокль «тащит» пойманную лодку домой и сам встаёт на место; лодка на своём месте, волны идут дальше.
import numpy as np, cv2
from lib import *

SRC='site/images/work-73-2.jpg'
LC=(568.5,436.8)                 # центр стекла (px фото 1400×1400)
RING=(808.,465.)                 # центр ушка
ATT=(838.,465.)                  # где шнурок выходит из ушка
CORD0=(842.,467.)                # затравка шнурка
SAIL=[(587,272),(462,433),(592,440)]
MAST=[(571,428),(589,428),(586,466),(570,466)]
HULL_IN=[(493,459),(657,470),(639,501),(511,501)]   # корпус целиком (в фото нижний левый угол скрыт волной)
BOAT_ANC=(575.,501.)             # опора лодки — середина днища
RS=215
DARK=np.array([48,40,46],np.float32)
BG=(120,160,140)                 # фон сцены вместо голубой панели — шалфейно-зелёный
SEA_Y=478.                       # линия моря в мире (px фото)
SINK=18.                         # днище ниже гребня

def polymask(H,W,pts,grow=0):
    m=np.zeros((H,W),np.uint8); cv2.fillPoly(m,[np.array(pts,np.int32)],255)
    if grow: m=cv2.dilate(m,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*grow+1,2*grow+1)))
    return m>0

def light(a):
    """свет фото по голубой панели: многочлен 3-й степени по (x, y)"""
    H,W,_=a.shape; g=a.mean(2); Y,X=np.mgrid[0:H,0:W]
    r=np.hypot(X-699.5,Y-699.5); rl=np.hypot(X-LC[0],Y-LC[1]); rr=np.hypot(X-RING[0],Y-RING[1])
    m=(g>175)&(r<672)&(rl>222)&(rr>45)&(a[...,2]>a[...,0]+8)
    m=cv2.erode(m.astype(np.uint8),np.ones((9,9),np.uint8))>0
    m[1000:1420,1000:1340]=False                                   # подпись
    ys,xs=np.where(m); sel=np.arange(0,len(xs),7); xs,ys=xs[sel],ys[sel]
    def basis(x,y):
        x=(x-700)/700.; y=(y-700)/700.
        return np.stack([np.ones_like(x),x,y,x*x,x*y,y*y,x**3,x*x*y,x*y*y,y**3],-1)
    B=basis(xs.astype(np.float64),ys.astype(np.float64))
    fit=np.zeros((H,W,3),np.float32); Bf=basis(X.astype(np.float64),Y.astype(np.float64))
    for c in range(3):
        co=np.linalg.lstsq(B,a[ys,xs,c].astype(np.float64),rcond=None)[0]; fit[...,c]=Bf@co
    return fit

def trace_cord(g):
    """центр шнурка от ушка до свободного конца: шаг 3 px по самому тёмному направлению"""
    p=np.array(CORD0,np.float64); d=np.array([1.,0.]); pts=[np.array(ATT),p.copy()]
    for _ in range(3000):
        best=None
        for da in np.radians(np.arange(-50,51,5)):
            c,s=np.cos(da),np.sin(da); dd=np.array([c*d[0]-s*d[1],s*d[0]+c*d[1]])
            q=p+dd*4; x,y=int(round(q[0])),int(round(q[1]))
            if not(0<=x<g.shape[1] and 0<=y<g.shape[0]): continue
            v=160-g[y,x]
            if v>0 and (best is None or v*np.cos(da)**.5>best[0]): best=(v*np.cos(da)**.5,dd)
        if best is None: break
        d=best[1]; p=p+d*3
        # подтянуть к середине линии поперёк
        n=np.array([-d[1],d[0]]); ws=[]; os_=np.arange(-6,7)
        for o in os_:
            q=p+n*o; x,y=int(round(q[0])),int(round(q[1])); ws.append(max(0,150-g[y,x]))
        ws=np.array(ws,np.float64)
        if ws.sum()<50: break
        p=p+n*(ws*os_).sum()/ws.sum()*.5; pts.append(p.copy())
    P=np.array(pts)
    for _ in range(3): P[1:-1]=(P[:-2]+P[1:-1]*2+P[2:])/4
    return P

def layers():
    """фото → тон, свет, обод стекла (по углам), рамка, лодка, подпись, шнурок, цвета моря"""
    a0=load_work(SRC); H,W,_=a0.shape; g=a0.mean(2)
    fit=light(a0)
    Y,X=np.mgrid[0:H,0:W]; rpan=np.hypot(X-699.5,Y-699.5)
    tone=np.median(fit[rpan<650].reshape(-1,3),0)
    a=a0+(tone-fit)                                                # свет фото снят
    # --- обод стекла по углам: внутренний и внешний край
    ang=np.arange(0,360,1.); rin=np.zeros_like(ang); rout=np.zeros_like(ang)
    for i,t in enumerate(np.radians(ang)):
        rs=np.arange(175,225,.5); xs=LC[0]+rs*np.cos(t); ys=LC[1]+rs*np.sin(t)
        col=cv2.remap(a,xs.astype(np.float32).reshape(1,-1),ys.astype(np.float32).reshape(1,-1),cv2.INTER_LINEAR)[0]
        dk=(col.mean(1)<88)&(col[:,2]<82)
        runs=[]; st=None
        for j,v in enumerate(np.r_[dk,False]):
            if v and st is None: st=j
            if not v and st is not None: runs.append((st,j)); st=None
        runs=[r for r in runs if r[1]-r[0]>=5]
        if runs:
            j0,j1=min(runs,key=lambda r:abs(rs[(r[0]+r[1])//2]-202)); rin[i]=rs[j0]; rout[i]=rs[j1-1]+.5
        else: rin[i]=rout[i]=np.nan
    for v in (rin,rout):
        bad=np.isnan(v); v[bad]=np.interp(ang[bad],ang[~bad],v[~bad],period=360)
    def smooth_c(v,s=4):
        v=np.concatenate([v[-30:],v,v[:30]]).reshape(1,-1).astype(np.float64)
        return cv2.GaussianBlur(v,(0,0),s).ravel()[30:-30]
    def medf(v,k=4): return np.array([np.median(np.take(v,range(i-k,i+k+1),mode='wrap')) for i in range(len(v))])
    rin=smooth_c(medf(rin),2.5); rout=smooth_c(medf(rout),2.5)
    rout=np.clip(rout,rin+5,rin+12)
    rout=np.clip(rout,rin+5,rin+12)
    # --- цвета моря и неба из фото
    def med(x,y,r=4): return np.median(a[y-r:y+r+1,x-r:x+r+1].reshape(-1,3),0)
    pink=np.median(a[(np.abs(a0-[243,184,178]).sum(2)<40)&(np.hypot(X-LC[0],Y-LC[1])<190)].reshape(-1,3),0)
    sea=[med(450,490),med(450,530),med(620,507)]             # тёмно-синий, светло-голубой, синий
    hullc=np.median(a[(np.abs(a0-[137,117,120]).sum(2)<25)&polymask(H,W,HULL_IN)].reshape(-1,3),0)
    # --- лодка: парус и мачта из фото, корпус — заново целиком (заливка + контур)
    reg=(polymask(H,W,SAIL,3)|polymask(H,W,MAST,1))&(Y<463)
    vis=np.clip((np.abs(a-pink).sum(2)-25)/60,0,1)*reg
    S=8; hm=np.zeros((H,W),np.uint8); q=np.round(np.array(HULL_IN,np.float32)*S).astype(np.int32)
    cv2.fillPoly(hm,[q],255,cv2.LINE_AA,3); hf=hm.astype(np.float32)/255
    ol=np.zeros((H,W),np.uint8); cv2.polylines(ol,[q],True,255,4,cv2.LINE_AA,3); olf=ol.astype(np.float32)/255
    brgb=a*vis[...,None]/np.maximum(vis[...,None],1e-3)
    brgb=brgb*(1-hf[...,None])+hullc*hf[...,None]; brgb=brgb*(1-olf[...,None])+DARK*olf[...,None]
    ba=np.maximum(vis*(1-hf),np.maximum(hf,olf))
    boat=Sprite.cut(brgb,ba.astype(np.float32),BOAT_ANC,3)
    # --- рамка: обод + ушко (поворачивается), из фото
    fx0,fy0,fx1,fy1=int(LC[0]-RS),int(LC[1]-RS),int(RING[0]+40),int(LC[1]+RS)
    fa=a[fy0:fy1,fx0:fx1]; fY,fX=np.mgrid[fy0:fy1,fx0:fx1]
    fr=np.hypot(fX-LC[0],fY-LC[1]); fth=(np.degrees(np.arctan2(fY-LC[1],fX-LC[0])))%360
    frin=np.interp(fth,ang,rin,period=360); frout=np.interp(fth,ang,rout,period=360)
    rmid=(frin+frout)/2
    dk=np.clip((np.abs(fa-tone).sum(2)-20)/(np.abs(tone-DARK).sum()-40),0,1)
    ringzone=(np.hypot(fX-RING[0],fY-RING[1])<33.5)&(fr>rmid)
    fal=np.where(fr<rmid,np.clip(fr-frin+.5,0,1),dk*((fr<frout+5)|ringzone))
    # край обода и ушка снаружи смешан со старой голубой панелью — вычесть её (иначе голубой ореол на новом фоне)
    fpm=np.clip(fa-(1-fal[...,None])*tone,0,None); fpm=np.minimum(fpm,fal[...,None]*120)
    fun=fpm/np.maximum(fal[...,None],1e-3)
    frgb=np.where((fr<frin+1)[...,None],DARK,np.where((fr<rmid)[...,None],fa,fun))
    frame=Sprite(frgb.astype(np.float32),fal.astype(np.float32),(fx0,fy0),LC)
    # --- подпись
    sm=np.zeros((H,W),bool); sm[1000:1420,1030:1320]=True; sm&=rpan<688
    sal=np.clip((np.abs(a-tone).sum(2)-18)/70,0,1)*sm
    # --- шнурок
    cord=trace_cord(g)
    cm=np.zeros((H,W),np.uint8); cv2.polylines(cm,[np.round(cord).astype(np.int32)],False,255,3)
    ccol=np.median(a[(cm>0)&(g<90)].reshape(-1,3),0)
    return dict(tone=tone,pink=pink,sea=sea,boat=boat,frame=frame,ang=ang,rin=rin,
                sig=np.minimum(np.clip(a-(1-sal[...,None])*tone,0,None),sal[...,None]*255).astype(np.float32),sal=sal.astype(np.float32),cord=cord,ccol=ccol,H=H,W=W)

def hermite(keys,t):
    """кривая через ключи (t, x, скорость): кубический Эрмит; вне ключей — крайние значения"""
    if t<=keys[0][0]: return keys[0][1]
    for (t0,x0,v0),(t1,x1,v1) in zip(keys,keys[1:]):
        if t<=t1:
            h=t1-t0; u=(t-t0)/h
            return (2*u**3-3*u*u+1)*x0+(u**3-2*u*u+u)*h*v0+(-2*u**3+3*u*u)*x1+(u**3-u*u)*h*v1
    return keys[-1][1]

def osc(t,u,w,z):
    x=np.zeros_like(t); v=0.; dt=t[1]-t[0]
    for i in range(1,len(t)):
        acc=-w*w*x[i-1]-2*z*w*v+u[i-1]; v+=acc*dt; x[i]=x[i-1]+v*dt
    return x

def softmax_hi(v,hi,m):
    """плавный потолок: до hi−m как есть, дальше мягко подходит к hi (непрерывно с производной)"""
    v=float(v); a=hi-m
    return v if v<=a else hi-m*np.exp(-(v-a)/m)

class PatchCv:
    """кусок холста (для put): та же картина, начало в (px0, py0)"""
    def __init__(p,cv,px0,py0,w,h): p.k=cv.k; p.ox=cv.ox-px0; p.oy=cv.oy-py0; p.W=w; p.H=h
    def xy(p,x,y): return p.ox+np.asarray(x)*p.k, p.oy+np.asarray(y)*p.k

# волны мира: (средняя высота, [(амплитуда, длина, период с (минус — влево), фаза)])
rng=np.random.default_rng(73)
WAVES=[(SEA_Y,[(7.,330.,2.6,.3),(3.,140.,-1.9,1.1)])]
for k in range(1,12):
    WAVES.append((SEA_Y+37*k+rng.uniform(-5,5),[(rng.uniform(9,17),rng.uniform(240,460),rng.uniform(2.2,3.4),rng.uniform(0,6.3)),
                                               (rng.uniform(3,6),rng.uniform(110,190),-rng.uniform(1.4,2.2),rng.uniform(0,6.3))]))

def wave_y(k,x,t):
    b,cs=WAVES[k]; y=np.full_like(np.asarray(x,np.float64),b); d=np.zeros_like(y)
    for a,l,P,f in cs:
        ph=2*np.pi*(x/l-t/P)+f; y=y+a*np.sin(ph); d=d+a*2*np.pi/l*np.cos(ph)
    return y,d

# сюжет. view — какую точку мира (x, px фото) показывает центр стекла, boat — где лодка в мире: ключи (t, x, скорость).
# Монокль на экране: x = sx·view, y — по ключам dy; масштаб до scmax в погоне (приближается к зрителю), вид в стекле
# увеличен вместе с ним. По вертикали вид идёт за стеклом (горизонт всегда в стекле), по горизонтали — окно в мир.
VIEW=[(0,0,0),(1.9,-40,0),(3.2,700,650),(3.7,980,300),(4.0,1030,0),(4.3,1050,0),(5.0,1900,500),(5.35,2030,0),
      (5.7,1985,0),(8.6,450,-650),(10.3,-25,0),(11.0,0,0)]
BOAT=[(0,575,0),(1.0,610,120),(2.2,900,400),(3.1,1450,500),(3.7,1700,220),(4.0,1740,60),(4.5,2150,800),
      (5.0,2480,350),(5.4,2560,0)]
PATH={'d':dict(sxk=[(0,0,0),(1.9,-40,0),(3.2,800,300),(3.7,880,0),(4.0,700,-400),(4.3,600,0),(5.0,900,300),
                      (5.35,930,0),(5.7,900,0),(8.6,300,-300),(10.3,-15,0),(11.0,0,0)],scmax=1.8,cq=.75,dy=[(0,0,0),(1.9,-55,0),(3.2,-700,0),(4.0,-300,500),(4.3,150,900),(5.0,1300,400),
                                    (5.35,1150,0),(5.7,1100,0),(8.6,300,-400),(10.3,0,0),(11.0,0,0)]),
      'm':dict(sx=.13,scmax=1.4,cq=.5,dy=[(0,0,0),(1.9,-30,0),(3.2,-200,0),(4.0,-60,200),(4.3,60,350),(5.0,430,150),
                                    (5.35,390,0),(5.7,370,0),(8.6,100,-150),(10.3,0,0),(11.0,0,0)])}
PATH['d2']=PATH['d']

class Scene:
    T=11.0; work=73
    def __init__(s,geo):
        ly=layers(); s.ly=ly; H,W=ly['H'],ly['W']
        s.cv=cv=Canvas(geo,H,W); s.tone=np.float32(BG)
        s.plate=over(cv.flat(s.tone),cv.warp(ly['sig']),np.clip(cv.warp(ly['sal']),0,1))
        s.P=PATH[geo]; s.th=None
        tt=np.arange(0,s.T+1e-6,1/240); s.tt=tt
        px=np.array([hermite(VIEW,t) for t in tt]); vx=np.gradient(px,tt); ax=np.gradient(vx,tt)
        sw=osc(tt,ax,2*np.pi/1.4,.3); sw=sw/np.abs(sw).max()
        s.th=-(6*sw+4*vx/np.abs(vx).max())                    # наклон монокля: по скорости и по инерции, °
        c=ly['cord']; sg=np.r_[0,np.cumsum(np.hypot(*np.diff(c,axis=0).T))]; s.cs=sg/sg[-1]
        tg=np.gradient(c,axis=0); tg/=np.linalg.norm(tg,axis=1,keepdims=True)+1e-9; s.cn=np.stack([-tg[:,1],tg[:,0]],1)
        s.rmax=float(ly['rin'].max())

    def view(s,t): return hermite(VIEW,t)
    def mon(s,t):
        """монокль на экране: смещение центра стекла (px фото), масштаб, наклон°"""
        dx=hermite(s.P['sxk'],t) if 'sxk' in s.P else s.P['sx']*s.view(t); dy=hermite(s.P['dy'],t)+4*np.sin(2*np.pi*t/2.75)
        sc=1+(s.P['scmax']-1)*win(t,2.0,3.6,6.0,10.3)
        # весь монокль с ушком — в видимой части страницы даже на ноутбуке (замер живой страницы): в d x ≤ 2320,
        # y 260…1380; в d2 — ×4/3; на телефоне всё и так влезает
        cv=s.cv; X,Y=cv.xy(LC[0]+dx,LC[1]+dy)
        if cv.geo in ('d','d2'):
            f=1. if cv.geo=='d' else 4/3; kk=cv.k*sc
            X=softmax_hi(X,2320*f-(RING[0]+31-LC[0])*kk,30)
            Y=softmax_hi(Y,1380*f-209*kk,30); Y=-softmax_hi(-Y,-(260*f+209*kk),30)
        dx=(X-cv.ox)/cv.k-LC[0]; dy=(Y-cv.oy)/cv.k-LC[1]
        th=0. if s.th is None else float(np.interp(t,s.tt,s.th))+1.5*np.sin(2*np.pi*t/2.2+.5)-7*win(t,1.85,2.0,2.05,2.45)
        return dx,dy,sc,th

    def boat_x(s,t):
        if t<=5.4: return hermite(BOAT,t)
        r0=hermite(BOAT,5.4)-BOAT_ANC[0]-s.view(5.4)
        return BOAT_ANC[0]+s.view(t)+r0*(1-ramp(t,5.4,6.4))

    def frame(s,t):
        ly=s.ly; cv=s.cv; img=s.plate.copy()
        dx,dy,sc,th=s.mon(t); cx,cy=LC[0]+dx,LC[1]+dy
        # --- шнурок: начало ходит вместе с ушком, конец лежит на месте
        c=ly['cord']; cs=s.cs; T_=np.radians(th)
        def rig(p):
            q=p-np.array(LC); r=np.stack([np.cos(T_)*q[:,0]+np.sin(T_)*q[:,1],-np.sin(T_)*q[:,0]+np.cos(T_)*q[:,1]],1)
            return r*sc+np.array([cx,cy])
        # шнурок волочится за моноклем: точка s повторяет сдвиг и масштаб монокля с запаздыванием 0,2·s с
        lagt=t-0.2*cs; ms=np.array([s.mon(q)[:3] for q in lagt[::8]])
        Dx,Dy,Sc=[np.interp(cs,cs[::8],ms[:,i]) for i in range(3)]
        cq=s.P.get('cq',1.); Sx=Sc*(1-(1-cq)*(Sc-1)/(s.P['scmax']-1)*ramp(cs,.04,.3))      # на телефоне шнурок уже, чтобы не уходил за край
        P=np.stack([LC[0]+Dx,LC[1]+Dy],1)+(c-np.array(LC))*np.stack([Sx,Sc],1)
        e=(.12*ease(cs))[:,None]; P=P*(1-e)+c*e
        b=(1-ramp(cs,0,.10))[:,None]; P=rig(c)*b+P*(1-b)
        sp=np.hypot(*(np.array(s.mon(t+.05)[:2])-np.array(s.mon(t-.05)[:2])))/0.1
        amp=np.clip(sp/16,0,25)*np.sin(np.pi*cs)
        P=P+s.cn*(amp*np.sin(2*np.pi*(1.6*cs-t/1.3)))[:,None]
        al=lines_alpha(cv,[P[::2]],6.4*(1+.6*(sc-1))); img=img*(1-al[...,None])+ly['ccol']*al[...,None]
        # --- вид сквозь стекло: мир (небо, лодка, море) рисуется прямо в px экрана и обрезается стеклом
        Mx,My=cv.xy(cx,cy); R=(s.rmax+3)*cv.k*sc
        x0,y0=max(int(Mx-R)-2,0),max(int(My-R)-2,0); x1,y1=min(int(Mx+R)+3,cv.W),min(int(My+R)+3,cv.H)
        if x1>x0 and y1>y0:
            w,h=x1-x0,y1-y0; pc=PatchCv(cv,x0,y0,w,h); kk=cv.k*sc; vx0=LC[0]+s.view(t)
            pc.k=kk; pc.ox=Mx-x0-vx0*kk; pc.oy=My-y0-LC[1]*kk
            patch=np.broadcast_to(ly['pink'].astype(np.float32),(h,w,3)).copy()
            xb=s.boat_x(t); y0b,d0b=wave_y(0,np.array([xb-22,xb,xb+22]),t)
            ang=-np.degrees(np.arctan((y0b[2]-y0b[0])/44))+2.0*np.sin(2*np.pi*t/1.7+.4)
            patch=put(pc,patch,ly['boat'],xb,y0b[1]+SINK,1.,ang)
            wx=vx0+(np.arange(x0,x1)+.5-Mx)/kk; wy=(LC[1]+(np.arange(y0,y1)+.5-My)/kk)[:,None]
            for k in range(len(WAVES)):
                yk,dk=wave_y(k,wx,t); nrm=np.sqrt(1+dk*dk)[None,:]
                dd=(wy-yk[None,:])/nrm*kk                       # расстояние до кромки, px экрана
                fill=np.clip(dd+.5,0,1); line=np.clip(2.1*kk-np.abs(dd)+.5,0,1)
                col=ly['sea'][k%3]
                patch=patch*(1-fill[...,None])+col*fill[...,None]
                patch=patch*(1-line[...,None])+DARK*line[...,None]
            YY,XX=np.mgrid[y0:y1,x0:x1]; dX=XX+.5-Mx; dY=YY+.5-My
            rr=np.hypot(dX,dY)/(cv.k*sc); phi=(np.degrees(np.arctan2(dY,dX))+th)%360
            ri=np.interp(phi,ly['ang'],ly['rin'],period=360)
            a=np.clip((ri+1.2-rr)*cv.k*sc,0,1)[...,None]
            img[y0:y1,x0:x1]=img[y0:y1,x0:x1]*(1-a)+patch*a
        img=put(cv,img,ly['frame'],cx,cy,sc,th)
        return np.clip(img+.5,0,255).astype(np.uint8)

if __name__=='__main__': cli(Scene,'s73')
