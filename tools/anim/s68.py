# «В унисон» (работа 68). Нотный стан вытягивается до краёв экрана и колышется, по нему вправо за край прыгают
# ноты — маленькие копии головок; две фигуры-головки покачиваются в такт, синхронно. Зелёная на долю сбивается
# (качается не туда, нота-«крышка» перекашивается) и торопливо догоняет. В конце всё встаёт на место — повтор бесшовный.
import numpy as np, cv2
from lib import *

SRC='site/images/work-68.jpg'
LINE_RGB=np.array([38,38,36],np.float32)
BGC=np.array([241,238,224],np.float32)                          # бумага фото (для разбора)
BG=(214,140,130)                                                # фон сцены: тёплый коралловый вместо бумаги
SEEDS=[[(100,274)],[(100,323)],[(100,385)],[(100,439)],[(100,504),(372,432),(600,365)]]       # линии стана сверху вниз (x, y)
BEAM=np.array([(334,118),(500,82),(500,127),(334,163)],np.int32)  # сдвоенное ребро ноты
HEADC=[(300.8,439.0),(443.6,426.4)]                             # центры головок: синяя (левая), зелёная (правая)
TOPY=[592,582]                                                  # верх тела: синяя, зелёная
AXIS=288.5                                                      # ось синей фигуры
BOT=900                                                         # тела обрываются нижним краем картины
STEM=(40.5,-12,290)                                            # штиль от центра головки: сдвиг x, y, длина
P=0.6                                                           # доля, с (100 ударов в минуту)

def track(g,dark,skip):
    H,W=g.shape; out=[]
    for seeds in SEEDS:
        tr={}
        for x0,y0 in seeds:
            for dirn in (1,-1):
                y=float(y0); x=x0; miss=0
                while 0<=x<W:
                    lo,hi=int(round(y))-5,int(round(y))+6
                    if skip[lo:hi,x].any(): break                     # головка/штиль — дальше другая затравка
                    col=g[lo:hi,x]; ys=np.arange(lo,hi)
                    w=np.clip(170-col,0,None); w[~dark[lo:hi,x]]=0
                    if w.sum()>40: y=(w*ys).sum()/w.sum(); tr[x]=y; miss=0
                    else:
                        miss+=1
                        if miss>4: break
                    x+=dirn
        xs=np.array(sorted(tr)); ys=np.array([tr[x] for x in xs])
        xf=np.arange(0,W,dtype=np.float64); yf=np.interp(xf,xs,ys)       # за головками — по прямой между краями
        yf=cv2.GaussianBlur(yf.reshape(1,-1),(0,0),4).ravel()
        out.append((xf,yf))
    return out

INK=np.array([40,40,38],np.float32)                            # тёмный контур картины

def soft(d,core,grow=2,a=None,fill=False):
    """альфа куска: силуэт (заливка + тёмный контур) — 1, наружная кайма контура — по яркости между бумагой и чернилами"""
    ring=cv2.dilate(core.astype(np.uint8),np.ones((2*grow+1,2*grow+1),np.uint8))>0
    L=a.mean(2); Lp=float(np.median(L[~ring][::50]))
    sil=ring&(L<Lp-100)                                         # только то, что темнее бумаги: без светлой каймы
    if fill:
        ff=np.pad(sil.astype(np.uint8),1); cv2.floodFill(ff,None,(0,0),2); sil=ff[1:-1,1:-1]!=2
    al=np.clip((Lp-L-12)/(Lp-60),0,1)*ring; al[al<.08]=0; al[sil]=1
    return al.astype(np.float32)

def layers():
    a=load_work(SRC); H,W,_=a.shape; g=a.mean(2)
    d=np.abs(a-BGC).sum(2); nb=d>55
    dark=(g<150)&(np.ptp(a,2)<45)
    heads=cv2.morphologyEx(dark.astype(np.uint8),cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)))
    n,lab=cv2.connectedComponents(heads)
    hl=lab==lab[int(HEADC[0][1]),int(HEADC[0][0])]; hr=lab==lab[int(HEADC[1][1]),int(HEADC[1][0])]
    hl=cv2.dilate(hl.astype(np.uint8),np.ones((3,3),np.uint8))>0; hr=cv2.dilate(hr.astype(np.uint8),np.ones((3,3),np.uint8))>0
    bm=np.zeros((H,W),np.uint8); cv2.fillPoly(bm,[BEAM],1); bm=bm>0
    beam=dark&bm
    vert=cv2.morphologyEx(dark.astype(np.uint8),cv2.MORPH_OPEN,np.ones((25,1),np.uint8))>0
    vcore=vert; vert=cv2.dilate(vert.astype(np.uint8),np.ones((1,3),np.uint8))>0
    sl=np.zeros((H,W),bool); sl[100:445,332:350]=True; sl&=vert&~bm
    sr=np.zeros((H,W),bool); sr[80:432,482:498]=True; sr&=vert&~bm
    L=hl|sl; R=hr|sr
    note=L|R|beam
    skip=cv2.dilate((hl|hr).astype(np.uint8),np.ones((5,5),np.uint8))>0
    blue=(a[...,2]>a[...,0]+50); green=(a[...,1]>a[...,0]+40)&(a[...,1]>a[...,2]+20)
    def body(m):
        m=cv2.morphologyEx(m.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((15,15),np.uint8))
        n,lab,st,_=cv2.connectedComponentsWithStats(m); m=lab==1+np.argmax(st[1:,4])
        ff=m.astype(np.uint8).copy(); ff=np.pad(ff,1); cv2.floodFill(ff,None,(0,0),2); m=(ff[1:-1,1:-1]!=2)
        return m
    bb,gb=body(blue),body(green)
    gcore=cv2.dilate(gb.astype(np.uint8),np.ones((7,7),np.uint8))>0
    bcore=cv2.dilate(bb.astype(np.uint8),np.ones((7,7),np.uint8))>0
    sig=np.zeros((H,W),bool); sig[820:895,610:745]=nb[820:895,610:745]; sig&=~gcore
    Lines=track(g,dark,skip|gcore|bcore)
    lm=np.zeros((H,W),np.uint8)
    for xf,yf in Lines: cv2.polylines(lm,[np.round(np.stack([xf,yf],1)*16).astype(np.int32)],False,255,8,cv2.LINE_8,4)
    hole=cv2.dilate((nb|(lm>0)|note|gcore|bcore).astype(np.uint8)*255,np.ones((7,7),np.uint8))
    bg=cv2.inpaint(np.clip(a,0,255).astype(np.uint8),hole,9,cv2.INPAINT_TELEA).astype(np.float32)
    tone=np.median(bg.reshape(-1,3),0); lit=tone-cv2.GaussianBlur(bg,(0,0),40)
    a=a+lit
    # синяя под зелёной: правый край синей — зеркало её левого края (ось 288,5 по плечам), с контуром;
    # при рассинхроне из-под зелёной показывается настоящий скруглённый бок, а не срез
    bal=soft(d,bb,3,a,True)
    bal[green]=0
    ba=a.copy(); bin_=cv2.erode(bb.astype(np.uint8),np.ones((9,9),np.uint8))>0
    La=a.mean(2); Lp=float(np.median(La[::7,::7])); prev=None
    for y in range(TOPY[0],H):
        gx=np.where(green[y])[0]; bx=np.where(blue[y])[0]
        if len(gx)==0 or len(bx)==0: continue
        xl,xb,e0=bx.min(),bx.max(),gx.min()
        if xb<e0-12 or (La[y,xb:e0]>Lp-60).any(): prev=None; continue  # между ними бумага — край синей виден сам
        xr=2*AXIS-xl
        xr=xb+3 if prev is None else min(xr,prev+1.0)                # от видимого края к зеркальному плавно, без ступеньки
        prev=xr; xr=int(round(xr))
        for dst in range(xb-2,xr+6):                                 # спрятанная часть — зеркало левого бока (с тенью и контуром)
            src=xl+(xr-dst); ba[y,dst]=a[y,src]; bal[y,dst]=bal[y,src]
    gal=soft(d,gb,3,a,True)
    lal,ral,beal=soft(d,L,2,a)*~skip_other(L,R|beam),soft(d,R,2,a)*~skip_other(R,L|beam),soft(d,beam,2,a)
    stub=(lm>0)&~vcore&~cv2.dilate((hl|hr).astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)&~bm
    lal[stub]=0; ral[stub]=0                                   # обрывки линий стана у штилей остаются стану
    # маленькая нота: копия левой головки (штиль дорисовывается линией)
    sal=soft(d,hl,2,a)*~skip_other(hl,sl)
    return dict(a=a,ba=ba,tone=tone,lines=Lines,bal=bal,gal=gal,lal=lal,ral=ral,beal=beal,
                sig=np.clip((d-12)/60,0,1).astype(np.float32)*sig,sal=sal)

def clean(rgb,al,paper):
    """кайма без старой бумаги: там, где альфа < 1, цвет — чернила контура (кайма — это край тёмного контура)"""
    al=al.astype(np.float32)
    return np.where((al<1)[...,None],INK,rgb).astype(np.float32),al

def skip_other(m,o):
    return (cv2.dilate(o.astype(np.uint8),np.ones((3,3),np.uint8))>0)&~m

class Layer:
    """неподвижный кусок картины на холсте (цвет×альфа, альфа) в рамке с запасом mg px картины"""
    def __init__(L,cv,rgb,al,mg):
        pm=cv.warp(np.dstack([rgb*al[...,None],al]))
        ys,xs=np.where(pm[...,3]>.002); m=int(mg*cv.k)
        L.x0,L.y0=max(xs.min()-m,0),max(ys.min()-m,0); L.x1,L.y1=min(xs.max()+m+1,cv.W),min(ys.max()+m+1,cv.H)
        L.pm=np.ascontiguousarray(pm[L.y0:L.y1,L.x0:L.x1])
        L.X,L.Y=np.meshgrid(np.arange(L.x0,L.x1,dtype=np.float32),np.arange(L.y0,L.y1,dtype=np.float32))
    def draw(L,img,mx=None,my=None):
        """mx, my — откуда (абсолютные px холста) брать каждую точку рамки; None — на месте"""
        w=L.pm if mx is None else cv2.remap(L.pm,np.float32(mx-L.x0),np.float32(my-L.y0),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        a=np.clip(w[...,3:4],0,1); r=img[L.y0:L.y1,L.x0:L.x1]; img[L.y0:L.y1,L.x0:L.x1]=r*(1-a)+w[...,:3]
        return img

def rig_inv(X,Y,c,th,T):
    """обратное к «повернуть на th вокруг c, сдвинуть на T» (холст): откуда брать точку (X, Y)"""
    dx,dy=X-c[0]-T[0],Y-c[1]-T[1]; co,si=np.cos(th),np.sin(th)
    return (c[0]+co*dx+si*dy).astype(np.float32),(c[1]-si*dx+co*dy).astype(np.float32)

class Scene:
    T=12.0; work=68
    def __init__(s,geo):
        ly=layers(); h,w=ly['a'].shape[:2]
        s.cv=cv=Canvas(geo,h,w); s.tone=np.float32(BG); pp=ly['tone']
        s.plate=cv.flat(s.tone)
        # подпись: её отличие от бумаги переносится на новый фон (бледная подпись остаётся бледной)
        sa=cv.warp(ly['sig']); s.plate=s.plate+sa[...,None]*cv.warp(np.clip(ly['a']-pp,-255,0)*(ly['sig']>0)[...,None])
        s.xr=cv.x_right+80; s.xl=cv.x_left-80
        # компьютер: слева текст сайта — стан тает за 120…320 px картины от рамы, через текст не идёт
        X=np.arange(cv.W,dtype=np.float32)
        s.lf=(ease((X-(cv.ox-320*cv.k))/(200*cv.k)) if GEO[geo]['box'][0]>0 else np.ones_like(X))[None,:]
        s.L=[]
        for xf,yf in ly['lines']:
            slr=(yf[-1]-yf[-25])/24; sll=(yf[24]-yf[0])/24
            xe=np.arange(xf[-1]+1,s.xr+1); ye=yf[-1]+slr*260*(1-np.exp(-(xe-xf[-1])/260))
            xb=np.arange(s.xl,xf[0]); yb=yf[0]-sll*260*(1-np.exp(-(xf[0]-xb)/260))
            s.L.append(dict(x=np.concatenate([xb,xf,xe]),y=np.concatenate([yb,yf,ye]),x0=xf[0],xe=xf[-1]))
        s.blue=Layer(cv,*clean(ly['ba'],ly['bal'],pp),40); s.green=Layer(cv,*clean(ly['a'],ly['gal'],pp),40)
        s.nl=Layer(cv,*clean(ly['a'],ly['lal'],pp),60); s.nr=Layer(cv,*clean(ly['a'],ly['ral'],pp),60)
        s.nb=Layer(cv,*clean(ly['a'],ly['beal'],pp),60)
        s.small=Sprite.cut(*clean(ly['a'],ly['sal'],pp),HEADC[0])
        rng=np.random.default_rng(68)
        # ноты: (время появления, из какой головки, мелодия — позиции на стане по прыжкам)
        s.notes=[]
        for j,t0 in enumerate(np.arange(2.4,6.1,P)):
            hi=j%2; p=[4.0 if hi==0 else 3.6]
            for _ in range(24): p.append(float(np.clip(p[-1]+rng.choice([-1.5,-1,-.5,.5,1,1.5]),-0.5,3.5)))
            s.notes.append(dict(t0=t0,hi=hi,p=p))

    # --- стан ---
    def wave(s,i,x,t):
        A=win(t,1.0,3.0,9.0,11.0)
        g=.3+.7*ease(np.abs(x-380)/420)
        return A*g*(7*np.sin(2*np.pi*(x/420-t/(4*P))+i*.6)+1.2*np.sin(2*np.pi*(x/170+t/(2*P))+i*1.9))
    def ly(s,i,x,t): l=s.L[i]; return np.interp(x,l['x'],l['y'])+s.wave(i,x,t)
    def staff(s,p,x,t):
        """y точки стана: p — номер линии (дробный = между линиями), x px картины"""
        i=int(np.clip(np.floor(p),0,3)); f=p-i
        return s.ly(i,x,t)*(1-f)+s.ly(i+1,x,t)*f

    # --- качание ---
    def sway(s,t):
        """(сдвиг верха тела, подъём) для синей и зелёной"""
        A=win(t,1.2,2.6,9.2,10.8); out=[]
        ph=(t-1.2)/(2*P)                                     # циклов: туда-обратно за две доли
        lag=.5*ramp(t,5.4,5.75)*(1-ramp(t,7.2,7.75))         # зелёная отстаёт на долю и догоняет
        for k,q in enumerate((ph,ph-lag)):
            sx=17*A*np.sin(2*np.pi*q)
            b=6*A*(.5-.5*np.cos(4*np.pi*q))                  # на каждой доле тянется вверх
            out.append((sx,b))
        return out

    def body_map(s,Lr,top,sx,b):
        cv=s.cv; yp=(Lr.Y-cv.oy)/cv.k
        y=(yp+BOT*b/(BOT-top))/(1+b/(BOT-top))
        gy=np.clip((BOT-y)/(BOT-top),0,None)**2
        return (Lr.X-sx*gy*cv.k).astype(np.float32),(cv.oy+y*cv.k).astype(np.float32)

    def head_rig(s,k,sx,b):
        cv=s.cv; top=TOPY[k]; c=HEADC[k]; up=top-c[1]
        th=.3*np.arctan(2*sx/(BOT-top))
        T=((sx*(1+.6*up/(BOT-top)))*cv.k,-b*cv.k)
        return cv.xy(*c),th,T

    def note_pos(s,n,t):
        """маленькая нота: (x, y, масштаб, поворот) или None"""
        u=t-n['t0']
        if u<0: return None
        hop=P/2; k=int(u//hop); f=u/hop-k
        if k>=len(n['p'])-1: return None
        c=HEADC[n['hi']]
        X=lambda v: c[0]+30*v+14*v*v                          # разгон: прыжки всё длиннее
        x=X(k+ease(f))
        p=n['p'][k]+(n['p'][k+1]-n['p'][k])*ease(f)
        y=s.staff(p,x,t)-(30+6*min(k,8))*np.sin(np.pi*f)*(k>0 or f>.5)
        if k==0 and f<.5: y=c[1]+(s.staff(p,x,t)-c[1])*ease(f*2)
        sc=.5*eout(u/.35)
        rot=-14*np.sin(np.pi*f)*(1 if n['p'][k+1]<n['p'][k] else -.6)
        if x>s.xr+60: return None
        return x,y,sc,rot

    def frame(s,t):
        cv=s.cv; img=s.plate.copy()
        ext=win(t,1.3,3.4,9.2,11.4)
        polys=[]
        for i,l in enumerate(s.L):
            xs=l['x'][(l['x']>=l['x0']-ext*(l['x0']-s.xl))&(l['x']<=l['xe']+ext*(s.xr-l['xe']))][::2]
            polys.append(np.stack([xs,s.ly(i,xs,t)],1))
        stems=[]; qs=[q for q in (s.note_pos(n,t) for n in s.notes) if q]
        for x,y,sc,rot in qs:                                 # штиль маленькой ноты — такой же линией, как стан
            r=np.radians(rot); v=np.array([[STEM[0],STEM[1]],[STEM[0],STEM[1]-STEM[2]]])*sc
            stems.append(np.stack([x+v[:,0]*np.cos(r)+v[:,1]*np.sin(r),y-v[:,0]*np.sin(r)+v[:,1]*np.cos(r)],1))
        a=lines_alpha(cv,polys,3.0)*s.lf; img=img*(1-a[...,None])+LINE_RGB*a[...,None]
        if stems: a=lines_alpha(cv,stems,2.2); img=img*(1-a[...,None])+LINE_RGB*a[...,None]
        for x,y,sc,rot in qs: img=put(cv,img,s.small,x,y,sc,rot)
        sw=s.sway(t); still=all(abs(v)<1e-4 for p in sw for v in p)
        if still:
            for Lr in (s.nb,s.nl,s.nr,s.blue,s.green): img=Lr.draw(img)
        else:
            (c0,t0,T0),(c1,t1,T1)=s.head_rig(0,*sw[0]),s.head_rig(1,*sw[1])
            # ребро: левый конец идёт за синей головкой, правый — за зелёной
            Lr=s.nb; w=np.clip((s.cv.xy(492,0)[0]-Lr.X)/(147*cv.k),0,1)
            ax,ay=rig_inv(Lr.X,Lr.Y,c0,t0,T0); bx,by=rig_inv(Lr.X,Lr.Y,c1,t1,T1)
            img=Lr.draw(img,ax*w+bx*(1-w),ay*w+by*(1-w))
            img=s.nl.draw(img,*rig_inv(s.nl.X,s.nl.Y,c0,t0,T0))
            img=s.nr.draw(img,*rig_inv(s.nr.X,s.nr.Y,c1,t1,T1))
            img=s.blue.draw(img,*s.body_map(s.blue,TOPY[0],*sw[0]))
            img=s.green.draw(img,*s.body_map(s.green,TOPY[1],*sw[1]))
        return np.clip(img+.5,0,255).astype(np.uint8)

if __name__=='__main__': cli(Scene,'s68')
