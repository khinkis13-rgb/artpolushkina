# «CHEERS!» (работа 66). Застывшая в картине пробка договаривает свой хлопок: улетает за край экрана, кувыркаясь,
# бутылка от отдачи качается на головах, из горлышка фонтанчиком летят золотые капли игристого — часть падает
# в бокалы на свитерах и наполняет их, остальные улетают за низ кадра; бокалы чокаются. Ирония: пробка возвращается
# и затыкает бутылку, бокалы «выпиты» в три глотка, бутылка икает на головах. Бумага картины и фон — цвет BG.
import numpy as np, cv2
from lib import *

SRC='site/images/work-66.jpg'
BG=(224,172,166)                               # фон сцены: пыльно-розовый вместо кремовой бумаги
BGC=np.array([247,242,214],np.float32)         # бумага на фото (для разбора)
WINE=np.array([226,180,74],np.float32)        # игристое: на бордовом и синем видно, на бумаге — тоже
FOAM=np.array([250,238,196],np.float32)
# бокалы: стык чаши с ножкой (x, y) и полоса x, где искать стенки чаши
GLASSES=[dict(j=(320,761),xs=(290,353)),dict(j=(387,764),xs=(351,412)),dict(j=(481,757),xs=(447,507))]

def soft_alpha(p,col,under):
    """доля цвета col в пикселе p поверх подложки under (проекция на отрезок under→col)"""
    v=col-under; return np.clip(((p-under)*v).sum(-1)/np.maximum((v*v).sum(-1),1),0,1)

def repaper(img,tone,bg,protect):
    """заменить бумагу tone на bg без ореолов: в полосе 1 px у бумаги — доля бумаги по проекции на цвет
    самого «сильного» соседа (центр контура, тело бутылки), p' = p + (1−α)(bg − tone)"""
    dlt=np.float32(bg)-tone; d=np.abs(img-tone).sum(2); paper=(d<40)&~protect
    Z=(cv2.dilate(paper.astype(np.uint8),np.ones((5,5),np.uint8))>0)&~paper&~protect
    H,W=d.shape; best=d.copy(); C=img.copy(); pd=np.pad(d,2,mode='edge'); pi=np.pad(img,((2,2),(2,2),(0,0)),mode='edge')
    for dy in range(-2,3):
        for dx in range(-2,3):
            dd=pd[2+dy:2+dy+H,2+dx:2+dx+W]; m=dd>best; best[m]=dd[m]; C[m]=pi[2+dy:2+dy+H,2+dx:2+dx+W][m]
    v=C-tone; pv=img-tone; al=np.clip((pv*v).sum(2)/np.maximum((v*v).sum(2),1),0,1)
    res=np.linalg.norm(pv-al[...,None]*v,axis=2); al[res>30]=1         # не на отрезке бумага→сосед — не трогаем
    a=np.where(paper,0.,np.where(Z,al,1.))
    return img+(1-a)[...,None]*dlt

def head_shapes(a,bot):
    """головы под бутылкой: эллипс по видимому контуру (сбоку и сверху до шеи)"""
    R,G,B=a[...,0],a[...,1],a[...,2]
    skin=(R>195)&(G>165)&(R-G>15)&(R-B>35)
    n,lab,st,_=cv2.connectedComponentsWithStats(skin.astype(np.uint8))
    db=cv2.distanceTransform((~bot).astype(np.uint8),cv2.DIST_L2,3)
    out=[]
    for i in [i for i in range(1,n) if st[i,4]>5000]:
        m=lab==i; ys=np.where(m.any(1))[0]
        wd=np.array([np.ptp(np.where(m[y])[0]) for y in ys]); top=ys[0]
        w8=wd[:80]; im=np.argmax(w8); yn=ys[im]+np.argmax(wd[im:]<.7*w8.max())   # где голова переходит в шею
        cs,_=cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        P=max(cs,key=len)[:,0,:]
        P=P[(P[:,1]<yn)&(db[P[:,1],P[:,0]]>5)]
        el=cv2.fitEllipse(P.astype(np.float32))
        out.append(dict(m=m,el=el,cx=P[:,0].mean(),yn=yn))
    return sorted(out,key=lambda h:-h['cx'])          # сначала синяя (правая), красная — поверх

def glass_bowls(a):
    """чаши бокалов: внутренность по строкам между стенками, верх (край), стык с ножкой"""
    R,G,B=a[...,0],a[...,1],a[...,2]
    gold=(G>92)&(R>B+10)&(G<190); gold[:675]=0; gold[870:]=0
    H,W=gold.shape; res=[]
    for g in GLASSES:
        jx,jy=g['j']; x0,x1=g['xs']; rows={}
        for y in range(675,jy):
            xs=np.where(gold[y,x0:x1])[0]+x0
            if len(xs)<2: continue
            # средняя линия бокала: от середины края к стыку
            rows[y]=xs
        y0=min(rows); top=[rows[y] for y in range(y0,y0+6) if y in rows]
        cm=np.mean([ (r.min()+r.max())/2 for r in top])
        L={};Rr={}
        for y in range(y0,jy):
            if y not in rows: continue
            c=cm+(jx-cm)*(y-y0)/(jy-y0); xs=rows[y]
            l=xs[xs<c-1]; r=xs[xs>c+1]
            if len(l): L[y]=l.max()
            if len(r): Rr[y]=r.min()
        yy=np.arange(y0,jy); yl=np.array(sorted(L)); yr=np.array(sorted(Rr))
        xl=np.interp(yy,yl,[L[y] for y in yl]); xr=np.interp(yy,yr,[Rr[y] for y in yr])
        inside=np.zeros((H,W),bool)
        for y,l,r in zip(yy,xl,xr):
            if r-l>2: inside[y,int(l)+1:int(r)]=True
        rim=max(yl.min(),yr.min()); bot=jy-2
        while not inside[bot].any(): bot-=1
        res.append(dict(inside=inside,rim=rim,bot=bot,cx=(xl[-8:].mean()+xr[-8:].mean())/2,
                        top=((xl[yy>=rim][0]+xr[yy>=rim][0])/2,rim)))
    return res,gold

def layers():
    a=load_work(SRC); H,W,_=a.shape; R,G,B=a[...,0],a[...,1],a[...,2]
    nb=np.abs(a-BGC).sum(2)>55
    bot=(G>R+12)&(G<140)&(G>B-2); bot[480:]=0
    n,lab,st,_=cv2.connectedComponentsWithStats(bot.astype(np.uint8)); bot=lab==1+np.argmax(st[1:,4])
    cork=(R>80)&(R<150)&(G<80)&(B<75)&(R-G>25); cork[200:]=0; cork[:,:590]=0
    dash=(a.mean(2)<175); dash[:140]=0; dash[190:]=0; dash[:,:550]=0; dash[:,598:]=0
    # свет фото
    hole=cv2.dilate(nb.astype(np.uint8)*255,np.ones((7,7),np.uint8))
    bg=cv2.inpaint(np.clip(a,0,255).astype(np.uint8),hole,9,cv2.INPAINT_TELEA).astype(np.float32)
    tone=np.median(bg.reshape(-1,3),0); a=a+(tone-cv2.GaussianBlur(bg,(0,0),40))
    heads=head_shapes(a,bot)
    skin_all=np.zeros((H,W),bool)
    for h in heads: skin_all|=h['m']
    SK=np.median(a[skin_all],0)
    ring=(cv2.dilate(skin_all.astype(np.uint8),np.ones((5,5),np.uint8))>0)&~skin_all&(a.mean(2)<110)
    OL=np.median(a[ring],0)
    # бутылка: альфа по проекции на её цвет — над бумагой или над кожей
    CB=np.median(a[cv2.erode(bot.astype(np.uint8),np.ones((5,5),np.uint8))>0],0)
    zone=cv2.dilate(bot.astype(np.uint8),np.ones((5,5),np.uint8))>0
    nearskin=cv2.dilate(skin_all.astype(np.uint8),np.ones((9,9),np.uint8))>0
    under=np.where(nearskin[...,None],SK,tone)
    ba=np.where(zone,soft_alpha(a,CB,under),0); ba[cv2.erode(bot.astype(np.uint8),np.ones((3,3),np.uint8))>0]=1
    brgb=np.where((ba>.97)[...,None],a,CB)
    # пробка и чёрточки полёта — над бумагой
    CK=np.median(a[cv2.erode(cork.astype(np.uint8),np.ones((3,3),np.uint8))>0],0)
    cz=cv2.dilate(cork.astype(np.uint8),np.ones((5,5),np.uint8))>0
    ca=np.where(cz,soft_alpha(a,CK,tone),0); crgb=np.where((ca>.97)[...,None],a,CK)
    dz=cv2.dilate(dash.astype(np.uint8),np.ones((5,5),np.uint8))>0
    DK=np.array([40,38,36],np.float32)
    da=np.where(dz,soft_alpha(a,DK,tone),0); drgb=np.broadcast_to(DK,a.shape)
    # неподвижный слой: под бутылкой, пробкой и чёрточками — бумага и дорисованные верхушки голов
    rem=cv2.dilate((zone|cz|dz).astype(np.uint8),np.ones((3,3),np.uint8))>0
    # бумага внутри картины — ровным тоном, как вокруг (иначе фактура фото читается прямоугольником)
    nb=nb.copy(); nb[:4]=0; nb[:,:4]=0; nb[:,-4:]=0                  # кромка фото
    n,lab,st,_=cv2.connectedComponentsWithStats(nb.astype(np.uint8))   # соринки на бумаге (подпись не трогаем)
    junk=[i for i in range(1,n) if st[i,4]<25 and not (st[i,0]>600 and st[i,1]>800)]
    nb&=~np.isin(lab,junk)
    nb&=~(cv2.dilate((cz|dz).astype(np.uint8),np.ones((9,9),np.uint8))>0)
    keep=cv2.GaussianBlur(cv2.dilate(nb.astype(np.uint8),np.ones((7,7),np.uint8)).astype(np.float32),(0,0),2)
    base=tone+(a-tone)*keep[...,None]; base[rem]=tone
    # головы отдельным слоем (струи проходят за ними): видимое как есть + дорисованное под бутылкой
    hz=(cv2.dilate(skin_all.astype(np.uint8),np.ones((9,9),np.uint8))>0)&~rem
    ha=np.where(hz,soft_alpha(a,OL,tone),0); ha[cv2.dilate(skin_all.astype(np.uint8),np.ones((3,3),np.uint8))>0]=1
    ha[~hz]=0; hrgb=np.where(((ha<.97)&hz)[...,None],OL,a)       # кромка контура — чистым цветом, без бумаги
    S=4; x0,y0,x1,y1=250,250,520,480
    for h in heads:
        (ex,ey),(ew,eh),ang=h['el']
        fill=np.zeros(((y1-y0)*S,(x1-x0)*S),np.uint8); line=fill.copy()
        c=(int(round((ex-x0)*S*16)),int(round((ey-y0)*S*16))); ax=(int(round(ew/2*S*16)),int(round(eh/2*S*16)))
        cv2.ellipse(fill,c,ax,ang,0,360,255,-1,cv2.LINE_AA,4)
        cv2.ellipse(line,c,ax,ang,0,360,255,int(1.7*S),cv2.LINE_AA,4)
        fa=cv2.resize(np.maximum(fill,line).astype(np.float32)/255,(x1-x0,y1-y0),interpolation=cv2.INTER_AREA)
        la=cv2.resize(line.astype(np.float32)/255,(x1-x0,y1-y0),interpolation=cv2.INTER_AREA)
        # только ниже шеи — нет: эллипс выше перехода в шею; ниже — видимая голова как есть
        ys=np.arange(y0,y1)[:,None]; fa=fa*(ys<h['yn']); la=la*(ys<h['yn'])
        col=SK*(1-la[...,None])+OL*la[...,None]
        r=rem[y0:y1,x0:x1]; sr=hrgb[y0:y1,x0:x1]; sa=ha[y0:y1,x0:x1]
        na=sa+fa*(1-sa); nc=(sr*sa[...,None]*(1-fa[...,None])+col*fa[...,None])/np.maximum(na,1e-4)[...,None]
        sr[r]=nc[r]; sa[r]=na[r]
    bowls,gold=glass_bowls(a)
    gz=cv2.dilate(gold.astype(np.uint8),np.ones((3,3),np.uint8))>0
    # стенки бокалов (чтобы лечь поверх вина): доля золота над свитером
    sweat=cv2.medianBlur(np.clip(a,0,255).astype(np.uint8),9).astype(np.float32)
    GC=np.median(a[gold],0)
    ga=np.where(gz,soft_alpha(a,GC,sweat),0); ga[gold]=np.maximum(ga[gold],.85)
    # ось бутылки, горлышко, пробка
    ys,xs=np.where(bot); P=np.stack([xs,ys],1).astype(np.float64); m2=P.mean(0)
    D=np.linalg.svd(P-m2,full_matrices=False)[2][0]; D=D if D[0]>0 else -D; N=np.array([-D[1],D[0]])
    pa=(P-m2)@D; pn=(P-m2)@N; tip=pa.max(); nk=pa>tip-40
    noff=(pn[nk].min()+pn[nk].max())/2; low=np.percentile(pn[pa<pa.min()+250],99.5)
    ys,xs=np.where(cork); Q=np.stack([xs,ys],1).astype(np.float64); cc=Q.mean(0)
    cd=np.linalg.svd(Q-cc,full_matrices=False)[2][0]; cang=np.degrees(np.arctan2(-cd[1],cd[0]))%180
    clen=np.ptp((Q-cc)@D)
    ys,xs=np.where(dash); dc=np.array([xs.mean(),ys.mean()])
    body=(np.abs(a-tone).sum(2)>70)&~(cv2.dilate(skin_all.astype(np.uint8),np.ones((5,5),np.uint8))>0)
    body[:520]=0
    body=cv2.morphologyEx(body.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))>0
    base=repaper(base,tone,BG,skin_all)
    # подпись художницы: светло-серая на бумаге почти тонет в розовом — тот же штрих, но темнее
    sy,sx=slice(815,H),slice(600,W); sub=a[sy,sx]; ds=np.abs(sub-tone).sum(2)
    SG=np.median(sub[ds>np.percentile(ds,99)],0); sa=soft_alpha(sub,SG,tone)*(ds>12)
    base[sy,sx]=np.float32(BG)*(1-sa[...,None])+np.float32([92,70,70])*sa[...,None]
    return dict(a=a,base=base,tone=np.float32(BG),otone=tone,head=(hrgb,ha),body=body,bot=(brgb,ba),cork=(crgb,ca),dash=(drgb,da),bowls=bowls,gold=(a,ga),
                heads=heads,m2=m2,D=D,N=N,tip=tip,noff=noff,low=low,cc=cc,cang=cang,clen=clen,dc=dc,
                bang=np.degrees(np.arctan2(-D[1],D[0])))

def mk_sprite(cv,rgb,alpha,anchor):
    """спрайт, заранее уменьшенный до масштаба холста (без ряби фактуры при повороте); ставить с sc=1/f"""
    sp=Sprite.cut(rgb,alpha.astype(np.float32),anchor,3); ph,pw=sp.pm.shape[:2]; f=cv.k
    nw,nh=max(1,int(round(pw*f))),max(1,int(round(ph*f)))
    sp.pm=cv2.resize(sp.pm,(nw,nh),interpolation=cv2.INTER_AREA if f<1 else cv2.INTER_CUBIC)
    fx,fy=nw/pw,nh/ph; sp.off=(sp.off[0]*fx,sp.off[1]*fy); sp.anc=(anchor[0]*fx,anchor[1]*fy); sp.f=fx
    return sp

def rot(v,ang):
    """поворот вектора (px картины, y вниз) на ang° против часовой — как в put"""
    c,s=np.cos(np.radians(ang)),np.sin(np.radians(ang)); v=np.asarray(v,np.float64)
    return np.array([c*v[0]+s*v[1],-s*v[0]+c*v[1]])

def kick(t,t0,amp,per,tau):
    """толчок: затухающее качание с мгновенным стартом"""
    u=t-t0
    return 0. if u<=0 else amp*np.exp(-u/tau)*np.sin(2*np.pi*u/per)

def bump(t,t0,dur):
    u=(t-t0)/dur; return 4*u*(1-u) if 0<u<1 else 0.

def disc_mask(H,W,pts,rad,box=None):
    """круги (px холста, субпиксельно) → альфа"""
    m=np.zeros((H,W),np.uint8)
    for (x,y),r in zip(pts,rad):
        if r<=.15: continue
        cv2.circle(m,(int(round(x*16)),int(round(y*16))),max(1,int(round(r*16))),255,-1,cv2.LINE_AA,4)
    return m.astype(np.float32)/255

# налив каплями: после хлопка из горлышка бьёт фонтанчик золотых капель по параболам — часть падает в бокалы
# (левый красный, правый красный, синий; давление падает — дуги всё ниже), остальные летят вперёд за низ кадра
GRAV=1600.                                   # px картины / с²
HITS=[dict(g=0,t=1.10,h=(150,190)),dict(g=1,t=1.38,h=(105,145)),dict(g=2,t=1.66,h=(65,100))]
T_POP,T_BACK,T_PLUG,T_DRAIN,T_HIC,T_CLINK=1.0,5.85,6.7,7.05,(8.2,8.7),3.9
FULL=.8

class Scene:
    T=9.6; work=66
    def __init__(s,geo):
        ly=layers(); h,w=ly['a'].shape[:2]; s.ly=ly
        s.cv=cv=Canvas(geo,h,w); s.tone=ly['tone']
        inn=cv.warp(np.ones((h,w),np.float32))[...,None]
        s.plate=cv.flat(s.tone)*(1-inn)+cv.warp(ly['base'])
        hr,ha=ly['head']; s.hda=np.clip(cv.warp(ha),0,1); s.hdc=cv.warp(hr*ha[...,None])
        ys,xs=np.where(s.hda>0); s.hbox=(ys.min(),ys.max()+1,xs.min(),xs.max()+1)
        s.m2,s.D,s.N=ly['m2'],ly['D'],ly['N']
        s.sb=mk_sprite(cv,*ly['bot'],tuple(s.m2))
        s.sc=mk_sprite(cv,*ly['cork'],tuple(ly['cc']))
        s.sd=mk_sprite(cv,*ly['dash'],tuple(ly['dc']))
        # опоры качания: где нижний край бутылки над серединами голов
        hx=sorted(hd['el'][0][0] for hd in ly['heads'])
        def edge(x):
            q=s.m2+s.N*ly['low']; return q+s.D*((x-q[0])/s.D[0])
        s.PL,s.PR=edge(hx[0]),edge(hx[1])
        s.T0=s.m2+s.D*ly['tip']+s.N*ly['noff']              # середина среза горлышка
        s.dang=-(ly['cang']-ly['bang'])                     # доворот пробки с картины на ось бутылки
        s.plug=s.m2+s.D*(ly['tip']+.12*ly['clen'])+s.N*ly['noff']
        # куда улетает пробка: по прямой с картины до края экрана и дальше
        d=(ly['cc']-s.T0); d/=np.hypot(*d); s.cdir=d
        tx=(cv.x_right+80-ly['cc'][0])/d[0]; ty=(ly['cc'][1]-(cv.y_top-80))/-d[1]
        s.cfar=ly['cc']+d*min(tx,ty)
        # бокалы: ROI, слой стенок поверх вина
        s.bowls=ly['bowls']; a,ga=ly['gold']
        s.roi=(280,670,520,870); x0,y0,x1,y1=s.roi
        s.ga=ga[y0:y1,x0:x1]; s.garb=a[y0:y1,x0:x1]
        s.yy,s.xx=np.mgrid[y0:y1,x0:x1].astype(np.float32)
        for b in s.bowls:
            b['in']=cv2.GaussianBlur(cv2.dilate(b['inside'].astype(np.uint8),np.ones((3,3),np.uint8)).astype(np.float32),
                                    (0,0),.6)[y0:y1,x0:x1]
            ys=np.where(b['inside'].any(1))[0]; b['wid']={y:np.where(b['inside'][y])[0] for y in ys}
        b0,b1=s.bowls[0],s.bowls[1]; y=max(b0['rim'],b1['rim'])+1
        s.clink=np.array([(np.where(b0['inside'][y])[0].max()+np.where(b1['inside'][y])[0].min())/2,y-8.])
        # где капли видны: не за свитерами — кроме места над чашами бокалов (капля падает в вино)
        vis=1-ly['body'].astype(np.float32)
        for b in s.bowls:
            xs=np.where(b['inside'].any(0))[0]; vis[b['rim']-26:b['bot']+3,xs.min()-4:xs.max()+5]=1
        s.vis=np.clip(cv.warp(cv2.GaussianBlur(vis,(0,0),2.5)),0,1);
        s.vis=np.maximum(s.vis,1-inn[...,0])
        rng=np.random.default_rng(66)
        s.bub=[[(rng.random(),rng.random(),.45+.5*rng.random(),rng.random()*6.28) for _ in range(6)] for _ in s.bowls]
        # капли: (старт, точка вылета, скорость, радиус, конец полёта, бокал или -1)
        s.drops=[]; s.lands=[[] for _ in s.bowls]
        def launch(te,h,vx,r,g=-1,tgt=None):
            p0=s.F(s.T0,te); vy=-np.sqrt(2*GRAV*h)
            if tgt is None:
                yb=cv.y_bottom+30; T=(-vy+np.sqrt(vy*vy+2*GRAV*(yb-p0[1])))/GRAV
            else:
                T=np.sqrt(2*h/GRAV)+np.sqrt(2*(h+tgt[1]-p0[1])/GRAV); vx=(tgt[0]-p0[0])/T
                s.lands[g].append(te+T)
            s.drops.append(dict(te=te,p0=p0,v=np.array([vx,vy]),r=r,T=T,g=g,sp=rng.uniform(-1,1,(2,2))))
        for hh in HITS:
            b=s.bowls[hh['g']]; r_=b['wid'][b['bot']-4]
            for j in range(8):
                te=hh['t']+.08*j+rng.uniform(-.025,.025)
                tx=b['cx']+rng.uniform(-.3,.3)*(r_.max()-r_.min())
                launch(te,rng.uniform(*hh['h']),0,rng.uniform(2.3,3.1),hh['g'],(tx,b['bot']-3.))
        for j in range(16):                                    # хлопок: веер вперёд
            launch(T_POP+.01+.016*j,rng.uniform(70,280),rng.uniform(70,400),rng.uniform(2.,3.4))
        for te in np.arange(1.3,2.55,.055):                     # и дальше реже, всё ниже
            u=(te-1.3)/1.25
            launch(te+rng.uniform(-.02,.02),rng.uniform(40,200)*(1-.6*u),rng.uniform(50,300),rng.uniform(1.8,2.9)*(1-.3*u))
        for l in s.lands: l.sort()

    # --- бутылка ---
    def pose(s,t):
        th=(kick(t,T_POP,-3.4,.5,.36)+kick(t,T_PLUG,-3.0,.46,.32)
            +1.4*bump(t,T_HIC[0],.30)-.8*bump(t,T_HIC[1],.24))
        sl=0.
        for t0,A in ((T_POP,-6),(T_PLUG,-5)):
            u=(t-t0)/.11
            if u>0: sl+=A*u*np.exp(1-u)
        dy=-9*bump(t,T_HIC[0],.30)-5*bump(t,T_HIC[1],.24)
        return th,sl,dy
    def F(s,p,t):
        th,sl,dy=s.pose(t); piv=s.PL if th>0 else s.PR
        return piv+rot(np.asarray(p,np.float64)-piv,th)+s.D*sl+np.array([0,dy])

    # --- пробка: (центр, доворот°) или None ---
    def cork(s,t):
        th=s.pose(t)[0]; cc=s.ly['cc']
        plug=lambda: (s.F(s.plug,t),th+s.dang)
        if t<T_POP: return cc,0.
        if t<T_POP+1.0:
            u=(t-T_POP)/1.0; k=.72*u+.28*u*u
            return cc+(s.cfar-cc)*k,-760*k
        if t<T_BACK: return None
        if t<T_PLUG:
            u=(t-T_BACK)/(T_PLUG-T_BACK); k=.25*u*u+.75*u
            pe=s.F(s.plug,T_PLUG); far=s.T0+rot(s.cfar-s.T0,14)
            ctrl=pe+s.D*np.hypot(*(far-pe))*.5
            p=(1-k)**2*far+2*k*(1-k)*ctrl+k*k*pe
            return p,th+s.dang+620*(1-k)**2
        return plug()                                     # конец сцены — пробка в горлышке

    # --- вино ---
    def level(s,i,t):
        l=s.lands[i]; L=FULL*ease((t-l[0])/(l[-1]-l[0]+.3))
        u=np.clip((t-T_DRAIN-.07*(2-i))/.3,0,3); k=(np.floor(u)+ease(u-np.floor(u)))/3 if u<3 else 1.
        return L*(1-k)
    def tilt(s,i,t):
        return (kick(t,s.lands[i][0],.08,.55,.6)+kick(t,T_CLINK,.09 if i<2 else .05,.5,.5)
                +sum(kick(t,T_DRAIN+.07*(2-i)+.3*j,.05,.3,.15) for j in range(3)))

    def glasses(s,img,t):
        cv=s.cv; x0,y0,x1,y1=s.roi; al=np.zeros(s.xx.shape,np.float32); col=np.zeros(s.xx.shape+(3,),np.float32)
        any_=False
        for i,b in enumerate(s.bowls):
            L=s.level(i,t)
            if L<=.004: continue
            any_=True; H=b['bot']-b['rim']; ysurf=b['bot']-L*H+s.tilt(i,t)*(s.xx-b['cx'])
            d=s.yy-ysurf; a=b['in']*np.clip(d+.5,0,1)*min(1,L/.04)
            c=np.broadcast_to(WINE,col.shape).copy()
            fb=np.clip(1-d/2.4,0,1)*min(1,L/.12); c=c*(1-fb[...,None])+FOAM*fb[...,None]
            # пузырьки: со дна к поверхности
            for bx,ph,sp,wo in s.bub[i]:
                yb=b['bot']-((t*sp+ph)%1)*(b['bot']-ysurf.mean()-2)
                yi=int(np.clip(round(yb),min(b['wid']),max(b['wid']))); r=b['wid'][yi]
                xb=r.min()+1+(r.max()-r.min()-2)*(.2+.6*bx)+1.2*np.sin(t*5+wo)
                bm=np.clip(1.25-np.hypot(s.xx-xb,s.yy-yb),0,1)*(d>1.5)
                c=c*(1-bm[...,None])+FOAM*bm[...,None]
            col=col*(1-a[...,None])+c*a[...,None]; al=al+a*(1-al)
        if not any_: return img
        col=col*(1-s.ga[...,None])+s.garb*s.ga[...,None]
        rgbw=cv2.warpAffine(np.dstack([col*al[...,None],al]),np.float32([[cv.k,0,cv.ox+x0*cv.k],[0,cv.k,cv.oy+y0*cv.k]]),
                            (cv.W,cv.H),flags=cv2.INTER_AREA,borderValue=0)
        return img*(1-rgbw[...,3:4])+rgbw[...,:3]

    def streams(s,t):
        """капли и брызги (px холста): список (точки, радиусы)"""
        cv=s.cv; k=cv.k; pts=[]; rad=[]
        for d in s.drops:
            tau=t-d['te']
            if tau<=0 or tau>=d['T']: continue
            p=d['p0']+d['v']*tau+np.array([0,.5*GRAV*tau*tau]); v=d['v']+np.array([0,GRAV*tau])
            sp=np.hypot(*v); u=v/max(sp,1e-6); ln=min(sp*.012,3.2*d['r'])     # капля вытянута по ходу полёта
            for j in range(6):
                f=j/5; pts.append(tuple(cv.xy(*(p-u*ln*f)))); rad.append(d['r']*(1-.75*f)*k*min(1,tau/.04))
            # попадание: две брызги над вином
        for i,b in enumerate(s.bowls):
            for n,tl in enumerate(s.lands[i]):
                v=t-tl
                if not 0<v<.22: continue
                yl=b['bot']-s.level(i,tl)*(b['bot']-b['rim'])-1.5
                for sx in (-1,1):
                    x=b['cx']+sx*(40+25*(n%3))*v; y=yl-(150+30*(n%2))*v+GRAV*.5*v*v
                    pts.append(tuple(cv.xy(x,y))); rad.append(1.3*k*(1-v/.22))
        return pts,rad

    def dashes(s,t):
        """чёрточки удара (как на картине): хлопок и пробка, вбитая обратно"""
        out=[]
        for t0,n,ln in ((T_POP,5,14),(T_PLUG,3,10)):
            v=t-t0
            if not 0<v<.28: continue
            tip=s.F(s.T0,t); a=1-v/.28
            for j in range(n):
                d=rot(s.cdir,(j-(n-1)/2)*34+ (0 if t0==T_POP else 180))
                if t0==T_PLUG: d=rot(s.cdir,90+(j-1)*28) if j!=1 else rot(s.cdir,90)
                r0=16+70*eout(v/.28); out.append((np.stack([tip+d*r0,tip+d*(r0+ln*a)]),a))
        return out

    def frame(s,t):
        cv=s.cv; img=s.plate.copy()
        pts,rad=s.streams(t)
        if pts:
            m=disc_mask(cv.H,cv.W,pts,rad)*s.vis; img=img*(1-m[...,None])+WINE*m[...,None]
        # вино в бокалах поверх струи — струя уходит под поверхность; головы поверх струй
        img=s.glasses(img,t)
        y0,y1,x0,x1=s.hbox; a=s.hda[y0:y1,x0:x1,None]
        img[y0:y1,x0:x1]=img[y0:y1,x0:x1]*(1-a)+s.hdc[y0:y1,x0:x1]
        c=s.cork(t)
        if c is not None: img=put(cv,img,s.sc,c[0][0],c[0][1],1/s.sc.f,c[1])
        th=s.pose(t)[0]; p=s.F(s.m2,t)
        img=put(cv,img,s.sb,p[0],p[1],1/s.sb.f,th)
        # чёрточки полёта с картины: тают на хлопке
        da=1-ramp(t,T_POP,T_POP+.16)
        if da>.002:
            sh=12*ramp(t,T_POP,T_POP+.16)
            q=s.ly['dc']+s.cdir*sh; img=put(cv,img,s.sd,q[0],q[1],1/s.sd.f,0,min(1,da))
        # «дзынь»: искра там, где сходятся края красных бокалов
        v=t-T_CLINK
        if 0<v<.5:
            g=eout(v/.12)*(1-ramp(v,.18,.5)); segs=[]
            for j in range(8):
                d=rot(np.array([1.,0]),j*45+15); ln=(15 if j%2==0 else 8)*(.5+.5*eout(v/.25))
                r0=3+6*eout(v/.4); segs.append(np.stack([s.clink+d*r0,s.clink+d*(r0+ln)]))
            al=lines_alpha(cv,segs,2.2)*g; img=img*(1-al[...,None])+np.float32([255,246,206])*al[...,None]
        for seg,a in s.dashes(t):
            al=lines_alpha(cv,[seg],2.0)*a; img=img*(1-al[...,None])+np.float32([40,38,36])*al[...,None]
        return np.clip(img+.5,0,255).astype(np.uint8)

if __name__=='__main__': cli(Scene,'s66')
