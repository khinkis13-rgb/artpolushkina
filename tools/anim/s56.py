# «Последняя капля» (работа 56). Из кончика капли вверх до края экрана тянется нить; капля набухает, качается,
# город внутри трясётся всё сильнее — капля отрывается и падает через весь экран, у нижнего края — всплеск
# брызгами-столбиками. Нить пружинит, по ней сверху сползает новая капля (сначала крошечная), встаёт на место,
# нить уходит обратно в кончик — повтор бесшовный.
import numpy as np, cv2
from lib import *

SRC='site/images/work-56.jpg'
BGC=np.array([246,242,210],np.float32)
TIP=(394.6,193.0)                        # кончик капли, px картины 800×900
DROP_BOX=(250,180,550,720)               # капля целиком (x0, y0, x1, y1)
LINE_RGB=np.array([92,98,88],np.float32)  # цвет контура капли — им же нить
PAD=60
BG=(207,224,234)                         # фон сцены: светлый небесно-голубой — и вокруг, и вместо бумаги
TN=284+10                                # кадров на всех раскладках: после всплеска ~0,3 с оседают последние брызги
A1=4.0                                   # капля села на место картины
OFF=3.2                                  # сдвиг фаз набухания/отрыва

def layers():
    a=load_work(SRC); H,W,_=a.shape
    nb=np.abs(a-BGC).sum(2)>45
    # фон под рисунком: залить, по нему тон бумаги и свет фото
    hole=cv2.dilate(nb.astype(np.uint8)*255,np.ones((9,9),np.uint8))
    bg=cv2.inpaint(np.clip(a,0,255).astype(np.uint8),hole,9,cv2.INPAINT_TELEA).astype(np.float32)
    tone=np.median(bg.reshape(-1,3),0); lit=tone-cv2.GaussianBlur(bg,(0,0),40)
    a=a+lit
    d=np.abs(a-tone).sum(2)
    al=np.clip((d-14)/60,0,1)
    # капля: контур + всё, что внутри (внутри — непрозрачно, бумага тоном: нить и брызги за ней не видны)
    x0,y0,x1,y1=DROP_BOX
    ink=((a.mean(2)<175)|nb).astype(np.uint8)
    ink[:y0]=0; ink[y1:]=0; ink[:,:x0]=0; ink[:,x1:]=0
    ink=cv2.morphologyEx(ink,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    ff=ink.copy()*255; m=np.zeros((H+2,W+2),np.uint8)
    cv2.floodFill(ff,m,(5,5),128)
    fill=(ff!=128)                                       # контур и внутренность
    inner=cv2.erode(fill.astype(np.uint8),np.ones((5,5),np.uint8))>0
    edge=cv2.dilate(fill.astype(np.uint8),np.ones((5,5),np.uint8))>0
    # перекраска бумаги без ореолов: линии (контур, лесенка, подпись) — цвет чернил с альфой по проекции
    # пикселя на отрезок «бумага → чернила»; цветные заливки домов — как есть, непрозрачно
    def proj(ink):
        v=tone-ink; q=((tone-a)*v).sum(2)/(v*v).sum()
        return np.clip((q-.06)/.88,0,1).astype(np.float32)
    house=(al>.98)&(np.ptp(a,2)>24)
    house=cv2.morphologyEx(house.astype(np.uint8),cv2.MORPH_OPEN,np.ones((4,4),np.uint8))>0   # тонкая лесенка — не дом
    la=proj(LINE_RGB)
    lal=np.where(house,1.,la); lrgb=np.where(house[...,None],a,LINE_RGB)
    bgn=np.float32(BG)
    da=np.where(inner,1.,lal*edge).astype(np.float32)
    drgb=np.where(inner[...,None],lrgb*lal[...,None]+bgn*(1-lal[...,None]),lrgb)
    # вес дрожи города: 0 у контура и у кончика, 1 в глубине
    dist=cv2.distanceTransform(inner.astype(np.uint8),cv2.DIST_L2,5)
    wc=np.clip((dist-3)/14,0,1)
    # подпись — неподвижна
    reg=np.zeros((H,W),bool); reg[800:,560:]=True
    sd=d[reg]; sink=a[reg][np.argsort(sd)[-60:]].mean(0)   # чернила подписи — самые тёмные её пиксели
    sm=np.where(reg,proj(sink),0).astype(np.float32)
    return dict(tone=tone,drgb=drgb,da=da,wc=wc.astype(np.float32),sig=sink*sm[...,None],sa=sm,a=a,sink=sink)

def col_sprite(tex,w,h,lw=1.6):
    """брызга-столбик w×h px картины: заливка фактурой здания, тёмный контур"""
    S=4; W,H=int(w*S),int(h*S)
    t=cv2.resize(tex,(W,H),interpolation=cv2.INTER_LINEAR)
    m=np.zeros((H,W),np.float32); m[:]=1
    o=np.zeros((H,W),np.float32); b=int(lw*S)
    o[:b]=o[-b:]=1; o[:,:b]=o[:,-b:]=1
    rgb=t*(1-o[...,None])+LINE_RGB*o[...,None]
    rgb=cv2.resize(rgb,(int(w),int(h)),interpolation=cv2.INTER_AREA); al=np.ones(rgb.shape[:2],np.float32)
    rgb=np.pad(rgb,((2,2),(2,2),(0,0))); al=np.pad(al,2)
    return Sprite(rgb,al,(0,0),(rgb.shape[1]/2,rgb.shape[0]/2))

class Scene:
    T=10.0; work=56                                  # T уточняется по геометрии: конец — последняя брызга ушла за край
    def __init__(s,geo):
        ly=layers(); a=ly['a']; h,w=a.shape[:2]
        s.cv=cv=Canvas(geo,h,w); s.tone=np.float32(BG); s.plate=cv.flat(s.tone); s.geo=geo
        s.sig=cv.warp(ly['sig']); s.siga=np.clip(cv.warp(ly['sa']),0,1)
        # капля — спрайт в масштабе холста на трёх уровнях (1, ½, ¼), чтобы малая новая капля не рябила
        x0,y0,x1,y1=DROP_BOX; x0-=PAD; y0-=PAD; x1+=PAD; y1+=PAD; s.dx0,s.dy0=x0,y0
        pm=np.dstack([ly['drgb']*ly['da'][...,None],ly['da'],ly['wc']])[y0:y1,x0:x1].astype(np.float32)
        s.mips=[]
        for m in (1.,.5,.25):
            f=cv.k*m; sz=(int(round((x1-x0)*f)),int(round((y1-y0)*f)))
            it=cv2.INTER_AREA if f<1 else cv2.INTER_CUBIC
            s.mips.append((m,f,(cv2.resize(np.ascontiguousarray(pm[...,:4]),sz,interpolation=it),
                                cv2.resize(np.ascontiguousarray(pm[...,4]),sz,interpolation=it))))
        # падение: от кончика, пока верх капли не уйдёт за нижний край экрана
        s.ybot=cv.y_bottom; s.ytop=cv.y_top-20
        s.t_det=OFF+4.0; s.Tf=0.78 if geo=='m' else 1.0
        s.Dfall=s.ybot-TIP[1]+40
        # удар: низ капли (≈ кончик + 512·растяжение) у нижнего края экрана
        s.t_hit=s.t_det+s.Tf*np.sqrt(np.clip((s.ybot-TIP[1]-512*1.3)/s.Dfall,0,1))
        # брызги-столбики
        org=a[514:616,395:404]; blu=a[544:605,324:334]
        rng=np.random.default_rng(56); s.parts=[]
        n=18
        for i in range(n):
            tex=org if i%2==0 else blu
            sp=col_sprite(tex,rng.uniform(10,16),rng.uniform(30,70))
            side=(-1)**(i//2)*rng.uniform(.2,1)
            if geo=='m': hp,vx,dur=rng.uniform(120,300),side*rng.uniform(80,420),rng.uniform(.36,.48)
            else: hp,vx,dur=rng.uniform(160,560),side*rng.uniform(120,700),rng.uniform(.48,.64)
            tilt=-np.degrees(np.arctan2(vx,2*hp/dur))             # столбик летит вдоль скорости — корона всплеска
            s.parts.append(dict(sp=sp,x=TIP[0]+side*30,hp=hp/cv.k,vx=vx/cv.k,tu=dur,
                                dl=rng.uniform(0,.12),w0=tilt+rng.uniform(-12,12),wv=np.sign(tilt)*rng.uniform(30,150),sc=rng.uniform(1.,1.4)*(1.3 if geo=='m' else 1.7)))
        # конец сцены — последняя брызга ушла за нижний край (без паузы после всплеска)
        end=0
        for p in s.parts:
            g=2*p['hp']/p['tu']**2; vy=g*p['tu']
            end=max(end,p['dl']+(vy+np.sqrt(vy*vy+180*g))/g)
        # последние мелкие брызги: вылетают в хвосте всплеска, медленнее и ниже, оседают за край к самому концу
        s.T=TN/FPS                                            # одна длина на все раскладки (таблица времён страницы)
        for j in range(4):
            sp=col_sprite(org if j%2 else blu,rng.uniform(7,10),rng.uniform(16,30))
            side=(-1)**j*rng.uniform(.3,.9); hp=rng.uniform(60,130)*(0.8 if geo=='m' else 1); tu=rng.uniform(.42,.5)
            vx=side*rng.uniform(60,200); g=2*(hp/cv.k)/tu**2; vy=g*tu
            fl=(vy+np.sqrt(vy*vy+180*g))/g
            dl=max(s.T-.03-.07*j-s.t_hit-fl,.15)
            tilt=-np.degrees(np.arctan2(vx,2*hp/tu))
            s.parts.append(dict(sp=sp,x=TIP[0]+side*30,hp=hp/cv.k,vx=vx/cv.k,tu=tu,dl=dl,w0=tilt,
                                wv=np.sign(tilt)*rng.uniform(30,120),sc=1.2*(1.3 if geo=='m' else 1.7)))
        s.ends=[s.t_hit+p['dl']+(lambda g,vy:(vy+np.sqrt(vy*vy+180*g))/g)(2*p['hp']/p['tu']**2,2*p['hp']/p['tu']) for p in s.parts]

    # ---------- капля
    def drop_state(s,t):
        """(x, y кончика в px картины, масштаб, поворот°, sy, sx0, выпуклость низа, качание низа px, дрожь города)"""
        # спуск сверху на конце нити: растёт из крошечной, садится на место картины с упругим покачиванием
        u=np.clip(t/A1,0,1); e=.35*eout(u)+.65*ease(u)
        ys=s.cv.y_top-512*.18*.6; ya=ys+(TIP[1]-ys)*e
        sc=.18+.82*ease(np.clip(u*1.15,0,1))
        tb=t-A1
        osca=.045*np.exp(-tb/.28)*np.sin(2*np.pi*tb/.42) if tb>0 else -.02*ramp(t,A1-.3,A1)
        anga=2.2*np.sin(2*np.pi*t/1.3)*(1-u)
        # набухает, качается, город трясётся, горлышко тянется — отрыв
        q=t-OFF
        sw=ramp(q,1.2,3.7)
        osc=.022*np.sin(2*np.pi*(q-1.2)/.62)*win(q,1.3,2.2,3.3,3.9)+osca
        neck=.05*ramp(t,s.t_det-.45,s.t_det)
        sy=(1+.10*sw+neck)*(1+osc); sx0=1/np.sqrt(1+osc)
        ang=1.6*np.sin(2*np.pi*(q-1.2)/1.7)*win(q,1.2,2.2,3.0,3.9)+anga
        sh=ramp(q,2.3,3.6)
        jig=sh*(1.6*np.sin(2*np.pi*7.3*t)+1.0*np.sin(2*np.pi*11.1*t+1))
        trem=.3*ramp(q,1.6,2.4)+.7*sh
        x,y=TIP[0],ya+8*neck/.05
        if t>s.t_det:                                         # падение с ускорением, вытягивается по ходу
            u=(t-s.t_det)/s.Tf; y+=s.Dfall*u*u
            sy*=1+.3*np.clip(u,0,1); sx0/=np.sqrt(1+.3*np.clip(u,0,1))
            ang+=3*np.sin(2*np.pi*u*.8)*np.clip(u*3,0,1); trem=1.
        return x,y,sc,ang,sy,sx0,.08*sw,jig,trem

    def drop_render(s,t,acc,acca,wgt):
        cv=s.cv; x,y,sc,ang,sy,sx0,bul,jig,trem=s.drop_state(t)
        if sc<=0: return
        m,f,(pm,pw)=next((mm for mm in reversed(s.mips) if mm[0]>=sc*.999),s.mips[0])
        K=cv.k*sc; cx,cy=cv.xy(x,y)
        r=np.radians(ang); ca,sa=np.cos(r),np.sin(r)
        # рамка на холсте
        ext=np.array([[-150,-PAD],[150,-PAD],[-150,560],[150,560]],np.float32)*[1.15,max(sy,1)*1.05]
        rc=ext@np.array([[ca,-sa],[sa,ca]]).T*K
        X0=int(max(np.floor(cx+rc[:,0].min())-4,0)); X1=int(min(np.ceil(cx+rc[:,0].max())+4,cv.W))
        Y0=int(max(np.floor(cy+rc[:,1].min())-4,0)); Y1=int(min(np.ceil(cy+rc[:,1].max())+4,cv.H))
        if X1<=X0 or Y1<=Y0: return
        X,Y=np.meshgrid(np.arange(X0,X1,dtype=np.float32),np.arange(Y0,Y1,dtype=np.float32))
        u=(X-cx)/K; v=(Y-cy)/K
        u,v=ca*u-sa*v, sa*u+ca*v                              # поворот (в договоре put: плюс — против часовой)
        vs=v/sy; q=np.clip(vs/512,0,1)
        us=u/(sx0*(1+bul*q*q))-jig*q*q
        px=(TIP[0]+us-s.dx0)*f; py=(TIP[1]+vs-s.dy0)*f
        px=px.astype(np.float32); py=py.astype(np.float32)
        if trem>0:
            wc=cv2.remap(pw,px,py,cv2.INTER_LINEAR,borderValue=0)
            hf=np.clip((705-(TIP[1]+vs))/170,0,1)               # верх зданий качается сильнее низа
            dx=trem*wc*(2.6*hf+.6)*np.sin(2*np.pi*6.1*t+vs/37)+trem*wc*1.2*np.sin(2*np.pi*9.7*t+us/19)
            dy=trem*wc*1.6*np.sin(2*np.pi*8.3*t+us/23+.7)
            px=(px+dx*f).astype(np.float32); py=(py+dy*f).astype(np.float32)
        o=cv2.remap(pm,px,py,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        acc[Y0:Y1,X0:X1]+=o[...,:3]*wgt; acca[Y0:Y1,X0:X1]+=o[...,3]*wgt

    def drop_layer(s,t):
        """слой капли с размытием движения на быстром падении"""
        cv=s.cv; acc=np.zeros((cv.H,cv.W,3),np.float32); acca=np.zeros((cv.H,cv.W),np.float32)
        n=1
        if t>s.t_det:
            u=(t-s.t_det)/s.Tf; vpx=2*s.Dfall*u/s.Tf*cv.k/FPS     # px холста за кадр
            n=int(np.clip(vpx/5,1,10))
        for i in range(n):
            dt=0 if n==1 else (i/(n-1)-.5)*.55/FPS
            s.drop_render(t+dt,acc,acca,1/n)
        return acc,acca

    # ---------- нить
    def thread(s,t):
        """полилиния нити (px картины) или None"""
        yt=s.ytop; ex=TIP[0]
        if t<=s.t_det: ey=s.drop_state(t)[1]                  # капля висит на конце нити
        else:
            tau=t-s.t_det                                     # отрыв: нить пружиной уходит вверх за край
            ey=TIP[1]+8+(s.ytop-TIP[1]-8)*eout(tau/.55)
            ex+=9*np.exp(-tau/.3)*np.sin(2*np.pi*tau/.4)
        if ey-yt<2: return None
        ys=np.linspace(yt,ey,80); q=(ys-yt)/max(ey-yt,1)
        xs=TIP[0]+(ex-TIP[0])*q+1.2*np.sin(np.pi*q)*np.sin(2*np.pi*t/2.3)
        return np.stack([xs,ys],1)

    # ---------- брызги
    def splash(s,img,t):
        cv=s.cv
        for p in s.parts:
            tau=t-s.t_hit-p['dl']
            if tau<=0: continue
            hp=p['hp']; tu=p['tu']; g=2*hp/tu**2; vy=g*tu
            y=s.ybot+30-(vy*tau-g*tau*tau/2)
            if y>s.ybot+120 and tau>tu: continue
            x=p['x']+p['vx']*tau
            ang=p['w0']+p['wv']*tau
            img=put(cv,img,p['sp'],x,y,p['sc'],ang)
        return img

    def frame(s,t):
        cv=s.cv; img=s.plate.copy()
        th=s.thread(t)
        if th is not None:
            a=lines_alpha(cv,[th],2.4); img=img*(1-a[...,None])+LINE_RGB*a[...,None]
        img=s.splash(img,t)
        if t<s.t_det+s.Tf*1.05:
            acc,acca=s.drop_layer(t); img=img*(1-np.clip(acca,0,1)[...,None])+acc
        img=over(img,s.sig,s.siga)
        return np.clip(img+.5,0,255).astype(np.uint8)

if __name__=='__main__': cli(Scene,'s56')
