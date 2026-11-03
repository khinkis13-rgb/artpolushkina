# Общее для анимированных сцен (tools/anim): геометрия кадра заставки, плавности, холст сцены (картина → кадр),
# спрайты с поворотом и масштабом, линии, кодирование в mp4. Каждая сцена — модуль sNN.py с классом Scene(geo).
#
# Договор сцены:
#   Scene(geo)          — geo: ключ GEO ('d2', 'd', 'm'); всё тяжёлое (разбор картины на слои) — здесь, один раз
#   .T                  — длительность, с (кратно 1/30); .work — id работы для подписи; .tone — (r,g,b) фона вокруг
#   .frame(t) → uint8   — кадр H×W×3 RGB в момент t ∈ [0, T); кадр 0 и кадр T — сама картина как есть (повтор и
#                         склейка с соседней сценой без рывка); всё, что движется, — внутри [~1 с, T−~1 с]
# Проверка сцены:  python3 tools/anim/sNN.py <папка> [d|d2|m] [кадры через запятую]  — png кадров или mp4 целиком
import sys, os, subprocess, numpy as np, cv2
from PIL import Image

FPS=30
# Геометрия — как у заставки (HERO_V в index.html): холст и где на нём «окно» картины (x, y, ш, в).
# Картина вписывается в окно целиком по центру; вокруг — фон сцены на весь экран.
GEO={'d2':dict(W=3744,H=2304,box=(1788,732,1280,720)),'d':dict(W=2808,H=1728,box=(1341,549,960,540)),
     'm':dict(W=720,H=2236,box=(0,124,720,720))}
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def ease(u): u=np.clip(u,0,1); return u*u*(3-2*u)
def ramp(t,a,b): return ease((t-a)/(b-a))
def eout(u): u=np.clip(u,0,1); return 1-(1-u)**3
def ein(u): u=np.clip(u,0,1); return u*u*u
def win(t,a,b,c,d): return ramp(t,a,b)*(1-ramp(t,c,d))        # 0 → 1 на [a,b], 1 → 0 на [c,d]

def load_work(path):
    return np.asarray(Image.open(os.path.join(ROOT,path)).convert('RGB')).astype(np.float32)

class Canvas:
    """Картина h×w (px картины) на холсте geo: k — масштаб, (ox, oy) — где её левый верхний угол."""
    def __init__(c,geo,h,w):
        g=GEO[geo]; c.geo=geo; c.W,c.H=g['W'],g['H']; bx,by,bw,bh=g['box']
        c.h,c.w=h,w; c.k=k=min(bw/w,bh/h); c.ox=bx+(bw-w*k)/2; c.oy=by+(bh-h*k)/2
        c.M=np.float32([[k,0,c.ox],[0,k,c.oy]])
        # края экрана в px картины
        c.x_left=-c.ox/k; c.x_right=(c.W-c.ox)/k; c.y_top=-c.oy/k; c.y_bottom=(c.H-c.oy)/k
    def xy(c,x,y): return c.ox+np.asarray(x)*c.k, c.oy+np.asarray(y)*c.k
    def warp(c,img):
        """неподвижный слой из px картины в px холста (снаружи картины — 0)"""
        return cv2.warpAffine(img.astype(np.float32),c.M,(c.W,c.H),flags=cv2.INTER_AREA if c.k<1 else cv2.INTER_CUBIC,
                              borderValue=0)
    def flat(c,tone): return np.broadcast_to(np.float32(tone),(c.H,c.W,3)).copy()

def over(img,rgb_pm,a):
    """наложить слой (цвет уже умножен на альфу) поверх img"""
    return img*(1-a[...,None])+rgb_pm

class Sprite:
    """Вырезанный кусок картины: rgb, альфа (в px картины), опора (ax, ay) — точка, вокруг которой крутим и которую ставим."""
    def __init__(s,rgb,a,off,anchor):
        s.pm=np.dstack([rgb*a[...,None],a]).astype(np.float32); s.off=off; s.anc=anchor
    @classmethod
    def cut(cls,img,alpha,anchor,pad=2):
        ys,xs=np.where(alpha>0); x0,x1=max(xs.min()-pad,0),xs.max()+1+pad; y0,y1=max(ys.min()-pad,0),ys.max()+1+pad
        return cls(img[y0:y1,x0:x1],alpha[y0:y1,x0:x1],(x0,y0),anchor)

def put(cv,img,sp,x,y,sc=1.,ang=0.,alpha=1.):
    """поставить спрайт опорой в точку (x, y) px картины, масштаб sc, поворот ang° (против часовой — плюс), прозрачность"""
    k=cv.k*sc; ax,ay=sp.anc; offx,offy=sp.off
    cx,cy=cv.xy(x,y)
    R=cv2.getRotationMatrix2D((0,0),ang,k); R[:,2]=[cx,cy]
    M=(np.vstack([R,[0,0,1]])@np.float32([[1,0,offx-ax],[0,1,offy-ay],[0,0,1]]))[:2]
    ph,pw=sp.pm.shape[:2]
    cs=np.array([[0,0,1],[pw,0,1],[0,ph,1],[pw,ph,1]],np.float32)@M.T
    x0,y0=np.floor(cs.min(0)).astype(int)-2; x1,y1=np.ceil(cs.max(0)).astype(int)+2
    x0,y0=max(x0,0),max(y0,0); x1,y1=min(x1,cv.W),min(y1,cv.H)
    if x1<=x0 or y1<=y0 or alpha<=0: return img
    M2=M.copy(); M2[:,2]-=[x0,y0]
    w=cv2.warpAffine(sp.pm,M2,(x1-x0,y1-y0),flags=cv2.INTER_LINEAR if sc*cv.k<1.5 else cv2.INTER_CUBIC,borderValue=0)*alpha
    a=np.clip(w[...,3:4],0,1); img[y0:y1,x0:x1]=img[y0:y1,x0:x1]*(1-a)+w[...,:3]
    return img

def lines_alpha(cv,polys,width):
    """альфа линий: polys — список массивов N×2 в px картины, width — толщина в px картины"""
    al=np.zeros((cv.H,cv.W),np.uint8)
    for p in polys:
        q=np.stack(cv.xy(p[:,0],p[:,1]),1)
        cv2.polylines(al,[np.round(q*16).astype(np.int32)],False,255,max(1,int(round(width*cv.k))),cv2.LINE_AA,4)
    return cv2.GaussianBlur(al.astype(np.float32)/255,(0,0),max(.35,.45*cv.k))

def hexrgb(c): return '#%02x%02x%02x'%tuple(int(round(v)) for v in c)

def encode(frames,W,H,out,crf=20):
    p=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-',
        '-vf','scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int,format=yuv420p',
        '-c:v','libx264','-preset','slower','-crf',str(crf),'-profile:v','high','-level','5.1','-g','300',
        '-x264-params','aq-mode=3:aq-strength=0.9','-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709',
        '-color_range','tv','-movflags','+faststart','-an',out],stdin=subprocess.PIPE)
    for n,f in enumerate(frames):
        p.stdin.write(f.tobytes())
        if n%60==0: print(out,n,flush=True)
    p.stdin.close(); p.wait()

def cli(Scene,name):
    """python3 tools/anim/sNN.py <папка> [geo] [кадры]: кадры → png, без кадров → <папка>/<name>-<geo>.mp4"""
    od=sys.argv[1]; geo=sys.argv[2] if len(sys.argv)>2 else 'd'; os.makedirs(od,exist_ok=True)
    sc=Scene(geo); W,H=GEO[geo]['W'],GEO[geo]['H']
    if len(sys.argv)>3:
        for f in sys.argv[3].split(','):
            Image.fromarray(sc.frame(int(f)/FPS)).save(f'{od}/{name}-{geo}-{f}.png')
        return
    N=int(round(sc.T*FPS)); encode((sc.frame(n/FPS) for n in range(N)),W,H,f'{od}/{name}-{geo}.mp4')
    print('tone',hexrgb(sc.tone),'T',sc.T)
