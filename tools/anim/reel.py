# Ролик из сцен tools/anim/sNN.py без пауз: у каждой сцены срезаются неподвижные куски в начале и в конце (ищутся по
# кадрам телефонной раскладки, кэш в .still.json), хвост сцены перетекает в начало следующей за XF кадров, пока обе
# движутся. Повтор бесшовный: ролик начинается с первой сцены сразу после её смешения с хвостом последней, а это
# смешение идёт в самом конце ролика.
# python3 tools/anim/reel.py <имя> <s57,s66,...> <d|d2|m|m1> <out.mp4>  → out.mp4 и out.json (строки HERO_SC)
import sys, json, importlib, subprocess, numpy as np
from lib import *

XF=24                                                        # перетекание, кадров (0,8 с)
LEAD=3                                                       # сколько кадров покоя оставить у начала/конца движения
HERE=os.path.dirname(os.path.abspath(__file__))

def active(name):
    """[a, b) — кадры сцены, где есть движение (по раскладке m, с запасом LEAD)"""
    cache=os.path.join(HERE,'.still.json'); src=os.path.join(HERE,name+'.py'); key=f'{name}:{os.path.getmtime(src):.0f}'
    try: c=json.load(open(cache))
    except Exception: c={}
    if key in c: return tuple(c[key])
    s=importlib.import_module(name).Scene('m'); N=int(round(s.T*FPS))
    small=lambda n: cv2.resize(s.frame(n/FPS),(180,559),interpolation=cv2.INTER_AREA).astype(np.int16)
    f0=small(0); a=0
    while a<N-1 and np.abs(small(a+1)-f0).max()<=3: a+=1
    fN=small(N-1); b=N-1
    while b>a and np.abs(small(b-1)-fN).max()<=3: b-=1
    r=(max(a+1-LEAD,0),min(b+LEAD,N)); c[key]=list(r); json.dump(c,open(cache,'w'),indent=1)
    return r

def main():
    name,scenes,geo,out=sys.argv[1:5]; scenes=scenes.split(','); g='m' if geo=='m1' else geo
    rng=[active(n) for n in scenes]; mods=[importlib.import_module(n) for n in scenes]
    rows=[]; head0=[]
    def frames():
        tail=None; t=0.
        for i,(m,(a,b)) in enumerate(zip(mods,rng)):
            s=m.Scene(g); rows.append([round(t,3),s.work,hexrgb(s.tone),hexrgb(s.tone)])
            for j,n in enumerate(range(a,b)):
                f=s.frame(n/FPS)
                if j<XF:                                     # начало сцены: смешение с хвостом предыдущей
                    if i==0: head0.append(f); continue       # у первой — в самом конце ролика
                    u=float(ease((j+1)/(XF+1))); f=np.clip(tail[j]*(1-u)+f*u+.5,0,255).astype(np.uint8)
                    yield f; t+=1/FPS; continue
                if n>=b-XF:                                  # хвост — копим для смешения со следующей
                    if n==b-XF: tail=[]
                    tail.append(f.astype(np.float32)); continue
                yield f; t+=1/FPS
            del s
        for j in range(XF):
            u=float(ease((j+1)/(XF+1))); yield np.clip(tail[j]*(1-u)+head0[j]*u+.5,0,255).astype(np.uint8)
    W,H=GEO[g]['W'],GEO[g]['H']
    if geo=='m1':                                            # телефоны, не тянущие 2236 px: тот же кадр 618×1920
        tmp=out+'.full.mp4'; encode(frames(),W,H,tmp)
        subprocess.run(['ffmpeg','-v','error','-y','-i',tmp,'-vf','scale=618:1920:flags=lanczos,format=yuv420p','-c:v','libx264',
            '-preset','slower','-crf','20','-profile:v','high','-level','4.2','-g','300','-colorspace','bt709','-color_primaries','bt709',
            '-color_trc','bt709','-color_range','tv','-movflags','+faststart','-an',out],check=True)
        os.remove(tmp)
    else: encode(frames(),W,H,out)
    json.dump(rows,open(out.rsplit('.',1)[0]+'.json','w'))
    print(json.dumps(rows))

if __name__=='__main__': main()
