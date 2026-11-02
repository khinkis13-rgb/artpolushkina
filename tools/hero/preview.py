# Просмотр отдельных кадров: python3 preview.py <цель> <мезонин> <папка> n1 n2 ... → png (уменьшенные до 1240 по ширине)
import sys,cv2,numpy as np
import build
tg,mz,out=sys.argv[1],sys.argv[2],sys.argv[3]
for n in map(int,sys.argv[4:]):
  img=build.compose(tg,n,build.one(mz,n)); img=np.clip(img+.5,0,255).astype(np.uint8)
  cv2.imwrite(f'{out}/{tg}-{n}.png',cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
  print(n,file=sys.stderr)
