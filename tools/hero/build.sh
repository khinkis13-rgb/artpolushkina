#!/bin/bash
# Сборка заставки из оригинала аниматора (3840×2160, 30 fps, 2504 кадра; первые 6 кадров — до начала, отбрасываем):
# site/videos/hero-desktop.mp4 (2808×1728, картина 960), hero-desktop-2x.mp4 (3744×2304, картина 1280),
# hero-phone.mp4 (720×2236) + кадры-заглушки. Края картин (assets/borders.npz) и рыба (assets/fish*) сняты заранее
# скриптами plates.py и fish.py — пересчитывать, только если сменится ролик или партитура (score.py).
# Запуск: tools/hero/build.sh <оригинал.mp4> <временная папка>  (~1 ч на 16 потоках; python3 + numpy, opencv; ffmpeg с libx264)
set -e
H=$(cd "$(dirname "$0")"; pwd); R=$(dirname "$(dirname "$H")"); SRC=${1:?оригинал аниматора}; TMP=${2:?временная папка}; mkdir -p "$TMP"
export HERO_TMP="$TMP"
# мезонин: 1920×1080 RGB без потерь (уменьшение 2:1 по площади; цвет 4K 4:2:0 — как раз полный в 1920×1080)
[ -f "$TMP/m1920.mkv" ] || ffmpeg -v error -y -i "$SRC" -an -vf "select='gte(n\,6)',setpts=N/30/TB,scale=1920:1080:in_color_matrix=bt709:in_range=tv:out_range=pc:flags=area+accurate_rnd+full_chroma_int,format=gbrp" \
  -vsync 0 -c:v ffv1 -level 3 -threads 16 -slices 16 -g 1 "$TMP/m1920.mkv"
MZ="$TMP/m1920.mkv"
python3 "$H/texture.py" "$MZ" "$TMP"
(cd "$H"; python3 build.py bands "$MZ")  # полосы краёв всех кадров — поправка тона бумаги (drift)
TOTAL=2498; STEP=250
# ключевые кадры — на склейках и сменах сцены; растворения в фактуру холста дешевле (зоны b=0.4): потом x264 догоняет качество
KF=$(cd "$H"; python3 -c "from score import *;print(','.join(['0']+['%.4f'%(round(r[0]*30)/30) for r in SC[1:]]))")
for m in lo:desktop:2808x1728:21 hi:desktop-2x:3744x2304:22 phone:phone:720x2236:23; do
  IFS=: read tg out sz crf <<<"$m"; i=0
  for ((a=0;a<TOTAL;a+=STEP)); do b=$((a+STEP)); ((b>TOTAL)) && b=$TOTAL
    (cd "$H"; python3 build.py $tg $a $b "$MZ" 2>"$TMP/$tg-$i.log" | ffmpeg -v error -y -f rawvideo -pix_fmt rgb24 -s $sz -r 30 -i - -c:v ffv1 -level 3 -pix_fmt bgr0 "$TMP/$tg-$(printf %02d $i).mkv") &
    i=$((i+1)); done
  wait
  ls "$TMP"/$tg-[0-9][0-9].mkv | sed "s#^#file '#;s#\$#'#" > "$TMP/$tg.txt"
  ffmpeg -v error -y -f concat -safe 0 -i "$TMP/$tg.txt" -vf "scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int,format=yuv420p" \
    -c:v libx264 -preset slower -crf $crf -profile:v high -level 5.1 -g 300 -keyint_min 30 -force_key_frames "$KF" \
    -x264-params "aq-mode=3:aq-strength=0.9:scenecut=0:zones=505,523,b=0.4/1689,1707,b=0.4" \
    -colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv -movflags +faststart -an "$R/site/videos/hero-$out.mp4"
  ffmpeg -v error -y -i "$R/site/videos/hero-$out.mp4" -vf "select='eq(n\,0)',scale=in_color_matrix=bt709:in_range=tv:out_range=pc:flags=accurate_rnd+full_chroma_int" \
    -frames:v 1 -q:v 4 "$R/site/images/hero-$out-poster.jpg"
done
ls -la "$R"/site/videos/hero-desktop.mp4 "$R"/site/videos/hero-desktop-2x.mp4 "$R"/site/videos/hero-phone.mp4
