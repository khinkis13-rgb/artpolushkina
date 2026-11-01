#!/bin/bash
# Сборка заставки «всё в одном ролике»: из site/videos/hero.mp4 (ролик аниматора) и партитуры score.js —
# site/videos/hero-desktop.mp4 (2480×1536) и hero-phone.mp4 (464×1440) + кадры-заглушки. ~15 мин на 16 ядрах.
# Запуск: tools/hero/build.sh <временная папка>   (нужны python3 + numpy, Pillow, opencv; ffmpeg с libx264)
set -e
H=$(cd "$(dirname "$0")"; pwd); R=$(dirname "$(dirname "$H")"); TMP=${1:?временная папка}; mkdir -p "$TMP"
TOTAL=2498; STEP=210
# ключевые кадры — на склейках и сменах сцены; растворения в фактуру холста дешевле (зоны b=0.4): потом x264 догоняет качество
KF=$(python3 -c "print(','.join(['0']+['%.4f'%(round(c*30)/30) for c in sorted([4.8,16.833,22.333,30.333,35.333,46.3,51.3,56.3,61.3,67.267,72.267,77.267,7.37,37.35,40.333])]))")
for m in desktop:2480x1536 phone:464x1440; do n=${m%%:*}; sz=${m#*:}; i=0
  for ((a=0;a<TOTAL;a+=STEP)); do b=$((a+STEP)); ((b>TOTAL)) && b=$TOTAL
    (python3 "$H/$n.py" $a $b - 2>"$TMP/$n-$i.log" | ffmpeg -v error -y -f rawvideo -pix_fmt rgb24 -s $sz -r 30 -i - -c:v ffv1 -level 3 -pix_fmt bgr0 "$TMP/$n-$(printf %02d $i).mkv") &
    i=$((i+1)); done
done
wait
for n in desktop phone; do
  ls "$TMP"/$n-[0-9][0-9].mkv | sed "s#^#file '#;s#\$#'#" > "$TMP/$n.txt"
  ffmpeg -v error -y -f concat -safe 0 -i "$TMP/$n.txt" -vf "scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int,format=yuv420p" \
    -c:v libx264 -preset slower -crf 22 -profile:v high -level 5.1 -g 300 -keyint_min 30 -force_key_frames "$KF" \
    -x264-params "aq-mode=3:aq-strength=0.9:scenecut=0:zones=505,523,b=0.4/1689,1707,b=0.4" \
    -colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv -movflags +faststart -an "$R/site/videos/hero-$n.mp4"
  ffmpeg -v error -y -i "$R/site/videos/hero-$n.mp4" -vf "select='eq(n\,0)',scale=in_color_matrix=bt709:in_range=tv:out_range=pc:flags=accurate_rnd+full_chroma_int" \
    -frames:v 1 -q:v 4 "$R/site/images/hero-$n-poster.jpg"
done
ls -la "$R"/site/videos/hero-desktop.mp4 "$R"/site/videos/hero-phone.mp4
