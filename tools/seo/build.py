#!/usr/bin/env python3
"""Статические страницы работ для поиска и превью ссылок.

Запуск из корня репозитория:  python3 tools/seo/build.py
Читает catalog.json и content.json, НИКОГДА их не переписывает (и index.html тоже). Пишет только своё:
  work/<id>/index.html, en/work/<id>/index.html   — страница работы (ru / en)
  work/index.html, en/work/index.html             — «Все работы»
  collection/<slug>/index.html, en/collection/<slug>/index.html — подборки из tools/seo/collections.json
  site/og/work-<id>.jpg                           — превью 1200×630 для ссылок (одно на оба языка)
  sitemap.xml                                     — карта сайта целиком
  tools/seo/state.json                            — манифест: хеши страниц и даты lastmod, хеши исходников превью
Устаревшие work/<число>/, en/work/<число>/, collection/<slug>/, en/collection/<slug>/ и site/og/work-<число>.jpg удаляются.

Подборки (tools/seo/collections.json, необязателен — без него подборок нет): список объектов
  {"slug": "large", "ru": "...", "en": "...", "descRu": "...", "descEn": "...",
   "titleRu": "...", "titleEn": "...",          — необязательно: свой <title> вместо «<ru> - Мария Полушкина, купить оригинал»
   "rule": {"minSide": 80, "maxSide": 40, "minPrice": 0, "maxPrice": 29999},   — любые из условий, все сразу
   "ids": [57, 58]}                             — явный список (вместе с rule — пересечение)
Сторона — большая сторона из dims() (как фильтр размера на главной), цена — p; в правиле по цене только непроданные.
Порядок работ — как в catalog.json, отсутствующие id пропускаются, подборка меньше MIN_COL работ не генерируется.

Что изображено (tools/seo/alt.json, необязателен): {"57": {"ru": "...", "en": "..."}} — невидимый гостю текст для
поисковиков и экранных чтецов, публикуется без одобрения. На странице работы идёт в alt главного фото (#mainImg) и первой
миниатюры галереи: «<work_alt>. <текст>»; остальные фото, карточки на других страницах и og:image:alt — коротко, как были.
Ещё — в description JSON-LD VisualArtwork, если нет одобренной подписи. В meta/og/twitter description НЕ идёт.

Видимые подписи (tools/seo/descriptions.json, необязателен — без него или с {} страницы как были):
  {"57": {"ru": "...", "en": "...", "ok": true, "note": "..."}}
Выводятся ТОЛЬКО при "ok": true (одобрила Маша): абзац под характеристиками до цены, <meta name="description">,
og:description и twitter:description (обрезка по границе слова/предложения до DESC_MAX знаков), description в JSON-LD
(целиком, приоритетнее alt.json). Язык по отдельности: пустой "en" — на en-странице как без подписи. "note" — для себя.
CSS абзаца — только на страницах с подписью.
Оба файла: нет файла — нет текстов; битый JSON или не объект — выход с ошибкой, ничего не записав.
Список изменённых в этом прогоне адресов печатается в stdout и, с --changed <файл>, пишется в файл
(для IndexNow в GitHub Action). Повторный запуск без изменений входа ничего не меняет.

Зависимости: Python 3.12, Pillow.
"""
import argparse
import datetime
import hashlib
import html
import json
import os
import re
import shutil
import sys
import urllib.parse

from PIL import Image, ImageFilter

SITE = 'https://artpolushkina.com'
HOST = 'artpolushkina.com'
# Ключ IndexNow публичный по протоколу (лежит файлом в корне сайта) — это не секрет
INDEXNOW_KEY = 'a3b7763e080f5bcd05d788c7dc67d9ed'
YM_ID = 114487567
OG_VERSION = '1'          # поднять, чтобы пересобрать все превью (сменился вид картинки)
OG_W, OG_H = 1200, 630
OG_BG = (247, 241, 232)   # --canvas
BOT_RE = (r'googlebot|google-inspectiontool|adsbot-google|storebot-google|yandex(bot|images|metrika|\.com\/bots)|'
          r'bingbot|duckduckbot|applebot|baiduspider|petalbot|crawler|spider|slurp|lighthouse')
NBSP = '\u00a0'
MIN_COL = 4               # подборка, где работ меньше, не генерируется
DESC_MAX = 200            # длина описания в meta/og/twitter; в JSON-LD и в абзаце на странице — целиком
SLUG_RE = r'[a-z0-9]+(?:-[a-z0-9]+)*'

# ───────────────────────── данные ─────────────────────────

def esc(s):
    return html.escape(str(s), quote=True)


def cap_t(t):
    """Заглавные названия («ГАМАК») — в обычный регистр, как capT в index.html."""
    t = str(t or '').strip()
    if re.search(r'[^\W\d_]', t) and t == t.upper():
        return t[:1] + t[1:].lower()
    return t


def medium(a, lang):
    """«Диптих. Холст, акрил» → «диптих, холст, акрил» (как workAlt в index.html)."""
    m = str((a.get('m') if lang == 'ru' else a.get('me') or a.get('m')) or '')
    m = re.sub(r'\.\s*', ', ', m).strip()
    m = re.sub(r',\s*$', '', m)
    return m.lower()


def size_disp(s):
    return re.sub(r'(?<=\d)\s*[×xXхХ*]\s*(?=\d)', '×', str(s or '').strip())


def dims(a):
    """[высота, ширина] в см; у диптиха «40×30 + 40×30» ширины складываются (как dims в index.html)."""
    ps = [(float(x.replace(',', '.')), float(y.replace(',', '.')))
          for x, y in re.findall(r'(\d+(?:[.,]\d+)?)\s*[×xXхХ*]\s*(\d+(?:[.,]\d+)?)', str(a.get('s') or ''))]
    if not ps:
        return None
    return max(p[0] for p in ps), sum(p[1] for p in ps) + (len(ps) - 1) * 3


def num(x):
    return int(x) if float(x).is_integer() else x


def price_ru(p):
    return f'{p:,}'.replace(',', NBSP) + NBSP + '₽'


def price_en(p):
    return f'₽{p:,}'


def plural_ru(n, one, few, many):
    if n % 10 == 1 and n % 100 != 11:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many


def page_path(wid, lang):
    return f'/work/{wid}/' if lang == 'ru' else f'/en/work/{wid}/'


def list_path(lang):
    return '/work/' if lang == 'ru' else '/en/work/'


def col_path(slug, lang):
    return f'/collection/{slug}/' if lang == 'ru' else f'/en/collection/{slug}/'


def home_path(lang):
    return '/' if lang == 'ru' else '/?lang=en'


def abs_url(path):
    return SITE + path


def file_url(rel):
    return '/' + urllib.parse.quote(rel.lstrip('/'), safe='/')


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode('utf-8')).hexdigest()


def js_json(o):
    # '</' закрыл бы <script>, '<!--' переводит парсер скрипта в «escaped»-режим — оба безопасно экранируем
    return (json.dumps(o, ensure_ascii=False, separators=(',', ':'))
            .replace('</', '<\\/').replace('<!--', '\\u003c!--'))


def clip(s, n=DESC_MAX):
    """Обрезка для meta/og: целыми предложениями, если влезает хотя бы ~60 % длины, иначе по слову с «…»."""
    s = re.sub(r'\s+', ' ', str(s or '')).strip()
    if len(s) <= n:
        return s
    ends = [m.end() for m in re.finditer(r'[.!?…](?=\s)', s[:n + 1])]
    if ends and ends[-1] >= n * 0.6:
        return s[:ends[-1]].strip()
    cut = s[:n - 1]
    if ' ' in cut:
        cut = cut[:cut.rindex(' ')]
    return cut.rstrip(' ,;:-–—(«"\'') + '…'


class Ctx:
    def __init__(self, root, out, works, contacts):
        self.root, self.out, self.works, self.contacts = root, out, works, contacts
        self._sizes = {}
        self.og = {}   # id → (published path, version tag) or None
        self.cols = []        # [(подборка, [работы])] — только те, что генерируются
        self.work_cols = {}   # id работы → [подборки, где она есть]
        self.desc = {}        # id работы → {'ru': текст, 'en': текст} — видимые подписи, только одобренные (ok: true)
        self.alt = {}         # id работы → {'ru': текст, 'en': текст} — что изображено (alt.json), без одобрения

    def img_size(self, rel):
        if rel not in self._sizes:
            p = os.path.join(self.root, rel)
            try:
                with Image.open(p) as im:
                    w, h = im.size
                    try:
                        if im.getexif().get(274) in (5, 6, 7, 8):
                            w, h = h, w
                    except Exception:
                        pass
                self._sizes[rel] = (w, h)
            except Exception:
                self._sizes[rel] = None
        return self._sizes[rel]


def media_of(a):
    ms = [m for m in (a.get('media') or []) if isinstance(m, dict) and m.get('path')]
    if not ms:
        ms = [{'type': 'img', 'path': f"site/images/work-{a['id']}.jpg"}]
    return ms


def images_of(a):
    return [m['path'] for m in media_of(a) if m.get('type', 'img') == 'img']


def title_of(a, lang):
    return str((a.get('t') if lang == 'ru' else a.get('te') or a.get('t')) or '').strip()


def work_alt(a, lang, n=0):
    ru = lang == 'ru'
    mm = medium(a, lang)
    s = f"{cap_t(title_of(a, lang))} - {'Мария Полушкина' if ru else 'Maria Polushkina'}"
    if mm:
        s += ', ' + mm
    if a.get('s'):
        s += f", {size_disp(a['s'])} {'см' if ru else 'cm'}"
    if n:
        s += f", {'фото' if ru else 'photo'} {n + 1}"
    return s


def has_price(a):
    return isinstance(a.get('p'), (int, float)) and a['p'] > 0


def buy_text(a, lang, dm):
    """Текст для лички Маши — как oM/workLine в index.html, ссылка на страницу работы."""
    ru = lang == 'ru'
    line = f"«{title_of(a, lang)}» (№{a['id']}), {a.get('s', '')} {'см' if ru else 'cm'}"
    url = abs_url(page_path(a['id'], lang))
    if a.get('sold'):
        line += f' - {url}'
        return (f'Здравствуйте, Мария! Мне очень понравилась ваша работа {line}. Вижу, что она продана - есть ли похожие?'
                if ru else f'Hello Maria! I really love your work {line}. I see it has been sold - do you have anything similar?')
    if has_price(a):
        line += ', ' + price_ru(int(a['p']))
    line += f' - {url}'
    return ('Здравствуйте, Мария! Хочу купить работу ' if ru else 'Hello Maria! I would like to buy ') + line


def buy_subj(a, lang):
    """Тема письма — как mBuy в index.html."""
    t, n = title_of(a, lang), a['id']
    if lang == 'ru':
        return f'Похожая на «{t}» (№{n})' if a.get('sold') else f'Работа «{t}» (№{n})'
    return f'Similar to “{t}” (No. {n})' if a.get('sold') else f'Painting “{t}” (No. {n})'


def channels(ctx, lang):
    """Каналы связи по порядку языка: у своих первым Telegram, иностранцам — WhatsApp и почта (как chList в index.html)."""
    order = ['tg', 'wa', 'mail'] if lang == 'ru' else ['wa', 'mail', 'tg']
    return [c for c in order if (c != 'wa' or ctx.wa) and (c != 'mail' or ctx.email)]


def ch_url(ctx, ch, msg, subj):
    q = urllib.parse.quote(msg, safe='')
    if ch == 'wa':
        return f'https://wa.me/{ctx.wa}?text={q}'
    if ch == 'mail':
        return f'mailto:{ctx.email}?subject={urllib.parse.quote(subj, safe="")}&body={q}'
    return f'https://t.me/{ctx.dm}?text={q}'


def ch_name(ch, lang):
    return {'tg': 'Telegram', 'wa': 'WhatsApp'}.get(ch) or ('почта' if lang == 'ru' else 'Email')


def ch_link(ctx, ch, msg, subj, cls, inner, id_=''):
    attrs = (f' class="{cls}"' if cls else '') + (f' id="{id_}"' if id_ else '')
    tgt = '' if ch == 'mail' else ' target="_blank" rel="noopener"'
    return f'<a{attrs} data-ch="{ch}" href="{esc(ch_url(ctx, ch, msg, subj))}"{tgt}>{inner}</a>'


# ───────────────────────── общие куски страницы ─────────────────────────

FONTS = [('Cormorant Garamond', 'cormorant-n', '300 600'), ('Nunito Sans', 'nunito-n', '300 500')]
RANGES = {
    'cyrillic-ext': 'U+0460-052F, U+1C80-1C8A, U+20B4, U+2DE0-2DFF, U+A640-A69F, U+FE2E-FE2F',
    'cyrillic': 'U+0301, U+0400-045F, U+0490-0491, U+04B0-04B1, U+2116',
    'latin-ext': ('U+0100-02BA, U+02BD-02C5, U+02C7-02CC, U+02CE-02D7, U+02DD-02FF, U+0304, U+0308, U+0329, '
                  'U+1D00-1DBF, U+1E00-1E9F, U+1EF2-1EFF, U+2020, U+20A0-20AB, U+20AD-20C4, U+2113, U+2C60-2C7F, U+A720-A7FF'),
    'latin': ('U+0000-00FF, U+0131, U+0152-0153, U+02BB-02BC, U+02C6, U+02DA, U+02DC, U+0304, U+0308, U+0329, '
              'U+2000-206F, U+20AC, U+2122, U+2191, U+2193, U+2212, U+2215, U+FEFF, U+FFFD'),
}
FONT_CSS = ''.join(
    f"@font-face{{font-family:'{fam}';font-style:normal;font-weight:{wt};font-display:swap;"
    f"src:url(/site/fonts/{f}-{sub}.woff2) format('woff2');unicode-range:{rng}}}\n"
    for fam, f, wt in FONTS for sub, rng in RANGES.items())

CSS = FONT_CSS + """:root{--canvas:#F7F1E8;--canvas-w:#F2EADD;--navy-d:#2C3E5A;--burg:#8B4049;--mars:#7A3040;--gold:#C8A96E;--gold-s:#DABB85;--char:#2D2A26;--wg:#8A8178;--lg:#D1CBC3;--fd:'Cormorant Garamond',Georgia,serif;--fb:'Nunito Sans',system-ui,sans-serif;--mw:1240px;--e:cubic-bezier(.25,.46,.45,.94)}
*,*::before,*::after{margin:0;padding:0;box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{font-family:var(--fb);background:var(--canvas);color:var(--char);font-weight:300;line-height:1.7;font-size:16px;overflow-x:hidden;min-height:100vh;display:flex;flex-direction:column}
a{color:var(--burg);text-decoration:none;transition:.3s var(--e)}
a:hover{color:var(--mars)}
img{display:block;max-width:100%}
.nav{position:sticky;top:0;z-index:100;background:rgba(247,241,232,.92);-webkit-backdrop-filter:blur(20px);backdrop-filter:blur(20px);border-bottom:1px solid rgba(200,169,110,.15)}
.ni{max-width:var(--mw);margin:0 auto;padding:0 28px;display:flex;align-items:center;justify-content:space-between;height:60px;gap:16px}
.nl img{height:46px;width:auto;margin-top:8px}
.nk{display:flex;align-items:center;gap:24px}
.nk a{font-size:.78rem;font-weight:400;letter-spacing:.08em;text-transform:uppercase;color:var(--wg);white-space:nowrap}
.nk a:hover{color:var(--burg)}
.nk a.lb{border:1px solid var(--lg);padding:4px 14px;border-radius:20px;font-size:.72rem;letter-spacing:.12em}
.nk a.lb:hover{border-color:var(--burg)}
main{flex:1;width:100%;max-width:var(--mw);margin:0 auto;padding:22px 28px 72px}
.bc{font-size:.78rem;color:var(--wg);margin-bottom:24px;display:flex;flex-wrap:wrap;gap:4px 8px}
.bc a{color:var(--wg)}.bc a:hover{color:var(--burg)}
.bc span[aria-hidden]{opacity:.6}
.wk{display:grid;grid-template-columns:minmax(0,3fr) minmax(0,2fr);gap:48px;align-items:start}
.wm{min-width:0}
.wm .main{margin:0 auto;height:auto;border-radius:6px;box-shadow:0 14px 36px rgba(45,42,38,.12);background:var(--canvas-w)}
.gal{display:flex;flex-wrap:wrap;gap:10px;margin-top:18px;justify-content:center}
.gt{border:1px solid var(--lg);background:var(--canvas-w);border-radius:6px;padding:3px;cursor:pointer;height:82px;transition:.2s var(--e)}
.gt img{height:100%;width:auto;border-radius:3px}
.gt:hover,.gt.on{border-color:var(--burg)}
.vids{margin-top:18px;display:grid;gap:14px}
.vids video{width:100%;max-height:80vh;border-radius:6px;background:var(--char)}
.wi h1{font-family:var(--fd);font-size:clamp(1.9rem,3.4vw,2.6rem);font-weight:400;line-height:1.15;margin-bottom:14px;overflow-wrap:anywhere}
.meta{font-size:.9rem;color:var(--wg);margin-bottom:20px;line-height:1.9}
.meta span{display:block}
.mp{font-family:var(--fd);font-size:1.8rem;font-weight:500;color:var(--burg);margin-bottom:22px;line-height:1.3}
.fx{display:block;font-family:var(--fb);font-weight:300;font-size:.85rem;color:var(--wg);margin-top:2px;letter-spacing:.01em}
.fx:empty{display:none}
.sold{display:inline-block;font-family:var(--fb);font-size:.78rem;font-weight:500;letter-spacing:.12em;text-transform:uppercase;color:var(--wg);border:1px solid var(--lg);border-radius:20px;padding:5px 16px;line-height:1.5;white-space:nowrap}
.mt{display:inline-flex;align-items:center;gap:10px;padding:13px 28px;border-radius:28px;background:var(--burg);color:#fff;font-size:.9rem;border:none;cursor:pointer;transition:.3s;text-decoration:none}
.mt:hover{background:var(--mars);color:#fff;transform:translateY(-2px)}
.mt svg{width:18px;height:18px;fill:#fff;flex:none}
.acts{display:flex;flex-wrap:wrap;gap:10px;margin-top:14px}
.m2{display:inline-flex;align-items:center;gap:8px;padding:10px 18px;border-radius:24px;border:1px solid var(--lg);background:transparent;font-family:var(--fb);font-size:.84rem;font-weight:300;color:var(--char);cursor:pointer;transition:.2s var(--e)}
.m2:hover{border-color:var(--burg);color:var(--burg)}
.m2 svg{width:16px;height:16px;flex:none}
.m2.try{border-color:var(--burg);color:var(--burg)}
.m2.try:hover{background:rgba(139,64,73,.06)}
.ds{margin-top:26px;font-size:.92rem;color:var(--char);max-width:34em}
.pn{display:flex;justify-content:space-between;gap:16px;margin-top:30px;padding-top:18px;border-top:1px solid rgba(200,169,110,.25);font-size:.85rem}
.pn a{color:var(--wg);display:flex;flex-direction:column;max-width:48%}
.pn a:hover{color:var(--burg)}
.pn a b{font-weight:400;font-size:.72rem;letter-spacing:.1em;text-transform:uppercase}
.pn a span{font-family:var(--fd);font-size:1.05rem;color:var(--char);line-height:1.25;overflow-wrap:anywhere}
.pn .nx{text-align:right;margin-left:auto}
.st{font-family:var(--fd);font-size:clamp(1.6rem,3vw,2.1rem);font-weight:300;margin:64px 0 6px;text-align:center}
.gl{width:48px;height:2px;background:var(--gold);margin:10px auto 28px;border-radius:2px}
.lead{max-width:40em;margin:-12px auto 30px;text-align:center;color:var(--wg);font-size:.92rem}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:34px 26px}
.grid.more{grid-template-columns:repeat(auto-fill,minmax(160px,1fr))}
.cd{display:block;color:var(--char);min-width:0}
.cd:hover{color:var(--char)}
.th{height:230px;display:flex;align-items:flex-end;justify-content:center;margin-bottom:12px}
.more .th{height:180px}
.th img{height:100%;width:auto;max-width:100%;object-fit:contain;object-position:center bottom;transition:.4s var(--e)}
.cd:hover .th img{transform:translateY(-4px)}
.ctt{font-family:var(--fd);font-size:1.08rem;font-weight:500;line-height:1.25;margin-bottom:3px;overflow-wrap:anywhere}
.cc{font-size:.8rem;color:var(--wg)}
.cp{font-family:var(--fd);font-size:1.1rem;font-weight:500;color:var(--burg);margin-top:2px}
.cp .fx{font-size:.72rem;line-height:1.3}
.cp .sold{font-size:.66rem;padding:2px 10px}
.all{text-align:center;margin-top:40px}
.all a{display:inline-flex;gap:6px;padding:11px 26px;border:1px solid var(--burg);border-radius:26px;font-size:.88rem}
.all a:hover{background:rgba(139,64,73,.06)}
.ft{background:var(--char);color:var(--wg);text-align:center;padding:20px 28px;font-size:.78rem}
.ft .mk{margin-top:6px;font-size:11px;opacity:.45}
.toast{position:fixed;left:50%;bottom:40px;transform:translate(-50%,20px);z-index:400;background:var(--char);color:var(--canvas);padding:11px 18px;border-radius:12px;font-size:.84rem;max-width:calc(100% - 32px);text-align:center;opacity:0;pointer-events:none;transition:.3s var(--e)}
.toast.vis{opacity:1;transform:translate(-50%,0)}
@media(max-width:900px){.wk{grid-template-columns:1fr;gap:30px}.ds{max-width:none}}
@media(max-width:600px){.ni{padding:0 16px}.nk{gap:14px}main{padding:16px 16px 56px}.grid{grid-template-columns:repeat(2,minmax(0,1fr));gap:26px 14px}.th,.more .th{height:170px}.st{margin-top:48px}.mt{width:100%;justify-content:center}}
@media(max-width:360px){.nk .na{display:none}.th,.more .th{height:140px}.m2{padding:9px 13px;font-size:.8rem}}
"""

# Подборки: кнопки-ссылки и строка «В подборках» (в CSS страницы — только когда подборки есть)
CSS_COL = """.chips{display:flex;flex-wrap:wrap;justify-content:center;align-items:center;gap:8px 10px;margin:-8px auto 34px;max-width:52em}
.chips b{font-weight:400;font-size:.74rem;letter-spacing:.1em;text-transform:uppercase;color:var(--wg);margin-right:4px}
.chips a{display:inline-block;padding:6px 16px;border:1px solid var(--lg);border-radius:20px;font-size:.84rem;line-height:1.4;color:var(--char);overflow-wrap:anywhere}
.chips a:hover{border-color:var(--burg);color:var(--burg)}
.chips.bot{margin:56px auto 0}
.inc{margin-top:14px;font-size:.84rem;color:var(--wg)}
.inc a{color:var(--wg);border-bottom:1px solid var(--lg)}
.inc a:hover{color:var(--burg);border-color:var(--burg)}
@media(max-width:600px){.chips a{padding:5px 13px;font-size:.8rem}}
"""

# Описание работы (одобренное Машей) — подключается только на странице, где оно есть
CSS_DESC = """.wd{font-size:1rem;line-height:1.75;color:var(--char);margin:-4px 0 22px;max-width:34em;overflow-wrap:break-word}
@media(max-width:900px){.wd{max-width:none}}
@media(max-width:600px){.wd{font-size:.96rem;margin-bottom:20px}}
"""

# «Купить»: строка доставки под ценой и другие каналы под кнопкой — только на страницах работ
CSS_BUY = """.ship{font-size:.84rem;color:var(--wg);line-height:1.5;margin:-14px 0 22px}
.och{margin-top:10px;font-size:.84rem;color:var(--wg);line-height:1.5}
.och a{color:var(--char);text-decoration:underline;text-decoration-color:var(--lg);text-underline-offset:3px}
.och a:hover{color:var(--burg);text-decoration-color:currentColor}
@media(max-width:600px){.och{text-align:center}}
"""

WA_SVG = ('<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M17.472 14.382c-.297-.149-1.758-.867-2.03-.967-.273-.099-.471-.148-.67.15-.197.297-.767.966-.94 1.164-.173.199-.347.223-.644.075-.297-.15-1.255-.463-2.39-1.475-.883-.788-1.48-1.761-1.653-2.059-.173-.297-.018-.458.13-.606.134-.133.298-.347.446-.52.149-.174.198-.298.298-.497.099-.198.05-.371-.025-.52-.075-.149-.669-1.612-.916-2.207-.242-.579-.487-.5-.669-.51-.173-.008-.371-.01-.57-.01-.198 0-.52.074-.792.372-.272.297-1.04 1.016-1.04 2.479 0 1.462 1.065 2.875 1.213 3.074.149.198 2.096 3.2 5.077 4.487.709.306 1.262.489 1.694.625.712.227 1.36.195 1.871.118.571-.085 1.758-.719 2.006-1.413.248-.694.248-1.289.173-1.413-.074-.124-.272-.198-.57-.347m-5.421 7.403h-.004a9.87 9.87 0 0 1-5.031-1.378l-.361-.214-3.741.982.998-3.648-.235-.374a9.86 9.86 0 0 1-1.51-5.26c.001-5.45 4.436-9.884 9.888-9.884 2.64 0 5.122 1.03 6.988 2.898a9.825 9.825 0 0 1 2.893 6.994c-.003 5.45-4.437 9.884-9.885 9.884m8.413-18.297A11.815 11.815 0 0 0 12.05 0C5.495 0 .16 5.335.157 11.892c0 2.096.547 4.142 1.588 5.945L.057 24l6.305-1.654a11.882 11.882 0 0 0 5.683 1.448h.005c6.554 0 11.89-5.335 11.893-11.893a11.821 11.821 0 0 0-3.48-8.413"/></svg>')
MAIL_SVG = ('<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2zm0 4l-8 5-8-5V6l8 5 8-5v2z"/></svg>')
TG_SVG = ('<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M11.944 0A12 12 0 0 0 0 12a12 12 0 0 0 12 12 12 12 0 0 0 12-12A12 12 0 0 0 12 0zm4.962 7.224c.1-.002.321.023.465.14a.506.506 0 0 1 .171.325c.016.093.036.306.02.472-.18 1.898-.962 6.502-1.36 8.627-.168.9-.499 1.201-.82 1.23-.696.065-1.225-.46-1.9-.902-1.056-.693-1.653-1.124-2.678-1.8-1.185-.78-.417-1.21.258-1.91.177-.184 3.247-2.977 3.307-3.23.007-.032.014-.15-.056-.212s-.174-.041-.249-.024c-.106.024-1.793 1.14-5.061 3.345-.479.33-.913.49-1.302.48-.428-.008-1.252-.241-1.865-.44-.752-.245-1.349-.374-1.297-.789.027-.216.325-.437.893-.663 3.498-1.524 5.83-2.529 6.998-3.014 3.332-1.386 4.025-1.627 4.476-1.635z"/></svg>')
TRY_SVG = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><rect x="5" y="4" width="14" height="12" rx="1"/><path d="M8 13l3-3 2 2 3-3M2 20h20"/></svg>')
SHARE_SVG = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><path d="M12 3v12M7 8l5-5 5 5M5 14v5a2 2 0 002 2h10a2 2 0 002-2v-5"/></svg>')

# Язык для людей — как window.LANG0 на главной; роботов не перекидываем никогда
LANG_JS = ("(function(){var pl=%s,alt=%s,w;"
           "try{var q=(new URLSearchParams(location.search).get('lang')||'').toLowerCase();if(q==='en'||q==='ru')w=q;}catch(_){}"
           "if(!w){if(/" + BOT_RE + "/i.test(navigator.userAgent))return;"
           "try{var s=localStorage.getItem('mp_lang');if(s==='en'||s==='ru')w=s;}catch(_){}}"
           "if(!w){var b=((navigator.languages&&navigator.languages[0])||navigator.language||'ru').toLowerCase();"
           "w=b==='ro-md'||/^(ru|uk|be|kk|ky|uz|tg|hy|az|ka)\\b/.test(b)?'ru':'en';}"
           "if(w!==pl)location.replace(alt+location.search+location.hash);})();")

# Общий скрипт внизу: Метрика (после load или через 1,5 с, как на главной), цели, буфер, тост, курс, язык, кнопки
TAIL_JS = """(function(){
var P=window.P||{},ru=P.lang==='ru',YM=%d;
if(YM>0){window.ym=window.ym||function(){(ym.a=ym.a||[]).push(arguments);};ym.l=+new Date;
ym(YM,'init',{clickmap:true,trackLinks:true,accurateTrackBounce:true,webvisor:true});
var ymLoad=function(){var s=document.createElement('script');s.async=true;s.src='https://mc.yandex.ru/metrika/tag.js';document.head.appendChild(s);};
if(document.readyState==='complete')ymLoad();else{var on=false,once=function(){if(on)return;on=true;ymLoad();};addEventListener('load',once,{once:true});setTimeout(once,1500);}}
function goal(n,p){if(YM>0&&typeof window.ym==='function')try{ym(YM,'reachGoal',n,p);}catch(_){}}
function copyText(t){
if(navigator.clipboard&&window.isSecureContext)return navigator.clipboard.writeText(t).then(function(){return true;},function(){return false;});
try{var ta=document.createElement('textarea');ta.value=t;ta.style.cssText='position:fixed;opacity:0';document.body.appendChild(ta);ta.select();var ok=document.execCommand('copy');ta.remove();return Promise.resolve(ok);}catch(_){return Promise.resolve(false);}}
var tT;function toast(t){var el=document.getElementById('toast');if(!el)return;el.textContent=t;el.classList.add('vis');clearTimeout(tT);tT=setTimeout(function(){el.classList.remove('vis');},3800);}
document.querySelectorAll('[data-lang]').forEach(function(a){a.addEventListener('click',function(e){
try{localStorage.setItem('mp_lang',a.getAttribute('data-lang'));}catch(_){e.preventDefault();location.href=a.getAttribute('href')+'?lang='+a.getAttribute('data-lang');}});});
var r10=function(x){return Math.max(10,Math.round(x/10)*10);};
if(!ru){var fx=document.querySelectorAll('.fx[data-p]');if(fx.length)fetch('/fx.json').then(function(r){return r.ok?r.json():null;}).then(function(j){
var r=j&&j.rub_per;if(!(r&&r.EUR>0&&r.USD>0))return;fx.forEach(function(el){var p=+el.getAttribute('data-p');if(p>0)el.textContent='\\u2248 \\u20ac'+r10(p/r.EUR).toLocaleString('en-US')+' \\u00b7 $'+r10(p/r.USD).toLocaleString('en-US');});}).catch(function(){});}
document.querySelectorAll('[data-ch]').forEach(function(b){b.addEventListener('click',function(){var ch=b.getAttribute('data-ch');
copyText(P.msg).then(function(ok){if(ok)toast(ch==='mail'?(ru?'Текст письма скопирован - вставьте его в письмо, если он не появился сам':'Message copied - paste it into the email if it did not appear')
:(ru?'Текст сообщения скопирован - вставьте его в чат, если он не появился сам':'Message copied - paste it into the chat if it did not appear'));});goal(P.sold?'sold_ask':'buy',{id:P.id,ch:ch});});});
var sh=document.getElementById('share');
if(sh)sh.addEventListener('click',function(){goal('share',{id:P.id});
var done=function(){copyText(P.url).then(function(ok){toast(ok?(ru?'Ссылка на работу скопирована':'Link copied'):P.url);});};
if(navigator.share)navigator.share({title:P.title,url:P.url}).catch(function(e){if(!e||e.name!=='AbortError')done();});else done();});
var mi=document.getElementById('mainImg');
document.querySelectorAll('.gt').forEach(function(b){b.addEventListener('click',function(){
if(!mi)return;mi.src=b.getAttribute('data-src');mi.width=+b.getAttribute('data-w');mi.height=+b.getAttribute('data-h');mi.alt=b.getAttribute('data-alt');mi.style.width=b.getAttribute('data-css');
document.querySelectorAll('.gt').forEach(function(x){x.classList.toggle('on',x===b);x.setAttribute('aria-pressed',x===b?'true':'false');});});});
})();"""


def head(lang, title, desc, canon_path, alt_path, ru_path, extra_meta, ld, css=CSS):
    preload = ['cormorant-n-cyrillic', 'nunito-n-cyrillic', 'nunito-n-latin'] if lang == 'ru' else ['cormorant-n-latin', 'nunito-n-latin']
    en_path = canon_path if lang == 'en' else alt_path
    out = [
        '<!DOCTYPE html>',
        f'<html lang="{lang}">',
        '<head>',
        '<meta charset="UTF-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
        f'<title>{esc(title)}</title>',
        f'<meta name="description" content="{esc(desc)}">',
        f'<link rel="canonical" href="{abs_url(canon_path)}">',
        f'<link rel="alternate" hreflang="ru" href="{abs_url(ru_path)}">',
        f'<link rel="alternate" hreflang="en" href="{abs_url(en_path)}">',
        f'<link rel="alternate" hreflang="x-default" href="{abs_url(ru_path)}">',
        f'<script>{LANG_JS % (js_json(lang), js_json(alt_path))}</script>',
        '<meta property="og:site_name" content="' + ('Мария Полушкина' if lang == 'ru' else 'Maria Polushkina') + '">',
        f'<meta property="og:url" content="{abs_url(canon_path)}">',
        f'<meta property="og:locale" content="{"ru_RU" if lang == "ru" else "en_US"}">',
        f'<meta property="og:locale:alternate" content="{"en_US" if lang == "ru" else "ru_RU"}">',
        *extra_meta,
        '<link rel="icon" type="image/png" sizes="32x32" href="/site/images/favicon-32.png">',
        '<link rel="icon" type="image/png" sizes="192x192" href="/site/images/favicon-192.png">',
        '<link rel="apple-touch-icon" href="/site/images/apple-touch-icon.png">',
        *[f'<link rel="preload" href="/site/fonts/{f}.woff2" as="font" type="font/woff2" crossorigin>' for f in preload],
        f'<script type="application/ld+json">{js_json(ld)}</script>',
        f'<style>\n{css}</style>',
        '</head>',
    ]
    return '\n'.join(out)


def nav(lang, alt_path):
    ru = lang == 'ru'
    return (
        '<nav class="nav"><div class="ni">'
        f'<a href="{home_path(lang)}" class="nl" aria-label="{"Мария Полушкина - на главную" if ru else "Maria Polushkina - home"}">'
        '<img src="/site/images/logo-signature.svg" alt="M. Polushkina" width="70" height="40"></a>'
        '<div class="nk">'
        f'<a class="na" href="{home_path(lang)}#catalog">{"Каталог" if ru else "Catalogue"}</a>'
        f'<a href="{list_path(lang)}">{"Все работы" if ru else "All works"}</a>'
        f'<a class="lb" href="{alt_path}" data-lang="{"en" if ru else "ru"}" hreflang="{"en" if ru else "ru"}" '
        f'lang="{"en" if ru else "ru"}">{"EN" if ru else "RU"}</a>'
        '</div></div></nav>'
    )


def footer(lang):
    return ('<footer class="ft"><div>© 2026 ' + ('Мария Полушкина' if lang == 'ru' else 'Maria Polushkina') +
            '</div><div class="mk">Made by Dmitry Khinkis</div></footer>\n<div class="toast" id="toast" role="status" aria-live="polite"></div>')


def page_css(ctx):
    return CSS + CSS_COL if ctx.cols else CSS


def col_name(c, lang):
    return c['ru'] if lang == 'ru' else c['en']


def chips_html(ctx, lang, label, skip=None, cls='chips'):
    """Строка кнопок-ссылок на подборки; пусто, если подборок (кроме skip) нет."""
    cs = [c for c, _ in ctx.cols if c['slug'] != skip]
    if not cs:
        return ''
    return (f'<nav class="{cls}" aria-label="{esc(label)}"><b>{esc(label)}:</b>'
            + ''.join(f'<a href="{col_path(c["slug"], lang)}">{esc(col_name(c, lang))}</a>' for c in cs) + '</nav>')


def tail(data):
    return f'<script>window.P={js_json(data)};</script>\n<script>{TAIL_JS % YM_ID}</script>\n</body>\n</html>\n'


def breadcrumbs_html(items):
    parts = []
    for i, (name, path) in enumerate(items):
        if i:
            parts.append('<span aria-hidden="true">/</span>')
        if path is not None and i < len(items) - 1:
            parts.append(f'<a href="{path}">{esc(name)}</a>')
        else:
            parts.append(f'<span aria-current="page">{esc(name)}</span>')
    return '<nav class="bc" aria-label="breadcrumbs">' + ''.join(parts) + '</nav>'


def breadcrumbs_ld(items):
    return {'@type': 'BreadcrumbList', 'itemListElement': [
        {'@type': 'ListItem', 'position': i + 1, 'name': n, 'item': abs_url(p)} for i, (n, p) in enumerate(items)]}


def price_block(a, lang, cls):
    if a.get('sold'):
        return f'<div class="{cls}"><span class="sold">{"Продано" if lang == "ru" else "Sold"}</span></div>'
    if not has_price(a):
        return f'<div class="{cls}">{"Цена по запросу" if lang == "ru" else "Price on request"}</div>'
    p = int(a['p'])
    if lang == 'ru':
        return f'<div class="{cls}">{price_ru(p)}</div>'
    return f'<div class="{cls}">{price_en(p)}<span class="fx" data-p="{p}"></span></div>'


def thumb_img(ctx, a, lang, lazy=True):
    imgs = images_of(a)
    rel = imgs[0] if imgs else f"site/images/work-{a['id']}.jpg"
    sz = ctx.img_size(rel)
    wh = f' width="{sz[0]}" height="{sz[1]}"' if sz else ''
    return (f'<img src="{file_url(rel)}" alt="{esc(work_alt(a, lang))}"{wh}'
            f'{" loading=\"lazy\"" if lazy else ""} decoding="async">')


def card(ctx, a, lang, with_price=True):
    ru = lang == 'ru'
    meta = f"{size_disp(a.get('s', ''))} {'см' if ru else 'cm'}" + (f", {a['y']}" if a.get('y') else '')
    return (f'<a class="cd" href="{page_path(a["id"], lang)}"><div class="th">{thumb_img(ctx, a, lang)}</div>'
            f'<div class="ctt">{esc(title_of(a, lang))}</div><div class="cc">{esc(meta)}</div>'
            + (price_block(a, lang, 'cp') if with_price else '') + '</a>')


# ───────────────────────── страница работы ─────────────────────────

def render_work(ctx, i, lang):
    a = ctx.works[i]
    ru = lang == 'ru'
    wid = a['id']
    path, alt_path = page_path(wid, lang), page_path(wid, 'en' if ru else 'ru')
    t_raw = title_of(a, lang)
    t = cap_t(t_raw)
    mm = medium(a, lang)
    sz = size_disp(a.get('s', ''))
    cm = 'см' if ru else 'cm'
    year = a.get('y')
    name = 'Мария Полушкина' if ru else 'Maria Polushkina'

    title = f'{t} - {name}' + (f', {mm}' if mm else '') + (f', {sz} {cm}' if sz else '')
    facts = ', '.join(x for x in [mm, f'{sz} {cm}' if sz else '', str(year) if year else ''] if x)
    if a.get('sold'):
        ptxt = 'Работа продана.' if ru else 'Sold.'
    elif has_price(a):
        ptxt = f'Цена {price_ru(int(a["p"]))}.' if ru else f'Price {price_en(int(a["p"]))}.'
    else:
        ptxt = ''
    if ru:
        desc = f'«{t}» - оригинальная работа Марии Полушкиной: {facts}. {ptxt} Покупка напрямую у художницы, примерка на стене.'
    else:
        desc = f'“{t}”, an original work by Maria Polushkina: {facts}. {ptxt} Buy directly from the artist, try it on your wall.'
    desc = re.sub(r'\s+', ' ', desc).strip()
    og_desc = ' · '.join(x for x in [facts, ptxt.rstrip('.')] if x)
    wd = (ctx.desc.get(wid) or {}).get(lang, '')   # одобренная видимая подпись; пусто — страница как была
    wa = (ctx.alt.get(wid) or {}).get(lang, '')    # что изображено — только alt главного фото и JSON-LD
    ld_desc = wa or wd or desc        # для разметки — что изображено; подпись короткая и образная
    if wd:                            # подпись Маши — первой, факты и цена остаются (иначе сниппет без сути)
        desc = clip(f'{wd} {desc}')
        og_desc = clip(f"{wd.rstrip('.')} · {og_desc}")
    main_alt = work_alt(a, lang) + (f'. {wa}' if wa else '')

    imgs = images_of(a)
    vids = [m['path'] for m in media_of(a) if m.get('type') == 'vid']
    og = ctx.og.get(wid)
    if og:
        og_img, og_w, og_h = abs_url(og), OG_W, OG_H
    else:
        og_img, og_w, og_h = abs_url('/site/images/og-cover.jpg'), OG_W, OG_H
    og_title = f'{t} - {name}'
    extra = [
        '<meta property="og:type" content="product">',
        f'<meta property="og:title" content="{esc(og_title)}">',
        f'<meta property="og:description" content="{esc(og_desc)}">',
        f'<meta property="og:image" content="{esc(og_img)}">',
        f'<meta property="og:image:width" content="{og_w}">',
        f'<meta property="og:image:height" content="{og_h}">',
        f'<meta property="og:image:alt" content="{esc(work_alt(a, lang))}">',
    ]
    if has_price(a) and not a.get('sold'):
        extra += [f'<meta property="product:price:amount" content="{int(a["p"])}">',
                  '<meta property="product:price:currency" content="RUB">']
    extra += [
        '<meta name="twitter:card" content="summary_large_image">',
        f'<meta name="twitter:title" content="{esc(og_title)}">',
        f'<meta name="twitter:description" content="{esc(og_desc)}">',
        f'<meta name="twitter:image" content="{esc(og_img)}">',
    ]

    crumbs = [('Главная' if ru else 'Home', home_path(lang)), ('Все работы' if ru else 'All works', list_path(lang)), (t, path)]
    person = {'@type': 'Person', '@id': SITE + '/#maria', 'name': 'Мария Полушкина' if ru else 'Maria Polushkina',
              'url': SITE + '/'}
    art = {
        '@type': ['VisualArtwork', 'Product'],
        '@id': abs_url(path) + '#work',
        'name': t,
        'url': abs_url(path),
        'inLanguage': lang,
        'image': [SITE + file_url(p) for p in imgs] or [og_img],
        'description': ld_desc,
        'creator': person,
        'brand': {'@type': 'Brand', 'name': 'Мария Полушкина' if ru else 'Maria Polushkina'},
        'artform': 'Живопись' if ru else 'Painting',
        'sku': f'MP-{wid}',
    }
    if mm:
        art['artMedium'] = mm
    surf = re.split(r',\s*', mm)
    surf = [s for s in surf if re.search(r'холст|canvas|картон|board', s)]
    if surf:
        art['artworkSurface'] = surf[0]
    d = dims(a)
    if d:
        art['height'] = {'@type': 'QuantitativeValue', 'value': num(d[0]), 'unitCode': 'CMT', 'unitText': cm}
        art['width'] = {'@type': 'QuantitativeValue', 'value': num(d[1]), 'unitCode': 'CMT', 'unitText': cm}
    if year:
        art['dateCreated'] = str(year)
    if has_price(a):
        art['offers'] = {'@type': 'Offer', 'price': int(a['p']), 'priceCurrency': 'RUB',
                         'availability': 'https://schema.org/SoldOut' if a.get('sold') else 'https://schema.org/InStock',
                         'itemCondition': 'https://schema.org/NewCondition', 'url': abs_url(path),
                         'seller': {'@id': SITE + '/#maria'}}
    ld = {'@context': 'https://schema.org', '@graph': [art, breadcrumbs_ld(crumbs)]}

    ru_path = page_path(wid, 'ru')
    css = page_css(ctx) + CSS_BUY + (CSS_DESC if wd else '')
    h = [head(lang, title, desc, path, alt_path, ru_path, extra, ld, css), '<body>', nav(lang, alt_path), '<main>',
         breadcrumbs_html(crumbs), '<article class="wk">', '<div class="wm">']

    # главное фото + галерея остальных (переключение лёгким JS; без JS — все фото всё равно в разметке)
    def img_css(sz_):
        r = sz_[0] / sz_[1]
        return f'width:min(100%,calc(78vh*{r:.4f}))'
    if imgs:
        s0 = ctx.img_size(imgs[0])
        wh = f' width="{s0[0]}" height="{s0[1]}" style="{img_css(s0)}"' if s0 else ''
        h.append(f'<img class="main" id="mainImg" src="{file_url(imgs[0])}" alt="{esc(main_alt)}"{wh} fetchpriority="high">')
        if len(imgs) > 1:
            h.append(f'<div class="gal" role="group" aria-label="{"Фото работы" if ru else "Photos"}">')
            for n, p in enumerate(imgs):
                sn = ctx.img_size(p) or (1, 1)
                ga = main_alt if n == 0 else work_alt(a, lang, n)
                h.append(f'<button type="button" class="gt{" on" if n == 0 else ""}" aria-pressed="{"true" if n == 0 else "false"}" '
                         f'data-src="{file_url(p)}" data-w="{sn[0]}" data-h="{sn[1]}" data-css="{img_css(sn)}" '
                         f'data-alt="{esc(ga)}" aria-label="{"Фото" if ru else "Photo"} {n + 1}">'
                         f'<img src="{file_url(p)}" alt="{esc(ga)}" width="{sn[0]}" height="{sn[1]}" loading="lazy" decoding="async"></button>')
            h.append('</div>')
    if vids:
        h.append('<div class="vids">' + ''.join(
            f'<video controls preload="none" playsinline src="{file_url(v)}"></video>' for v in vids) + '</div>')
    h.append('</div>')

    # описание и покупка
    meta = ''.join(f'<span>{esc(x)}</span>' for x in [
        (mm[:1].upper() + mm[1:]) if mm else '', f'{sz} {cm}' if sz else '', str(year) if year else ''] if x)
    msg, subj = buy_text(a, lang, ctx.dm), buy_subj(a, lang)
    chs = channels(ctx, lang)
    ch0 = chs[0]
    if ru:
        btn = 'Хочу похожую' if a.get('sold') else 'Купить в ' + ('письме' if ch0 == 'mail' else ch_name(ch0, lang))
        try_href, try_txt, share_txt = f'/#try-{wid}', 'Примерить на стене', 'Поделиться'
    else:
        btn = 'Ask about similar' if a.get('sold') else 'Buy via ' + ch_name(ch0, lang)
        try_href, try_txt, share_txt = f'/?lang=en#try-{wid}', 'Try on your wall', 'Share'
    icon = {'tg': TG_SVG, 'wa': WA_SVG, 'mail': MAIL_SVG}[ch0]
    buy_html = [ch_link(ctx, ch0, msg, subj, 'mt', f'{icon}<span>{btn}</span>', 'buy')]
    if len(chs) > 1:
        buy_html.append(f'<p class="och">{"или" if ru else "or"} ' + ' · '.join(
            ch_link(ctx, c, msg, subj, '', ch_name(c, lang)) for c in chs[1:]) + '</p>')
    ship = [] if a.get('sold') else [
        f'<p class="ship">{"Доставка по России и в другие страны - стоимость по запросу." if ru else "Worldwide shipping - cost on request."}</p>']
    ds_facts = ', '.join(x for x in [mm, f'{sz} {cm}' if sz else '', str(year) if year else ''] if x)
    if ru:
        ds = (f'Оригинальная работа Марии Полушкиной, {ds_facts}. Работы художницы находятся в частных коллекциях 11 стран. '
              + ('Работа продана - можно спросить художницу о похожих.' if a.get('sold')
                 else 'Покупка напрямую у художницы, примерка на стене - на сайте.'))
    else:
        ds = (f"Original work by Maria Polushkina, {ds_facts}. The artist's works are held in private collections in 11 countries. "
              + ('This work has been sold - you can ask the artist about similar ones.' if a.get('sold')
                 else 'Buy directly from the artist and try the painting on your wall right on the website.'))
    h += ['<div class="wi">', f'<h1>{esc(t_raw)}</h1>', f'<div class="meta">{meta}</div>']
    if wd:
        h.append(f'<p class="wd">{esc(wd)}</p>')
    h += [price_block(a, lang, 'mp'), *ship, *buy_html,
          '<div class="acts">',
          f'<a class="m2 try" href="{try_href}">{TRY_SVG}<span>{try_txt}</span></a>',
          f'<button type="button" class="m2" id="share">{SHARE_SVG}<span>{share_txt}</span></button>',
          '</div>', f'<p class="ds">{esc(ds)}</p>']
    inc = ctx.work_cols.get(wid)
    if inc:
        h.append(f'<p class="inc">{"В подборках" if ru else "In collections"}: ' + ', '.join(
            f'<a href="{col_path(c["slug"], lang)}">{esc(col_name(c, lang))}</a>' for c in inc) + '</p>')

    prev_a = ctx.works[i - 1] if i > 0 else None
    next_a = ctx.works[i + 1] if i + 1 < len(ctx.works) else None
    if prev_a or next_a:
        h.append('<nav class="pn" aria-label="' + ('Соседние работы' if ru else 'Adjacent works') + '">')
        if prev_a:
            h.append(f'<a class="pv" href="{page_path(prev_a["id"], lang)}" rel="prev"><b>← {"Предыдущая" if ru else "Previous"}</b>'
                     f'<span>{esc(title_of(prev_a, lang))}</span></a>')
        if next_a:
            h.append(f'<a class="nx" href="{page_path(next_a["id"], lang)}" rel="next"><b>{"Следующая" if ru else "Next"} →</b>'
                     f'<span>{esc(title_of(next_a, lang))}</span></a>')
        h.append('</nav>')
    h += ['</div>', '</article>']

    # ещё работы: шесть следующих по каталогу (по кругу), без цен — чтобы смена цены не трогала чужие страницы
    n = len(ctx.works)
    more = [ctx.works[(i + k) % n] for k in range(1, min(7, n))]
    if more:
        h += [f'<h2 class="st">{"Ещё работы" if ru else "More works"}</h2>', '<div class="gl"></div>',
              '<div class="grid more">', *[card(ctx, m, lang, with_price=False) for m in more], '</div>',
              f'<div class="all"><a href="{list_path(lang)}">{"Все работы" if ru else "All works"} →</a></div>']
    h += ['</main>', footer(lang)]
    page_title = f'{t} - {name}'
    h.append(tail({'lang': lang, 'id': wid, 'sold': bool(a.get('sold')), 'msg': msg, 'url': abs_url(path), 'title': page_title}))
    return '\n'.join(h)


# ───────────────────────── «Все работы» ─────────────────────────

def render_list(ctx, lang):
    ru = lang == 'ru'
    path, alt_path = list_path(lang), list_path('en' if ru else 'ru')
    n = len(ctx.works)
    if ru:
        title = 'Все работы - Мария Полушкина, картины с ценами'
        desc = (f'Все работы Марии Полушкиной: {n} {plural_ru(n, "картина", "картины", "картин")}, холст, акрил. '
                'Размеры и цены, примерка на стене, покупка напрямую у художницы.')
        lead = ('Оригинальные работы Марии Полушкиной. Работы художницы находятся в частных коллекциях 11 стран. '
                'Покупка напрямую у художницы.')
        h1 = 'Все работы'
        cnt = f'{n} {plural_ru(n, "работа", "работы", "работ")}'
    else:
        title = 'All works - Maria Polushkina, paintings with prices'
        desc = (f'All works by Maria Polushkina: {n} paintings, canvas, acrylic. '
                'Sizes and prices, try a painting on your wall, buy directly from the artist.')
        lead = ("Original works by Maria Polushkina. The artist's works are held in private collections in 11 countries. "
                'Buy directly from the artist.')
        h1 = 'All works'
        cnt = f'{n} works'
    crumbs = [('Главная' if ru else 'Home', home_path(lang)), (h1, path)]
    extra = [
        '<meta property="og:type" content="website">',
        f'<meta property="og:title" content="{esc(title)}">',
        f'<meta property="og:description" content="{esc(desc)}">',
        f'<meta property="og:image" content="{SITE}/site/images/og-cover.jpg">',
        '<meta property="og:image:width" content="1200">',
        '<meta property="og:image:height" content="630">',
        '<meta name="twitter:card" content="summary_large_image">',
        f'<meta name="twitter:title" content="{esc(title)}">',
        f'<meta name="twitter:description" content="{esc(desc)}">',
        f'<meta name="twitter:image" content="{SITE}/site/images/og-cover.jpg">',
    ]
    ld = {'@context': 'https://schema.org', '@graph': [
        {'@type': 'CollectionPage', '@id': abs_url(path), 'url': abs_url(path), 'name': title, 'description': desc,
         'inLanguage': lang, 'author': {'@id': SITE + '/#maria'},
         'mainEntity': {'@type': 'ItemList', 'numberOfItems': n, 'itemListElement': [
             {'@type': 'ListItem', 'position': k + 1, 'url': abs_url(page_path(a['id'], lang)), 'name': cap_t(title_of(a, lang))}
             for k, a in enumerate(ctx.works)]}},
        breadcrumbs_ld(crumbs)]}
    h = [head(lang, title, desc, path, alt_path, list_path('ru'), extra, ld, page_css(ctx)), '<body>', nav(lang, alt_path), '<main>',
         breadcrumbs_html(crumbs), f'<h1 class="st" style="margin-top:8px">{h1}</h1>', '<div class="gl"></div>',
         f'<p class="lead">{esc(lead)} <span style="white-space:nowrap">{cnt}.</span></p>',
         chips_html(ctx, lang, 'Подборки' if ru else 'Collections'),
         '<div class="grid">', *[card(ctx, a, lang) for a in ctx.works], '</div>', '</main>', footer(lang)]
    h = [x for x in h if x]
    h.append(tail({'lang': lang}))
    return '\n'.join(h)


# ───────────────────────── подборки ─────────────────────────

def load_collections(path):
    """Читает collections.json; нет файла — подборок нет. Кривые записи пропускаются с предупреждением."""
    try:
        with open(path, encoding='utf-8') as f:
            raw = json.load(f)
    except FileNotFoundError:
        return []
    if not isinstance(raw, list):
        sys.exit(f'{path}: ожидается список подборок')
    out, seen = [], set()
    for c in raw:
        slug = c.get('slug') if isinstance(c, dict) else None
        if not (isinstance(slug, str) and re.fullmatch(SLUG_RE, slug)) or slug in seen:
            print(f'подборки: пропуск записи со slug={slug!r} (нужны a-z, 0-9, дефис; без повторов)', file=sys.stderr)
            continue
        if not (str(c.get('ru') or '').strip() and str(c.get('en') or '').strip()):
            print(f'подборки: пропуск {slug} - нет названия ru/en', file=sys.stderr)
            continue
        rule, ids = c.get('rule'), c.get('ids')
        if not isinstance(rule, dict) and not isinstance(ids, list):
            print(f'подборки: пропуск {slug} - нет ни rule, ни ids', file=sys.stderr)
            continue
        seen.add(slug)
        out.append({**c, 'ru': str(c['ru']).strip(), 'en': str(c['en']).strip()})
    return out


def load_texts(path, ids, need_ok, required=False):
    """Читает descriptions.json (need_ok: только ok: true, строго булево) или alt.json; нет файла — текстов нет.
    Берёт непустой ru/en. Битый файл — выход с ошибкой (иначе бот снял бы все тексты с сайта)."""
    name = os.path.basename(path)
    try:
        with open(path, encoding='utf-8') as f:
            raw = json.load(f)
    except FileNotFoundError:
        if required:
            sys.exit(f'{path}: файла нет - ничего не меняю (без него сборка сняла бы alt со всех страниц работ)')
        return {}
    except ValueError as e:
        sys.exit(f'{path}: не читается как JSON ({e}) - ничего не меняю')
    if not isinstance(raw, dict):
        sys.exit(f'{path}: ожидается объект {{"<id>": {{"ru", "en", "ok"}}}}')
    out = {}
    for k, v in raw.items():
        if not (isinstance(k, str) and k.isdigit() and isinstance(v, dict)) or need_ok and v.get('ok') is not True:
            continue
        if int(k) not in ids:
            print(f'{name}: №{k} нет в каталоге - пропуск', file=sys.stderr)
            continue
        t = {lang: re.sub(r'\s+', ' ', v[lang]).strip() for lang in ('ru', 'en') if isinstance(v.get(lang), str)}
        t = {lang: x for lang, x in t.items() if x}
        if t:
            out[int(k)] = t
    return out


def col_works(works, c):
    """Работы подборки в порядке каталога."""
    rule = c.get('rule') if isinstance(c.get('rule'), dict) else {}
    ids = c.get('ids')
    if isinstance(ids, list):
        want = {int(x) for x in ids if isinstance(x, int) and not isinstance(x, bool)
                or isinstance(x, str) and x.isdigit()}
    else:
        want = None
    res = []
    for a in works:
        if want is not None and a['id'] not in want:
            continue
        if 'minSide' in rule or 'maxSide' in rule:
            d = dims(a)
            if not d:
                continue
            side = max(d)
            if 'minSide' in rule and side < rule['minSide']:
                continue
            if 'maxSide' in rule and side > rule['maxSide']:
                continue
        if 'minPrice' in rule or 'maxPrice' in rule:
            if a.get('sold') or not has_price(a):
                continue
            if 'minPrice' in rule and a['p'] < rule['minPrice']:
                continue
            if 'maxPrice' in rule and a['p'] > rule['maxPrice']:
                continue
        res.append(a)
    return res


def render_collection(ctx, c, ws, lang):
    ru = lang == 'ru'
    slug = c['slug']
    path, alt_path = col_path(slug, lang), col_path(slug, 'en' if ru else 'ru')
    name = col_name(c, lang)
    n = len(ws)
    title = str(c.get('titleRu' if ru else 'titleEn') or '').strip() or (
        f'{name} - Мария Полушкина, купить оригинал' if ru else f'{name} - Maria Polushkina, original art')
    desc = str(c.get('descRu' if ru else 'descEn') or '').strip() or (
        f'{name}: оригинальные работы Марии Полушкиной с ценами и размерами. Покупка напрямую у художницы.' if ru
        else f'{name}: original works by Maria Polushkina with sizes and prices. Buy directly from the artist.')
    cnt = f'{n} {plural_ru(n, "работа", "работы", "работ")}' if ru else f'{n} works'
    og = ctx.og.get(ws[0]['id'])
    og_img = abs_url(og) if og else f'{SITE}/site/images/og-cover.jpg'
    extra = [
        '<meta property="og:type" content="website">',
        f'<meta property="og:title" content="{esc(title)}">',
        f'<meta property="og:description" content="{esc(desc)}">',
        f'<meta property="og:image" content="{esc(og_img)}">',
        f'<meta property="og:image:width" content="{OG_W}">',
        f'<meta property="og:image:height" content="{OG_H}">',
        f'<meta property="og:image:alt" content="{esc(work_alt(ws[0], lang))}">',
        '<meta name="twitter:card" content="summary_large_image">',
        f'<meta name="twitter:title" content="{esc(title)}">',
        f'<meta name="twitter:description" content="{esc(desc)}">',
        f'<meta name="twitter:image" content="{esc(og_img)}">',
    ]
    crumbs = [('Главная' if ru else 'Home', home_path(lang)), ('Все работы' if ru else 'All works', list_path(lang)),
              (name, path)]
    ld = {'@context': 'https://schema.org', '@graph': [
        {'@type': 'CollectionPage', '@id': abs_url(path), 'url': abs_url(path), 'name': title, 'headline': name,
         'description': desc, 'inLanguage': lang, 'author': {'@id': SITE + '/#maria'},
         'isPartOf': {'@type': 'CollectionPage', '@id': abs_url(list_path(lang))},
         'primaryImageOfPage': og_img,
         'mainEntity': {'@type': 'ItemList', 'numberOfItems': n, 'itemListElement': [
             {'@type': 'ListItem', 'position': k + 1, 'url': abs_url(page_path(a['id'], lang)), 'name': cap_t(title_of(a, lang))}
             for k, a in enumerate(ws)]}},
        breadcrumbs_ld(crumbs)]}
    h = [head(lang, title, desc, path, alt_path, col_path(slug, 'ru'), extra, ld, page_css(ctx)), '<body>',
         nav(lang, alt_path), '<main>', breadcrumbs_html(crumbs),
         f'<h1 class="st" style="margin-top:8px">{esc(name)}</h1>', '<div class="gl"></div>',
         f'<p class="lead">{esc(desc)} <span style="white-space:nowrap">{cnt}.</span></p>',
         '<div class="grid">', *[card(ctx, a, lang) for a in ws], '</div>',
         chips_html(ctx, lang, 'Другие подборки' if ru else 'Other collections', skip=slug, cls='chips bot'),
         f'<div class="all"><a href="{list_path(lang)}">{"Все работы" if ru else "All works"} →</a></div>',
         '</main>', footer(lang)]
    h = [x for x in h if x]
    h.append(tail({'lang': lang}))
    return '\n'.join(h)


# ───────────────────────── превью 1200×630 ─────────────────────────

def make_og(src, dst):
    with Image.open(src) as im0:
        try:
            from PIL import ImageOps
            im = ImageOps.exif_transpose(im0)
        except Exception:
            im = im0
        im = im.convert('RGB')
    pad_x, pad_y = 90, 44
    w, h = im.size
    k = min((OG_W - 2 * pad_x) / w, (OG_H - 2 * pad_y) / h)
    nw, nh = max(1, round(w * k)), max(1, round(h * k))
    im = im.resize((nw, nh), Image.LANCZOS)
    x, y = (OG_W - nw) // 2, (OG_H - nh) // 2
    bg = Image.new('RGB', (OG_W, OG_H), OG_BG)
    # мягкая тень под картиной, как у работ на сайте
    sh = Image.new('L', (OG_W, OG_H), 0)
    sh.paste(70, (x + 4, y + 12, x + nw - 4, y + nh + 12))
    sh = sh.filter(ImageFilter.GaussianBlur(16))
    bg.paste(Image.new('RGB', (OG_W, OG_H), (45, 42, 38)), (0, 0), sh)
    bg.paste(im, (x, y))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + '.tmp'
    bg.save(tmp, 'JPEG', quality=82, optimize=True, progressive=True)
    os.replace(tmp, dst)


# ───────────────────────── sitemap ─────────────────────────

def sitemap(entries):
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" xmlns:xhtml="http://www.w3.org/1999/xhtml" '
           'xmlns:image="http://www.google.com/schemas/sitemap-image/1.1">']
    for e in entries:
        out.append('  <url>')
        out.append(f'    <loc>{esc(abs_url(e["path"]))}</loc>')
        if e.get('lastmod'):
            out.append(f'    <lastmod>{e["lastmod"]}</lastmod>')
        for hl, p in (('ru', e['ru']), ('en', e['en']), ('x-default', e['ru'])):
            out.append(f'    <xhtml:link rel="alternate" hreflang="{hl}" href="{esc(abs_url(p))}"/>')
        for img in e.get('images', []):
            out.append(f'    <image:image><image:loc>{esc(img)}</image:loc></image:image>')
        out.append('  </url>')
    out.append('</urlset>')
    return '\n'.join(out) + '\n'


# ───────────────────────── сборка ─────────────────────────

def write_if_changed(path, data):
    b = data.encode('utf-8') if isinstance(data, str) else data
    try:
        with open(path, 'rb') as f:
            if f.read() == b:
                return False
    except FileNotFoundError:
        pass
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        f.write(b)
    os.replace(tmp, path)
    return True


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description='Статические страницы работ, превью и sitemap.xml')
    ap.add_argument('--root', default=os.path.dirname(os.path.dirname(here)), help='корень репозитория (исходные картинки)')
    ap.add_argument('--catalog', help='catalog.json (по умолчанию <root>/catalog.json)')
    ap.add_argument('--content', help='content.json (по умолчанию <root>/content.json)')
    ap.add_argument('--collections', help='файл подборок (по умолчанию <root>/tools/seo/collections.json; нет файла - нет подборок)')
    ap.add_argument('--descriptions', help='описания работ (по умолчанию <root>/tools/seo/descriptions.json; '
                    'нет файла - нет описаний; на сайт идут только ok: true)')
    ap.add_argument('--alt', help='что изображено (по умолчанию <root>/tools/seo/alt.json; нет файла - нет текстов; '
                    'идёт в alt главного фото и JSON-LD без одобрения)')
    ap.add_argument('--out', help='куда писать (по умолчанию <root>); для проверок — папка в скратчпаде')
    ap.add_argument('--changed', help='файл, куда записать изменённые адреса (по одному на строку)')
    ap.add_argument('--today', help='дата для lastmod, ГГГГ-ММ-ДД (по умолчанию сегодня по UTC)')
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    out = os.path.abspath(args.out or root)
    today = args.today or datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    with open(args.catalog or os.path.join(root, 'catalog.json'), encoding='utf-8') as f:
        raw = json.load(f)
    with open(args.content or os.path.join(root, 'content.json'), encoding='utf-8') as f:
        content = json.load(f)

    works, seen = [], set()
    for a in raw if isinstance(raw, list) else []:
        if not isinstance(a, dict):
            continue
        wid = a.get('id')
        if isinstance(wid, str) and wid.isdigit():
            wid = int(wid)
        if not isinstance(wid, int) or isinstance(wid, bool) or wid < 0 or wid in seen:
            print(f'пропуск работы с id={a.get("id")!r}', file=sys.stderr)
            continue
        seen.add(wid)
        works.append({**a, 'id': wid})
    if not works:
        sys.exit('catalog.json пуст или не читается - ничего не меняю')

    contacts = content.get('contacts') or {}
    ctx = Ctx(root, out, works, contacts)
    ctx.dm = str(contacts.get('telegramDm') or contacts.get('telegram') or 'artpolushkina').lstrip('@')
    ctx.wa = re.sub(r'^8(\d{10})$', r'7\1', re.sub(r'\D', '', str(contacts.get('phone') or '')))   # телефон Маши — WhatsApp
    ctx.email = str(contacts.get('email') or '').strip()

    wids = {a['id'] for a in works}
    ctx.desc = load_texts(args.descriptions or os.path.join(root, 'tools', 'seo', 'descriptions.json'), wids, True)
    ctx.alt = load_texts(args.alt or os.path.join(root, 'tools', 'seo', 'alt.json'), wids, False, required=True)

    # подборки: считаем состав до страниц работ — на них строка «В подборках»
    col_file = args.collections or os.path.join(root, 'tools', 'seo', 'collections.json')
    for c in load_collections(col_file):
        ws = col_works(works, c)
        if len(ws) < MIN_COL:
            print(f'подборка {c["slug"]}: {len(ws)} работ (< {MIN_COL}) - не генерируется', file=sys.stderr)
            continue
        ctx.cols.append((c, ws))
        for a in ws:
            ctx.work_cols.setdefault(a['id'], []).append(c)

    state_path = os.path.join(out, 'tools', 'seo', 'state.json')
    try:
        with open(state_path, encoding='utf-8') as f:
            state = json.load(f)
    except (FileNotFoundError, ValueError):
        state = {}
    old_pages, old_og = state.get('pages', {}), state.get('og', {})
    new_pages, new_og = {}, {}

    # превью: пересчитываем, только если сменилась исходная картинка
    og_made = 0
    for a in works:
        wid = a['id']
        imgs = images_of(a)
        src = os.path.join(root, imgs[0]) if imgs else None
        if not src or not os.path.isfile(src):
            ctx.og[wid] = None
            continue
        with open(src, 'rb') as f:
            hsrc = sha(f.read() + OG_VERSION.encode())[:16]
        dst = os.path.join(out, 'site', 'og', f'work-{wid}.jpg')
        if old_og.get(str(wid)) != hsrc or not os.path.isfile(dst):
            make_og(src, dst)
            og_made += 1
        new_og[str(wid)] = hsrc
        ctx.og[wid] = f'/site/og/work-{wid}.jpg?v={hsrc[:8]}'

    changed, written = [], 0

    def emit(url_path, file_rel, text):
        nonlocal written
        hv = sha(text)[:16]
        prev = old_pages.get(url_path)
        lastmod = prev['lastmod'] if prev and prev.get('h') == hv else today
        if not prev or prev.get('h') != hv:
            changed.append(abs_url(url_path))
        new_pages[url_path] = {'h': hv, 'lastmod': lastmod}
        if write_if_changed(os.path.join(out, file_rel), text):
            written += 1

    for i, a in enumerate(works):
        for lang in ('ru', 'en'):
            p = page_path(a['id'], lang)
            emit(p, p.strip('/') + '/index.html', render_work(ctx, i, lang))
    for lang in ('ru', 'en'):
        p = list_path(lang)
        emit(p, p.strip('/') + '/index.html', render_list(ctx, lang))
    for c, ws in ctx.cols:
        for lang in ('ru', 'en'):
            p = col_path(c['slug'], lang)
            emit(p, p.strip('/') + '/index.html', render_collection(ctx, c, ws, lang))

    # главная: lastmod — когда менялись index.html или данные, которые она показывает
    hb = b''
    for fn in (os.path.join(root, 'index.html'), args.catalog or os.path.join(root, 'catalog.json'),
               args.content or os.path.join(root, 'content.json')):
        try:
            with open(fn, 'rb') as f:
                hb += sha(f.read()).encode()
        except FileNotFoundError:
            pass
    hh = sha(hb)[:16]
    prev = old_pages.get('/')
    new_pages['/'] = {'h': hh, 'lastmod': prev['lastmod'] if prev and prev.get('h') == hh else today}

    # устаревшие страницы и превью (только числовые каталоги/файлы)
    removed = 0
    ids = {str(a['id']) for a in works}
    for base, prefix in (('work', '/work/'), (os.path.join('en', 'work'), '/en/work/')):
        d = os.path.join(out, base)
        if os.path.isdir(d):
            for name in sorted(os.listdir(d)):
                if name.isdigit() and name not in ids and os.path.isdir(os.path.join(d, name)):
                    shutil.rmtree(os.path.join(d, name))
                    changed.append(abs_url(f'{prefix}{name}/'))
                    removed += 1
    ogd = os.path.join(out, 'site', 'og')
    if os.path.isdir(ogd):
        for name in sorted(os.listdir(ogd)):
            m = re.fullmatch(r'work-(\d+)\.jpg', name)
            if m and m.group(1) not in ids:
                os.remove(os.path.join(ogd, name))
    for p in list(old_pages):
        m = re.fullmatch(r'/(?:en/)?work/(\d+)/', p)
        if m and m.group(1) not in ids and abs_url(p) not in changed:
            changed.append(abs_url(p))
    # устаревшие подборки — только внутри collection/ и en/collection/
    slugs = {c['slug'] for c, _ in ctx.cols}
    for base, prefix in (('collection', '/collection/'), (os.path.join('en', 'collection'), '/en/collection/')):
        d = os.path.join(out, base)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if re.fullmatch(SLUG_RE, name) and name not in slugs and os.path.isdir(os.path.join(d, name)):
                shutil.rmtree(os.path.join(d, name))
                if abs_url(f'{prefix}{name}/') not in changed:
                    changed.append(abs_url(f'{prefix}{name}/'))
                removed += 1
        if not os.listdir(d):
            os.rmdir(d)
    for p in list(old_pages):
        m = re.fullmatch(r'/(?:en/)?collection/([^/]+)/', p)
        if m and m.group(1) not in slugs and abs_url(p) not in changed:
            changed.append(abs_url(p))

    # sitemap.xml
    entries = []
    for lang in ('ru', 'en'):
        entries.append({'path': home_path(lang), 'ru': '/', 'en': '/?lang=en', 'lastmod': new_pages['/']['lastmod']})
    for lang in ('ru', 'en'):
        p = list_path(lang)
        entries.append({'path': p, 'ru': '/work/', 'en': '/en/work/', 'lastmod': new_pages[p]['lastmod']})
    for c, _ in ctx.cols:
        for lang in ('ru', 'en'):
            p = col_path(c['slug'], lang)
            entries.append({'path': p, 'ru': col_path(c['slug'], 'ru'), 'en': col_path(c['slug'], 'en'),
                            'lastmod': new_pages[p]['lastmod']})
    for a in works:
        imgs = [SITE + file_url(x) for x in images_of(a) if os.path.isfile(os.path.join(root, x))]
        for lang in ('ru', 'en'):
            p = page_path(a['id'], lang)
            entries.append({'path': p, 'ru': page_path(a['id'], 'ru'), 'en': page_path(a['id'], 'en'),
                            'lastmod': new_pages[p]['lastmod'], 'images': imgs})
    sm_changed = write_if_changed(os.path.join(out, 'sitemap.xml'), sitemap(entries))

    state = {'og': dict(sorted(new_og.items(), key=lambda kv: int(kv[0]))),
             'pages': dict(sorted(new_pages.items()))}
    write_if_changed(state_path, json.dumps(state, ensure_ascii=False, indent=1) + '\n')

    if args.changed:
        with open(args.changed, 'w', encoding='utf-8') as f:
            f.write(''.join(u + '\n' for u in changed))
    print(f'работ: {len(works)}; alt-текстов: {len(ctx.alt)}; одобренных подписей: {len(ctx.desc)}; подборок: {len(ctx.cols)} ('
          + ', '.join(f'{c["slug"]} {len(ws)}' for c, ws in ctx.cols) + f'); страниц записано: {written}; превью пересчитано: {og_made}; '
          f'удалено страниц: {removed}; sitemap.xml {"обновлён" if sm_changed else "без изменений"}; '
          f'изменённых адресов: {len(changed)}', file=sys.stderr)
    for u in changed:
        print(u)


if __name__ == '__main__':
    main()
