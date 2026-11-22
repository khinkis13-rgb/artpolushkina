#!/usr/bin/env python3
"""Статические страницы работ для поиска и превью ссылок.

Запуск из корня репозитория:  python3 tools/seo/build.py
Читает catalog.json и content.json, НИКОГДА их не переписывает (и index.html тоже). Пишет только своё:
  work/<id>/index.html, en/work/<id>/index.html   — страница работы (ru / en)
  work/index.html, en/work/index.html             — «Все работы»
  site/og/work-<id>.jpg                           — превью 1200×630 для ссылок (одно на оба языка)
  sitemap.xml                                     — карта сайта целиком
  tools/seo/state.json                            — манифест: хеши страниц и даты lastmod, хеши исходников превью
Устаревшие work/<число>/, en/work/<число>/ и site/og/work-<число>.jpg удаляются.
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


def home_path(lang):
    return '/' if lang == 'ru' else '/?lang=en'


def abs_url(path):
    return SITE + path


def file_url(rel):
    return '/' + urllib.parse.quote(rel.lstrip('/'), safe='/')


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode('utf-8')).hexdigest()


def js_json(o):
    return json.dumps(o, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')


class Ctx:
    def __init__(self, root, out, works, contacts):
        self.root, self.out, self.works, self.contacts = root, out, works, contacts
        self._sizes = {}
        self.og = {}   # id → (published path, version tag) or None

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
           "if(w!==pl)location.replace(alt+location.hash);})();")

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
var buy=document.getElementById('buy');
if(buy)buy.addEventListener('click',function(){copyText(P.msg).then(function(ok){if(ok)toast(ru?'Текст сообщения скопирован - вставьте его в чат, если он не появился сам':'Message copied - paste it into the chat if it did not appear');});goal(P.sold?'sold_ask':'buy',{id:P.id});});
var sh=document.getElementById('share');
if(sh)sh.addEventListener('click',function(){goal('share',{id:P.id});
var done=function(){copyText(P.url).then(function(ok){toast(ok?(ru?'Ссылка на работу скопирована':'Link copied'):P.url);});};
if(navigator.share)navigator.share({title:P.title,url:P.url}).catch(function(e){if(!e||e.name!=='AbortError')done();});else done();});
var mi=document.getElementById('mainImg');
document.querySelectorAll('.gt').forEach(function(b){b.addEventListener('click',function(){
if(!mi)return;mi.src=b.getAttribute('data-src');mi.width=+b.getAttribute('data-w');mi.height=+b.getAttribute('data-h');mi.alt=b.getAttribute('data-alt');mi.style.width=b.getAttribute('data-css');
document.querySelectorAll('.gt').forEach(function(x){x.classList.toggle('on',x===b);x.setAttribute('aria-pressed',x===b?'true':'false');});});});
})();"""


def head(lang, title, desc, canon_path, alt_path, ru_path, extra_meta, ld):
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
        f'<style>\n{CSS}</style>',
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
        'description': desc,
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
    h = [head(lang, title, desc, path, alt_path, ru_path, extra, ld), '<body>', nav(lang, alt_path), '<main>',
         breadcrumbs_html(crumbs), '<article class="wk">', '<div class="wm">']

    # главное фото + галерея остальных (переключение лёгким JS; без JS — все фото всё равно в разметке)
    def img_css(sz_):
        r = sz_[0] / sz_[1]
        return f'width:min(100%,calc(78vh*{r:.4f}))'
    if imgs:
        s0 = ctx.img_size(imgs[0])
        wh = f' width="{s0[0]}" height="{s0[1]}" style="{img_css(s0)}"' if s0 else ''
        h.append(f'<img class="main" id="mainImg" src="{file_url(imgs[0])}" alt="{esc(work_alt(a, lang))}"{wh} fetchpriority="high">')
        if len(imgs) > 1:
            h.append(f'<div class="gal" role="group" aria-label="{"Фото работы" if ru else "Photos"}">')
            for n, p in enumerate(imgs):
                sn = ctx.img_size(p) or (1, 1)
                h.append(f'<button type="button" class="gt{" on" if n == 0 else ""}" aria-pressed="{"true" if n == 0 else "false"}" '
                         f'data-src="{file_url(p)}" data-w="{sn[0]}" data-h="{sn[1]}" data-css="{img_css(sn)}" '
                         f'data-alt="{esc(work_alt(a, lang, n))}" aria-label="{"Фото" if ru else "Photo"} {n + 1}">'
                         f'<img src="{file_url(p)}" alt="{esc(work_alt(a, lang, n))}" width="{sn[0]}" height="{sn[1]}" loading="lazy" decoding="async"></button>')
            h.append('</div>')
    if vids:
        h.append('<div class="vids">' + ''.join(
            f'<video controls preload="none" playsinline src="{file_url(v)}"></video>' for v in vids) + '</div>')
    h.append('</div>')

    # описание и покупка
    meta = ''.join(f'<span>{esc(x)}</span>' for x in [
        (mm[:1].upper() + mm[1:]) if mm else '', f'{sz} {cm}' if sz else '', str(year) if year else ''] if x)
    msg = buy_text(a, lang, ctx.dm)
    tg = f'https://t.me/{ctx.dm}?text={urllib.parse.quote(msg, safe="")}'
    if ru:
        btn = 'Хочу похожую' if a.get('sold') else 'Купить в Telegram'
        try_href, try_txt, share_txt = f'/#try-{wid}', 'Примерить на стене', 'Поделиться'
    else:
        btn = 'Ask about similar' if a.get('sold') else 'Buy via Telegram'
        try_href, try_txt, share_txt = f'/?lang=en#try-{wid}', 'Try on your wall', 'Share'
    ds_facts = ', '.join(x for x in [mm, f'{sz} {cm}' if sz else '', str(year) if year else ''] if x)
    if ru:
        ds = (f'Оригинальная работа Марии Полушкиной, {ds_facts}. Работы художницы находятся в частных коллекциях 10 стран. '
              + ('Работа продана - можно спросить художницу о похожих.' if a.get('sold')
                 else 'Покупка напрямую у художницы, примерка на стене - на сайте.'))
    else:
        ds = (f"Original work by Maria Polushkina, {ds_facts}. The artist's works are held in private collections in 10 countries. "
              + ('This work has been sold - you can ask the artist about similar ones.' if a.get('sold')
                 else 'Buy directly from the artist and try the painting on your wall right on the website.'))
    h += ['<div class="wi">', f'<h1>{esc(t_raw)}</h1>', f'<div class="meta">{meta}</div>', price_block(a, lang, 'mp'),
          f'<a class="mt" id="buy" href="{esc(tg)}" target="_blank" rel="noopener">{TG_SVG}<span>{btn}</span></a>',
          '<div class="acts">',
          f'<a class="m2 try" href="{try_href}">{TRY_SVG}<span>{try_txt}</span></a>',
          f'<button type="button" class="m2" id="share">{SHARE_SVG}<span>{share_txt}</span></button>',
          '</div>', f'<p class="ds">{esc(ds)}</p>']

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
        lead = ('Оригинальные работы Марии Полушкиной. Работы художницы находятся в частных коллекциях 10 стран. '
                'Покупка напрямую у художницы.')
        h1 = 'Все работы'
        cnt = f'{n} {plural_ru(n, "работа", "работы", "работ")}'
    else:
        title = 'All works - Maria Polushkina, paintings with prices'
        desc = (f'All works by Maria Polushkina: {n} paintings, canvas, acrylic. '
                'Sizes and prices, try a painting on your wall, buy directly from the artist.')
        lead = ("Original works by Maria Polushkina. The artist's works are held in private collections in 10 countries. "
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
    h = [head(lang, title, desc, path, alt_path, list_path('ru'), extra, ld), '<body>', nav(lang, alt_path), '<main>',
         breadcrumbs_html(crumbs), f'<h1 class="st" style="margin-top:8px">{h1}</h1>', '<div class="gl"></div>',
         f'<p class="lead">{esc(lead)} <span style="white-space:nowrap">{cnt}.</span></p>',
         '<div class="grid">', *[card(ctx, a, lang) for a in ctx.works], '</div>', '</main>', footer(lang)]
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

    # sitemap.xml
    entries = []
    for lang in ('ru', 'en'):
        entries.append({'path': home_path(lang), 'ru': '/', 'en': '/?lang=en', 'lastmod': new_pages['/']['lastmod']})
    for lang in ('ru', 'en'):
        p = list_path(lang)
        entries.append({'path': p, 'ru': '/work/', 'en': '/en/work/', 'lastmod': new_pages[p]['lastmod']})
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
    print(f'работ: {len(works)}; страниц записано: {written}; превью пересчитано: {og_made}; '
          f'удалено страниц: {removed}; sitemap.xml {"обновлён" if sm_changed else "без изменений"}; '
          f'изменённых адресов: {len(changed)}', file=sys.stderr)
    for u in changed:
        print(u)


if __name__ == '__main__':
    main()
