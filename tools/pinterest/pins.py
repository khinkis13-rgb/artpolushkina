#!/usr/bin/env python3
"""CSV для массовой загрузки пинов в Pinterest (Settings -> Import content -> Upload .csv or .txt file).

Запуск из корня репозитория:
    python3 tools/pinterest/pins.py                       # все работы, публикация сразу после загрузки
    python3 tools/pinterest/pins.py --start 2026-12-01 --per-day 5   # по 5 пинов в день с 1 декабря

Читает catalog.json, tools/seo/collections.json и, если есть, tools/seo/descriptions.json
({"<id>": {"ru": "...", "en": "...", "ok": true}} - en берётся только при ok: true, строго булевом;
--no-descriptions - без него).
Пишет _local/promo/pinterest/pins-en.csv (больше 200 строк - pins-en-2.csv и т.д.: лимит Pinterest 200 пинов на файл).

Формат - по справке Pinterest https://help.pinterest.com/en/business/article/bulk-upload-video-pins
и её образцу https://s.pinimg.com/sub/helpcenter/assets/pinterest-bulk-upload-sample.csv:
колонки Title, Media URL, Pinterest board, Thumbnail, Description, Link, Publish date, Keywords;
Title до 100 знаков, Description до 500, Thumbnail для картинок пустой, доски создаются сами.

Доска для работы (одна работа = один пин): из тематических подборок (с "ids", без "rule"), в которые она входит,
берётся первая по списку BOARD_PRIORITY - сначала узкие серии и сюжеты, в конце широкие ("про двоих", "деревья",
"девушка"); подборки, которых нет в списке, - после него, меньшая раньше. Работа без темы - в DEFAULT_BOARD.
Внешних зависимостей нет.
"""
import argparse
import csv
import datetime as dt
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SITE = 'https://artpolushkina.com/'
UTM = 'utm_source=pinterest&utm_medium=social&utm_campaign=pins'
OUT_DIR = os.path.join(ROOT, '_local', 'promo', 'pinterest')
MAX_ROWS = 200
TITLE_MAX, DESC_MAX = 100, 500
HEADER = ['Title', 'Media URL', 'Pinterest board', 'Thumbnail', 'Description', 'Link', 'Publish date', 'Keywords']

DEFAULT_BOARD = 'Paintings by Maria Polushkina'
BOARD_PRIORITY = ['yoga', 'gold-black', 'sea', 'flowers', 'city', 'hat', 'two', 'trees', 'woman']
BOARD_KEYWORDS = {
    'sea': ['sea painting', 'seascape'],
    'flowers': ['flower painting', 'floral art'],
    'hat': ['woman in a hat', 'figurative painting'],
    'trees': ['tree painting', 'landscape painting'],
    'two': ['couple painting', 'figurative painting'],
    'woman': ['woman painting', 'figurative painting'],
    'city': ['cityscape', 'houses painting'],
    'yoga': ['yoga art', 'yoga painting'],
    'gold-black': ['gold and black art', 'line art'],
}
BASE_KEYWORDS = ['painting', 'acrylic painting', 'original art', 'wall art', 'contemporary art']


def load(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def clean(s):
    """Без длинного тире и лишних пробелов."""
    s = (s or '').replace('—', ' - ').replace('–', '-')
    return re.sub(r'\s+', ' ', s).strip()


def cut(s, n):
    """Обрезать до n знаков по границе предложения, иначе по слову."""
    if len(s) <= n:
        return s
    head = s[:n]
    dot = max(head.rfind('. '), head.rfind('! '), head.rfind('? '))
    if dot > n * 0.5:
        return head[:dot + 1]
    sp = head.rfind(' ')
    return head[:sp if sp > 0 else n].rstrip(' ,;:-')


def fmt_num(x):
    return ('%.1f' % x).rstrip('0').rstrip('.')


def size_text(s):
    """'60×70' (высота×ширина, см) -> '60 x 70 cm (23.6 x 27.6 in)'; диптих '40×30 + 40×30' - по частям."""
    parts = []
    for chunk in (s or '').split('+'):
        nums = re.findall(r'\d+(?:[.,]\d+)?', chunk)
        if len(nums) != 2:
            continue
        h, w = (float(n.replace(',', '.')) for n in nums)
        parts.append('%s x %s cm (%s x %s in)' % (fmt_num(h), fmt_num(w), fmt_num(h / 2.54), fmt_num(w / 2.54)))
    return ' + '.join(parts) if parts else clean(s)


def media_url(w):
    for m in w.get('media') or []:
        if m.get('type') == 'img' and m.get('path'):
            return m['path']
    return 'site/images/work-%d.jpg' % w['id']


def boards_for(works, collections):
    themed = [c for c in collections if isinstance(c.get('ids'), list) and not isinstance(c.get('rule'), dict)]
    order = {slug: i for i, slug in enumerate(BOARD_PRIORITY)}
    themed.sort(key=lambda c: (order.get(c['slug'], len(order)), len(c['ids'])))
    by_id = {}
    for c in themed:
        for i in c['ids']:
            by_id.setdefault(i, c)
    return by_id


def build_rows(works, collections, descriptions, start=None, per_day=0):
    by_id = boards_for(works, collections)
    rows, warnings = [], []
    for n, w in enumerate(works):
        wid = w['id']
        name = clean(w.get('te') or w.get('t') or 'Untitled')
        coll = by_id.get(wid)
        board = clean(coll['en']) if coll else DEFAULT_BOARD

        title = name + ' - Maria Polushkina, original painting'
        if len(title) > TITLE_MAX:
            title = name + ' - Maria Polushkina'
        title = cut(title, TITLE_MAX)

        tech = clean(w.get('me') or w.get('m'))
        size = size_text(w.get('s'))
        facts = ', '.join(x for x in [tech, size] if x)
        sold = bool(w.get('sold'))
        tail = 'Original painting by Maria Polushkina.' + ('' if sold else ' Available at artpolushkina.com')

        d = (descriptions or {}).get(str(wid)) or {}
        if d.get('ok') is True and clean(d.get('en')):
            extra = ' ' + facts + '.'
            desc = cut(clean(d['en']), DESC_MAX - len(extra)) + extra
        else:
            year = w.get('y')
            desc = '%s. %s%s. %s' % (name, facts, (', %s' % year) if year else '', tail)
        desc = cut(desc, DESC_MAX)

        path = media_url(w)
        if not os.path.exists(os.path.join(ROOT, path)):
            warnings.append('#%d: нет файла %s' % (wid, path))

        kw = list(BASE_KEYWORDS)
        if coll:
            kw += BOARD_KEYWORDS.get(coll['slug'], [clean(coll['en']).lower()])
        if 'gold' in tech.lower() and 'gold and black art' not in kw:
            kw.append('gold leaf art' if 'leaf' in tech.lower() else 'gold thread art')
        kw = list(dict.fromkeys(kw))

        pub = ''
        if start and per_day > 0:
            pub = (start + dt.timedelta(days=n // per_day)).isoformat()

        rows.append({
            'Title': title,
            'Media URL': SITE + path,
            'Pinterest board': board,
            'Thumbnail': '',
            'Description': desc,
            'Link': '%sen/work/%d/?%s' % (SITE, wid, UTM),
            'Publish date': pub,
            'Keywords': ', '.join(kw),
        })
    return rows, warnings


def write(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    files = []
    for k in range(0, max(len(rows), 1), MAX_ROWS):
        part = k // MAX_ROWS + 1
        name = 'pins-en.csv' if part == 1 else 'pins-en-%d.csv' % part
        path = os.path.join(out_dir, name)
        with open(path, 'w', encoding='utf-8', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=HEADER, lineterminator='\r\n')
            wr.writeheader()
            wr.writerows(rows[k:k + MAX_ROWS])
        files.append(path)
    return files


def main():
    ap = argparse.ArgumentParser(description='CSV пинов Pinterest из catalog.json')
    ap.add_argument('--start', help='дата первого пина YYYY-MM-DD (по умолчанию - публиковать сразу)')
    ap.add_argument('--per-day', type=int, default=0, help='пинов в день при --start')
    ap.add_argument('--out', default=OUT_DIR, help='папка для CSV')
    ap.add_argument('--no-descriptions', action='store_true', help='не брать descriptions.json, только шаблон')
    a = ap.parse_args()
    start = dt.date.fromisoformat(a.start) if a.start else None
    if start and a.per_day <= 0:
        ap.error('--start нужен вместе с --per-day N')

    works = load(os.path.join(ROOT, 'catalog.json'), [])
    works = works if isinstance(works, list) else works.get('works', [])
    collections = load(os.path.join(ROOT, 'tools', 'seo', 'collections.json'), [])
    descriptions = {} if a.no_descriptions else load(os.path.join(ROOT, 'tools', 'seo', 'descriptions.json'), {})
    descriptions = descriptions if isinstance(descriptions, dict) else {}

    rows, warnings = build_rows(works, collections, descriptions, start, a.per_day)
    files = write(rows, a.out)
    for wmsg in warnings:
        print('внимание:', wmsg, file=sys.stderr)
    counts = {}
    for r in rows:
        counts[r['Pinterest board']] = counts.get(r['Pinterest board'], 0) + 1
    used = sum(1 for d in descriptions.values() if isinstance(d, dict) and d.get('ok') is True and clean(d.get('en')))
    print('пинов: %d, одобренных en-описаний: %d' % (len(rows), used))
    for b, c in sorted(counts.items(), key=lambda x: -x[1]):
        print('  %3d  %s' % (c, b))
    for p in files:
        print('файл:', os.path.relpath(p, ROOT))


if __name__ == '__main__':
    main()
