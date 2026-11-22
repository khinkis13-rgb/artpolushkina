#!/usr/bin/env python3
"""IndexNow: сообщить Яндексу и Bing (через api.indexnow.org) об изменённых адресах.

Запуск из корня репозитория после push:  python3 tools/seo/indexnow.py <changed.txt> [--wait 300] [--dry-run]
<changed.txt> — список адресов от tools/seo/build.py --changed (по одному на строку). Пустой — ничего не делает.
Сначала ждёт, пока GitHub Pages начнёт отдавать новую версию первого изменённого адреса (тело ответа совпадает
с локальным файлом), затем шлёт POST на оба адреса IndexNow. Никогда не завершается с ошибкой: IndexNow —
приятное дополнение, ронять из-за него сборку незачем; коды ответов печатает.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.dont_write_bytecode = True  # не оставлять __pycache__ в tools/seo
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build import HOST, INDEXNOW_KEY, SITE  # noqa: E402

ENDPOINTS = ['https://yandex.com/indexnow', 'https://api.indexnow.org/indexnow']
UA = 'artpolushkina-seo-bot'


def local_file(url, root):
    if not url.startswith(SITE + '/'):
        return None
    p = url[len(SITE):].split('?')[0]
    if p.endswith('/'):
        p += 'index.html'
    f = os.path.join(root, p.lstrip('/'))
    return f if os.path.isfile(f) else None


def fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent': UA, 'Cache-Control': 'no-cache'})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b''
    except Exception as e:  # сеть, таймаут
        return 0, str(e).encode()


def wait_deployed(urls, root, wait):
    probe = next(((u, f) for u in urls for f in [local_file(u, root)] if f), None)
    if not probe:
        print('нет локального файла для проверки выкладки - жду 90 с')
        time.sleep(min(90, wait))
        return
    url, f = probe
    with open(f, 'rb') as fh:
        want = fh.read()
    t0 = time.time()
    while True:
        code, body = fetch(f'{url}?nocache={int(time.time())}')
        if code == 200 and body == want:
            print(f'выложено: {url} ({int(time.time() - t0)} с)')
            return
        if time.time() - t0 > wait:
            print(f'не дождался новой версии {url} за {wait} с (последний код {code}) - пингую всё равно')
            return
        time.sleep(10)


def ping(urls, dry):
    body = json.dumps({'host': HOST, 'key': INDEXNOW_KEY, 'keyLocation': f'{SITE}/{INDEXNOW_KEY}.txt',
                       'urlList': urls[:10000]}).encode()
    for ep in ENDPOINTS:
        if dry:
            print(f'[dry-run] POST {ep}: {len(urls)} адресов')
            continue
        req = urllib.request.Request(ep, data=body, method='POST',
                                     headers={'Content-Type': 'application/json; charset=utf-8', 'User-Agent': UA})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                print(f'IndexNow {ep}: {r.status}')
        except urllib.error.HTTPError as e:
            print(f'IndexNow {ep}: {e.code} {e.read()[:300]!r}')
        except Exception as e:
            print(f'IndexNow {ep}: ошибка {e}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('changed')
    ap.add_argument('--root', default=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    ap.add_argument('--wait', type=int, default=300, help='сколько секунд ждать выкладки Pages')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    try:
        with open(a.changed, encoding='utf-8') as f:
            urls = list(dict.fromkeys(u.strip() for u in f if u.strip().startswith(SITE + '/')))
    except FileNotFoundError:
        urls = []
    if not urls:
        print('изменённых адресов нет - IndexNow не нужен')
        return
    print(f'изменённых адресов: {len(urls)}')
    if not a.dry_run:
        wait_deployed(urls, a.root, a.wait)
    ping(urls, a.dry_run)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'IndexNow: непредвиденная ошибка {e}')
    sys.exit(0)
