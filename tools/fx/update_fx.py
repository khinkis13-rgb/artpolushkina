#!/usr/bin/env python3
"""Курс ЦБ РФ для цен в € и $ на сайте: пишет fx.json в корне репозитория.

    {"date": "YYYY-MM-DD", "src": "cbr", "rub_per": {"EUR": 92.5, "USD": 81.3}}

rub_per — сколько рублей за 1 евро / доллар, до 2 знаков.
Источник — cbr-xml-daily.ru (JSON-зеркало ЦБ), запасной — XML самого ЦБ.
Курс не изменился — файл не трогаем (нет лишних коммитов).
Сеть или ответ сломались — файл не трогаем, выходим с кодом 0: сайт живёт со вчерашним курсом.
Запуск: python3 tools/fx/update_fx.py [путь к fx.json]. Только стандартная библиотека.
"""
import json
import sys
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "fx.json"
CODES = ("EUR", "USD")
UA = {"User-Agent": "Mozilla/5.0 (compatible; fx-updater/1.0)"}


def get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def from_json():
    d = json.loads(get("https://www.cbr-xml-daily.ru/daily_json.js").decode("utf-8"))
    rates = {}
    for c in CODES:
        v = d["Valute"][c]
        rates[c] = float(v["Value"]) / float(v.get("Nominal", 1))
    return d["Date"][:10], rates


def from_xml():
    root = ET.fromstring(get("https://www.cbr.ru/scripts/XML_daily.asp"))
    date = datetime.strptime(root.attrib["Date"], "%d.%m.%Y").strftime("%Y-%m-%d")
    rates = {}
    for v in root.findall("Valute"):
        c = v.findtext("CharCode")
        if c in CODES:
            val = float(v.findtext("Value").replace(",", "."))
            rates[c] = val / float(v.findtext("Nominal").replace(",", "."))
    return date, rates


def main():
    data = None
    for name, fn in (("cbr-xml-daily", from_json), ("cbr.ru XML", from_xml)):
        try:
            date, rates = fn()
            if set(rates) != set(CODES) or not all(10 < r < 1000 for r in rates.values()):
                raise ValueError(f"странные курсы: {rates}")
            data = {"date": date, "src": "cbr",
                    "rub_per": {c: round(rates[c], 2) for c in CODES}}
            break
        except Exception as e:  # сеть, формат, что угодно — пробуем следующий источник
            print(f"fx: {name} не ответил: {e}", file=sys.stderr)
    if data is None:
        print("fx: курс не получен, fx.json не трогаю", file=sys.stderr)
        return 0

    try:
        old = json.loads(OUT.read_text(encoding="utf-8"))
    except Exception:
        old = None
    if old and old.get("rub_per") == data["rub_per"]:
        print(f"fx: курс не изменился ({data['rub_per']}), fx.json не трогаю")
        return 0

    OUT.write_text(json.dumps(data, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"fx: записал {OUT.name}: {data}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
