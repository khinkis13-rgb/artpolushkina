# artpolushkina.com — сайт художницы Марии Полушкиной

Клиентский проект Дмитрия (Маша Полушкина, Telegram `@m_polushkina`, id 786726716).
Просьбы по сайту приходят от неё в Telegram.

## Как устроено

- **Хостинг:** GitHub Pages из корня ветки `main` репозитория
  `khinkis13-rgb/artpolushkina` (до 30.10.2026 назывался `-Users-dmitrykhinkis-Documents-CURSOR-FLAG-ART`),
  домен — файл `CNAME`.
  Сборки нет: что запушено в `main`, то через ~1 мин на сайте.
- **`index.html`** — весь сайт (HTML + CSS + JS в одном файле). Данные тянет при загрузке
  из `catalog.json` и `content.json`; если они не загрузились — берёт запасные `W_FB` / `CT_FB`.
- **`catalog.json`** — работы: `id`, `t`/`te` (название ru/en), `m`/`me` (техника),
  `s` (размер «высота×ширина» в см), `y` (год), `p` (цена ₽), `media` (`[{type:'img'|'vid', path}]`).
  Без `media` берётся `site/images/work-<id>.jpg`.
- **`content.json`** — биография, выставки, публикации, контакты.
- **`site/images/`** — все живые картинки (работы `work-*.jpg`, публикации `pub-*.jpg`, фото художницы).
- **`admin.html`** — админка (artpolushkina.com/admin.html). Вход по GitHub fine-grained токену,
  правки коммитит прямо в `main` через GitHub API (отсюда сотни коммитов «Обновлена работа #N»).
  Картинки грузит в `site/images/`, видео — в `site/videos/`.

## Как менять

- Тексты, цены, работы — через админку или правкой JSON.
- Перед правкой всегда `git pull`: админка коммитит в `main` в обход локальной копии.
- Проверка локально: `python3 -m http.server 8765` в корне → http://localhost:8765/.
- Каталог раскладывается по колонкам в JS (`rC` → `catCols`, `catRatio`): картины показываются
  в реальных пропорциях, без обрезки. Не возвращать `aspect-ratio:1` / `object-fit:cover`.

## Удалённое наследие

30.10.2026 удалены неиспользуемые `images/art-*.jpg`, старая `site/index.html` и дубль
`site/images/pub-inmyroom-2023.jpg.png` — при необходимости достать из git-истории.

## Знак

В футере — микро-подпись `Made by Dmitry Khinkis` (клиентский проект).
