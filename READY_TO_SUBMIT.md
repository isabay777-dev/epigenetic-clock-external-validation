# Статья №3 — готова к подаче. Твои 5 шагов

Статья, анализ, пакет и репозиторий готовы. Ниже — ровно то, что осталось сделать тебе (это публичные/необратимые действия под твоим именем, поэтому только ты).

Рабочая папка:
`/Users/issabayyesmagambetov/Documents/Claude/Projects/longevity-genomics-corpus/article3-methylation-clock`

---

## Шаг 1. Дата в сопроводительном письме
Открой `submission/COVER_LETTER.md`, замени `[DATE]` на сегодняшнюю дату. (В .docx версии тоже — или пересоберу по просьбе.)

## Шаг 2. Проверка на плагиат (iThenticate / Turnitin)
Прогони `submission/BLINDED_MANUSCRIPT.docx` через iThenticate (обычно доступ через твой университет). Цель: низкий процент совпадений. Текст оригинальный, данные публичные — проблем быть не должно.

## Шаг 3. GitHub (публикация кода)
Код уже закоммичен локально (38 файлов, чистый репозиторий). Тебе:
1. Создай пустой репозиторий на github.com (например `epigenetic-clock-external-validation`), без README.
2. В терминале, находясь в рабочей папке, выполни (подставь свой username):
   ```
   git remote add origin https://github.com/<ТВОЙ_USERNAME>/epigenetic-clock-external-validation.git
   git branch -M main
   git push -u origin main
   ```
   Или через gh CLI одной командой:
   ```
   gh repo create epigenetic-clock-external-validation --public --source=. --push
   ```

## Шаг 4. Zenodo (архив + DOI)
1. Зайди на zenodo.org (вход через тот же GitHub или ORCID).
2. Либо включи GitHub-интеграцию Zenodo и сделай релиз `v1.0.0` в репозитории — Zenodo автоматически создаст DOI; либо загрузи zip архива вручную (New upload).
3. Скопируй полученный DOI (вид `10.5281/zenodo.XXXXXXX`).
4. Вставь его в ДВА места:
   - `CITATION.cff` — поле `doi:` (сейчас там `[DOI PLACEHOLDER - Zenodo]`)
   - в манускрипт, раздел Data/Code availability
   Скажи мне DOI — я впишу за секунду и пересоберу пакет.

## Шаг 5. Подача в журнал (Biogerontology)
1. Editorial Manager журнала Biogerontology (сайт Springer).
2. Загрузи РАЗДЕЛЬНО:
   - `submission/TITLE_PAGE.docx` (с автором) — как Title Page
   - `submission/BLINDED_MANUSCRIPT.docx` (без автора) — как Blinded/Anonymous Manuscript
   - `submission/COVER_LETTER.docx` — как Cover Letter
   - 5 фигур из `figures/` (если система просит отдельными файлами, 300 dpi)
3. Тип статьи: Research. Подтверди single-author и декларации.

---

## Что уже проверено (тебе не нужно)
- Числа сверены с результатами анализа — 0 расхождений.
- Blinded-версия: leak-check чистый (0 идентификаторов автора).
- Формат Biogerontology (абстракт 150-250, keywords, декларации) — пройден.
- Главный вывод устоял под bootstrap-CI.
- Ноль AI-лексики, ноль em-dash.

Как сделаешь DOI (шаг 4) — верни его мне, и я закрою последнюю зависимость в тексте.
