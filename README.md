# Bank Churn — прогноз оттока клиентов банка

Бинарная классификация оттока (`Exited`) на данных
[Kaggle Playground Series S4E1](https://www.kaggle.com/competitions/playground-series-s4e1/data).
Проект — production-версия исследовательского ноутбука «Кейс 1»
([`notebooks/case_1_research.ipynb`](notebooks/case_1_research.ipynb)).

**Ключевая метрика — recall:** удержание клиента дешёвое, а потеря дорогая, поэтому
важно найти как можно больше уходящих клиентов. Финальная модель — CatBoost,
гиперпараметры подбираются Optuna.

---

## Структура

```
bank_churn_mlops/
├── src/bank_churn/           # устанавливаемый пакет (src-layout)
│   ├── config.py             # Settings (env CHURN_*, .env) + ExperimentConfig (YAML, pydantic)
│   ├── data.py               # чтение CSV, проверка схемы, стратифицированный split
│   ├── transformers.py       # sklearn-трансформеры: FeatureSelector, BalanceFlagger
│   ├── modeling.py           # реестр моделей и сборка Pipeline
│   ├── training.py           # GridSearchCV по кандидатам, Optuna для финальной модели
│   ├── evaluation.py         # метрики, classification report, важности признаков
│   ├── artifacts.py          # ModelBundle: модель + порог + паспорт (JSON)
│   ├── inference.py          # ChurnPredictor — скоринг сырых данных
│   ├── workflows.py          # сценарии compare / train / predict
│   └── cli.py                # консольная команда `bank-churn` (typer)
├── configs/train.yaml        # колонки, сетки гиперпараметров, пространство Optuna
├── tests/{unit,integration}/ # pytest на синтетических данных, покрытие ≥ 85 %
├── notebooks/                # EDA и исследование (в git — без выводов)
├── data/ models/ reports/    # артефакты: в git только .gitkeep
├── pyproject.toml            # зависимости + настройки black/isort/flake8/mypy/bandit/pytest + задачи poe
├── poetry.lock               # точные версии всех пакетов
├── poetry.toml               # .venv внутри проекта
├── .python-version           # 3.12
├── .pre-commit-config.yaml   # git-хуки
├── .gitlab-ci.yml            # lint → test → build → train
├── Dockerfile                # образ с прод-зависимостями
├── .env.example              # шаблон переменных окружения
└── .vscode/                  # VS Code использует ./.venv и форматирует при сохранении
```

## Быстрый старт

Нужны Python 3.11–3.13 и [Poetry](https://python-poetry.org/docs/#installation) ≥ 2.0.

```bash
git clone <repo> && cd bank_churn_mlops
poetry install                 # создаёт ./.venv по poetry.lock
poetry run poe hooks           # ставит git-хуки pre-commit и pre-push
cp .env.example .env           # при необходимости поправить пути
```

Положите `train.csv` из Kaggle в `data/raw/train.csv` (или укажите URL/путь в `CHURN_DATA_PATH`).

```bash
poetry run poe compare                       # сравнить модели → reports/model_comparison.csv
poetry run poe train                         # обучить CatBoost + Optuna → models/, reports/
poetry run poe predict data/raw/test.csv     # прогноз → reports/predictions.csv

poetry run bank-churn --help                 # все команды и опции
poetry run bank-churn train --n-trials 5     # быстрый прогон
```

Все задачи — `poetry run poe --help`
(poethepoet вместо Makefile: работает одинаково на Windows и Linux).

| Задача | Что делает |
|---|---|
| `poe format` | isort + black |
| `poe lint` | все pre-commit хуки по всем файлам |
| `poe typecheck` | mypy |
| `poe security` | bandit |
| `poe test` / `poe test-unit` | все тесты с покрытием / только быстрые unit |
| `poe check` | lint + test — то же, что проверяет CI |

### Артефакты обучения

| Файл | Содержимое |
|---|---|
| `models/churn_model.joblib` | `ModelBundle`: sklearn-пайплайн, порог, метаданные |
| `models/churn_model.meta.json` | паспорт модели: дата, версия пакета, признаки, параметры, метрики |
| `reports/metrics.json` | то же — для CI-артефактов и сравнения запусков |
| `reports/classification_report.txt` | отчёт по классам на отложенной выборке |
| `reports/feature_importances.csv` | важности признаков CatBoost |

Модель принимает **сырые** данные в формате Kaggle CSV. Технические колонки и
лишние поля отбрасываются внутри пайплайна, а порядок колонок может быть любым.

## Виртуальное окружение и Git

| Что | Где | В git? |
|---|---|---|
| Описание зависимостей | `pyproject.toml` (группы `main`, `dev`, `notebook`) | да |
| Точные версии | `poetry.lock` | да |
| Расположение окружения | `poetry.toml` → `in-project = true` → `./.venv` | да |
| Версия Python | `.python-version`, `requires-python` | да |
| Интерпретатор для IDE | `.vscode/settings.json` → `./.venv` | да |
| Само окружение | `.venv/` | **нет** (`.gitignore`) |
| Секреты, локальные пути | `.env` (шаблон `.env.example`) | **нет** |

Окружение не хранится в репозитории: оно тяжёлое и зависит от ОС. Вместо него в git
лежит его точное описание, и `poetry install` воспроизводит одно и то же окружение
на любой машине, в CI (кэш по хэшу `poetry.lock`) и в Docker.

- Добавить зависимость: `poetry add <pkg>`, для разработки — `poetry add --group dev <pkg>`.
  Коммитить оба файла: `pyproject.toml` и `poetry.lock`.
- Хук `poetry-check` не даст закоммитить `pyproject.toml`, рассинхронизированный с lock-файлом.
- Для ноутбука: `poetry install --with notebook` (jupyter, seaborn, phik, shap).
  В прод-окружение и Docker-образ эти пакеты не попадают.

## Качество кода

| Инструмент | Назначение | Где настроен |
|---|---|---|
| black | форматирование (строка 100) | `pyproject.toml` |
| isort | порядок импортов, профиль black | `pyproject.toml` |
| flake8 + bugbear + pep8-naming | PEP 8, баги, именование, сложность ≤ 8 | `pyproject.toml` (через flake8-pyproject) |
| mypy + pydantic plugin | статическая типизация `src/` и `tests/` | `pyproject.toml` |
| bandit | поиск проблем безопасности | `pyproject.toml` |
| pytest + coverage | тесты, порог покрытия 85 % | `pyproject.toml` |
| nbstripout | удаляет выводы ноутбуков перед коммитом | `.pre-commit-config.yaml` |
| pre-commit-hooks | пробелы, EOL, YAML/TOML/JSON, большие файлы, ключи, запрет коммита в `main` | `.pre-commit-config.yaml` |

Хуки срабатывают на двух этапах:

- **pre-commit** — форматтеры, линтеры, mypy, проверка lock-файла;
- **pre-push** — быстрые unit-тесты.

## Ветвление (GitLab Flow)

- `main` — стабильная версия. Прямые коммиты запрещены хуком `no-commit-to-branch`.
- Задачи делаются в ветках `feature/<описание>` и вливаются через Merge Request
  после зелёного CI и код-ревью.
- CI (`.gitlab-ci.yml`):
  - на MR — `lint` и `test`;
  - в `main` — дополнительно сборка Docker-образа;
  - обучение `train-model` запускается вручную, модель и метрики сохраняются как артефакты пайплайна.

## Что изменилось по сравнению с ноутбуком

| Было в ноутбуке | Стало |
|---|---|
| `!pip install` в ячейке, версии не зафиксированы | Poetry + `poetry.lock`, отдельные группы зависимостей |
| Данные по захардкоженному URL | `CHURN_DATA_PATH` (путь или URL) + проверка схемы и таргета |
| Колонки, сетки, сиды разбросаны по ячейкам | `configs/train.yaml`, валидация pydantic (опечатка в ключе — ошибка) |
| `FunctionTransformer(drop_non_informative)` | `FeatureSelector`: явный список признаков, фиксированный порядок, понятная ошибка при нехватке колонок |
| `FunctionTransformer(add_has_balance)` | `BalanceFlagger` — клонируемый трансформер с `get_feature_names_out` |
| Четыре скопированных блока GridSearch с переменными `DecisionTreeMetrics_info`, `RandomForestResults_info`… | один цикл `compare_candidates` → таблица с CV/train-метриками, разрывом train−CV и временем |
| `OptunaSearchCV` (experimental API) | `optuna.create_study` + TPE с сидом, пространство поиска в YAML |
| Ручное удаление `HasCrCard`, `EstimatedSalary`, `Tenure` из `X_train`/`X_test` | `tuning.excluded_features` — удаление внутри пайплайна, модель принимает полный CSV |
| `print(classification_report(...))` | `logging` + отчёты в `reports/` |
| Модель живёт в памяти ноутбука | `ModelBundle` в joblib + JSON-паспорт; `ChurnPredictor` для инференса |
| Нет тестов | 43 теста (unit + integration), покрытие ~99 % |

Пространство поиска глубины CatBoost сужено с 4–12 до 4–10: лучшая модель в ноутбуке
получилась с `depth=4`, а глубина 12 увеличивает время обучения в разы.

EDA, графики и SHAP-анализ остаются в ноутбуке. Это исследовательская часть, в
продовый пайплайн она не входит.
