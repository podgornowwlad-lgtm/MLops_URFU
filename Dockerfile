# syntax=docker/dockerfile:1
# Двухэтапная сборка: Poetry нужен только на этапе сборки, в итоговом образе —
# готовое окружение .venv с прод-зависимостями (без dev и notebook групп).

FROM python:3.12-slim AS builder

ENV POETRY_VERSION=2.5.1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PIP_NO_CACHE_DIR=1

RUN pip install "poetry==${POETRY_VERSION}"
WORKDIR /app

# Сначала только манифесты зависимостей — слой кэшируется, пока не меняется poetry.lock.
COPY pyproject.toml poetry.lock README.md ./
RUN poetry install --only main --no-root

COPY src ./src
RUN poetry install --only main


FROM python:3.12-slim AS runtime

RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY --from=builder /app/.venv ./.venv
COPY --from=builder /app/src ./src
COPY configs ./configs

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    CHURN_MODEL_PATH=/app/models/churn_model.joblib \
    CHURN_REPORTS_DIR=/app/reports

USER app
ENTRYPOINT ["bank-churn"]
CMD ["--help"]
