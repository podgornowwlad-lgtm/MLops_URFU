"""Командная строка ``bank-churn``: compare / train / predict."""

from pathlib import Path
from typing import Annotated

import typer

from bank_churn.config import ExperimentConfig, Settings, load_experiment
from bank_churn.logging_utils import configure_logging
from bank_churn.workflows import run_comparison, run_prediction, run_training

app = typer.Typer(
    name="bank-churn",
    help="Прогноз оттока клиентов банка. Пути и параметры окружения — переменные CHURN_*.",
    no_args_is_help=True,
    add_completion=False,
)

DEFAULT_OUTPUT = Path("reports/predictions.csv")

ConfigOption = Annotated[
    Path | None,
    typer.Option(
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        help="YAML эксперимента (по умолчанию CHURN_CONFIG_PATH).",
    ),
]


def _bootstrap(config: Path | None) -> tuple[Settings, ExperimentConfig]:
    settings = Settings()
    configure_logging(settings.log_level)
    return settings, load_experiment(config or settings.config_path)


@app.command()
def compare(config: ConfigOption = None) -> None:
    """Сравнить модели-кандидаты (GridSearchCV) и сохранить таблицу метрик."""
    settings, cfg = _bootstrap(config)
    table = run_comparison(settings, cfg)
    typer.echo(f"Лучшая модель по CV: {table.loc[0, 'model']}")


@app.command()
def train(
    config: ConfigOption = None,
    n_trials: Annotated[
        int | None, typer.Option(min=1, help="Переопределить число итераций Optuna.")
    ] = None,
) -> None:
    """Подобрать гиперпараметры финальной модели, оценить на тесте и сохранить."""
    settings, cfg = _bootstrap(config)
    if n_trials is not None:
        tuning = cfg.tuning.model_copy(update={"n_trials": n_trials})
        cfg = cfg.model_copy(update={"tuning": tuning})
    metadata = run_training(settings, cfg)
    typer.echo(f"Модель сохранена: {settings.model_path}")
    typer.echo(f"Метрики на тесте: {metadata['test_metrics']}")


@app.command()
def predict(
    input_path: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="CSV с сырыми признаками.")
    ],
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Куда сохранить прогноз.")
    ] = DEFAULT_OUTPUT,
    threshold: Annotated[
        float | None,
        typer.Option(min=0.0, max=1.0, help="Порог классификации (по умолчанию — из модели)."),
    ] = None,
) -> None:
    """Посчитать вероятность оттока для клиентов из CSV."""
    settings = Settings()
    configure_logging(settings.log_level)
    predictions = run_prediction(settings, input_path, output, threshold)
    typer.echo(f"Сохранено {len(predictions)} прогнозов: {output}")
