"""Прогноз оттока клиентов банка (Kaggle Playground Series S4E1)."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("bank-churn")
except PackageNotFoundError:  # pragma: no cover - запуск из исходников без установки
    __version__ = "0.0.0"

__all__ = ["__version__"]
