"""Позволяет запускать CLI как модуль: ``python -m bank_churn``."""

from bank_churn.cli import app

app(prog_name="bank-churn")
