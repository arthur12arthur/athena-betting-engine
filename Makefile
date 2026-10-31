.PHONY: help install doctor ingest backtest daily publish test lint clean

help:
	@echo "make install   — installe le paquet (pip install -e .)"
	@echo "make doctor    — vérifie les sources de données"
	@echo "make ingest    — télécharge l'historique"
	@echo "make backtest  — backtest walk-forward (3 saisons)"
	@echo "make daily     — pipeline quotidien + dashboard"
	@echo "make test      — tests unitaires"

install:
	pip install -e ".[dev]"

doctor:
	python -m athena.cli doctor

ingest:
	python -m athena.cli ingest

backtest:
	python -m athena.cli backtest --seasons 2024,2025,2026 --render

daily:
	python -m athena.cli daily

publish:
	python -m athena.cli publish

test:
	pytest

lint:
	ruff check athena tests

clean:
	rm -rf data/cache/*.json data/cache/*.csv __pycache__ .pytest_cache
