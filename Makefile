# Atajos del proyecto. En Windows (PowerShell) los equivalentes están en el README, sección «Equivalentes en PowerShell».
PYTHON ?= python
SRC    := src

.PHONY: help install install-dev requirements test test-unit test-integration lint format etl reproduce reproduce-fast notebooks clean

help:  ## lista los objetivos
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-18s %s\n", $$1, $$2}'

install:  ## instala el paquete (comando `desercion`) con las versiones fijas
	$(PYTHON) -m pip install -e .

install-dev:  ## instala el paquete con las herramientas de desarrollo (pytest, ruff, pre-commit, nbconvert)
	$(PYTHON) -m pip install -e ".[dev]"

requirements:  ## regenera requirements.txt desde pyproject.toml
	$(PYTHON) tools/generar_requirements.py

test:  ## todas las pruebas (unitarias e integración)
	$(PYTHON) -m pytest -q

test-unit:  ## solo las unitarias (no necesitan data/processed ni los resultados publicados)
	$(PYTHON) -m pytest -q -m "not integration"

test-integration:  ## solo las de integración
	$(PYTHON) -m pytest -q -m integration

lint:  ## ruff check
	$(PYTHON) -m ruff check src tests tools notebooks/construir

format:  ## ruff format (formatea el código)
	$(PYTHON) -m ruff format src tests tools notebooks/construir

etl:  ## ETL desde data/raw (sin descargar de nuevo)
	$(PYTHON) -m desercion_ies etl --saltar-extract

reproduce:  ## corrida completa (≈ 1,5 h)
	$(PYTHON) -m desercion_ies reproducir --etapa todo

reproduce-fast:  ## verificación rápida (< 10 min); escribe en reproduccion/ y compara con lo publicado
	$(PYTHON) -m desercion_ies reproducir --etapa todo --rapido

notebooks:  ## regenera y ejecuta los cuadernos (01 y 02 necesitan data/raw)
	for f in notebooks/construir/construir_*.py; do $(PYTHON) $$f || exit 1; done

clean:  ## borra cachés y salidas temporales (no toca datos ni resultados)
	rm -rf reproduccion .pytest_cache .ruff_cache build *.egg-info src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf reports/tables/_parcial_ajuste reports/tables/_parcial_sesgo reports/tables/_parcial_complejidad
