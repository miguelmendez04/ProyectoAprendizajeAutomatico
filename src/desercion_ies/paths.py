"""Rutas del proyecto, detectadas desde la ubicación de este archivo (no del directorio actual).

La raíz es la carpeta que contiene `pyproject.toml` y `configs/`. Con una instalación editable (`pip install -e .`) se encuentra subiendo desde este
archivo; si el paquete se instaló sin el árbol del repositorio se usa la variable de entorno `DESERCION_ROOT` o, como último recurso, el directorio actual.
"""

from __future__ import annotations

import os
from pathlib import Path


def _es_raiz(p: Path) -> bool:
    return (p / "pyproject.toml").is_file() and (p / "configs").is_dir()


def encontrar_raiz() -> Path:
    """Raíz del proyecto: `DESERCION_ROOT`, o el primer ancestro de este archivo (o del directorio actual) que contenga `pyproject.toml` y `configs/`."""
    env = os.environ.get("DESERCION_ROOT")
    if env:
        return Path(env).resolve()
    for base in (Path(__file__).resolve(), Path.cwd().resolve()):
        for p in (base, *base.parents):
            if _es_raiz(p):
                return p
    raise RuntimeError(
        "no se encontró la raíz del proyecto (carpeta con pyproject.toml y configs/); defina DESERCION_ROOT"
    )


PROJECT_ROOT = encontrar_raiz()
CONFIGS = PROJECT_ROOT / "configs"
CONFIG_FILE = CONFIGS / "config.yaml"
DATA = PROJECT_ROOT / "data"
DATA_RAW = DATA / "raw"
DATA_INTERIM = DATA / "interim"
DATA_PROCESSED = DATA / "processed"
DATASET = DATA_PROCESSED / "dataset_ies.csv"
CHECKSUMS = DATA / "checksums.json"
REPORTS = PROJECT_ROOT / "reports"
TABLES = REPORTS / "tables"
FIGURES = REPORTS / "figures"
REPORTS_ETL = REPORTS / "etl"
REPORTS_HISTORICO = REPORTS / "historico"
NOTEBOOKS = PROJECT_ROOT / "notebooks"
DOCS = PROJECT_ROOT / "docs"
MODELS = PROJECT_ROOT / "models"
TESTS = PROJECT_ROOT / "tests"
