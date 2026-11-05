"""Extracción de DIVIPOLA (DANE) desde datos.gov.co.

DANE (población, NBI) y TerriData no tienen API abierta verificada: se descargan a
mano a `data/raw/dane/` y `data/raw/terridata/` y se leen con `leer_manual`.
"""
from pathlib import Path

import pandas as pd

from src.extract.common import download_socrata


def extract_divipola(cfg: dict) -> Path:
    """Descarga el catálogo oficial de municipios."""
    destino = Path(cfg["rutas"]["raw"]) / "dane" / "divipola.csv"
    return download_socrata(cfg["datasets"]["divipola_municipios"], destino)


def leer_manual(path: str | Path) -> pd.DataFrame | None:
    """Lee un export manual (CSV/XLSX) o devuelve None si el archivo aún no existe."""
    path = Path(path)
    if not path.exists():
        return None
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, dtype=str)
    return pd.read_csv(path, dtype=str)
