"""Extracción de DIVIPOLA (DANE) desde datos.gov.co.

Población y NBI vienen de TerriData (`src/extract/terridata.py`), no de archivos del DANE.
"""
from pathlib import Path

from src.extract.common import download_socrata


def extract_divipola(cfg: dict) -> Path:
    """Descarga el catálogo oficial de municipios."""
    destino = Path(cfg["rutas"]["raw"]) / "dane" / "divipola.csv"
    return download_socrata(
        cfg["datasets"]["divipola_municipios"], destino,
        columnas=["cod_dpto", "dpto", "cod_mpio", "nom_mpio", "tipo_municipio", "longitud", "latitud"],
    )

