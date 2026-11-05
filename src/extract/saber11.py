"""Extracción de Saber 11 desde datos.gov.co (dataset unificado)."""
from pathlib import Path

from src.extract.common import download_socrata
from src.transform.saber11 import COLUMN_MAP


def extract_saber11(cfg: dict) -> Path:
    """Descarga solo las columnas necesarias y los periodos de la ventana (más el año previo).

    Las variables explicativas se rezagan un año, por eso se baja desde min(anios) - 1.
    """
    anios = cfg["anios"]
    desde, hasta = (min(anios) - 1) * 10, max(anios) * 10 + 9
    destino = Path(cfg["rutas"]["raw"]) / "saber11" / "saber11.csv"
    return download_socrata(
        cfg["datasets"]["saber11"],
        destino,
        columnas=list(COLUMN_MAP),
        where=f"periodo >= '{desde}' AND periodo <= '{hasta}'",
    )
