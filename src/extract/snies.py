"""Extracción de matrícula de educación superior (SNIES / MEN) desde datos.gov.co."""
from pathlib import Path

from src.extract.common import download_socrata


def extract_snies(cfg: dict) -> dict[str, Path]:
    """Descarga la matrícula por municipio y la matrícula por IES x programa."""
    base = Path(cfg["rutas"]["raw"]) / "snies"
    return {
        "matricula_mpio": download_socrata(cfg["datasets"]["snies_matricula_mpio"], base / "matricula_mpio.csv"),
        "matricula_programa": download_socrata(
            cfg["datasets"]["snies_matricula_programa"], base / "matricula_programa.csv"
        ),
    }
