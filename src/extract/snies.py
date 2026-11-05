"""Extracción de matrícula de educación superior (SNIES / MEN) desde datos.gov.co."""
from pathlib import Path

from src.extract.common import download_socrata

# Socrata omite los campos nulos del JSON: se piden las columnas explícitamente.
COLUMNAS_MATRICULA_MPIO = [
    "a_o", "c_digo_delmunicipio", "nombre_del_municipio",
    "tecnica_profesional", "tecnologica", "universitaria", "especializacion", "maestria", "doctorado",
    "ies_con_oferta",
]
COLUMNAS_MATRICULA_PROGRAMA = [
    "c_digo_de_la_instituci_n", "c_digo_snies_delprograma", "c_digo_del_municipio_ies",
    "c_digo_del_municipio_programa", "id_sector", "id_caracter", "a_o", "semestre", "id_g_nero",
    "matriculados_2015",
]


def extract_snies(cfg: dict) -> dict[str, Path]:
    """Descarga la matrícula por municipio y la matrícula por IES x programa."""
    base = Path(cfg["rutas"]["raw"]) / "snies"
    return {
        "matricula_mpio": download_socrata(
            cfg["datasets"]["snies_matricula_mpio"], base / "matricula_mpio.csv", columnas=COLUMNAS_MATRICULA_MPIO
        ),
        "matricula_programa": download_socrata(
            cfg["datasets"]["snies_matricula_programa"], base / "matricula_programa.csv",
            columnas=COLUMNAS_MATRICULA_PROGRAMA,
        ),
    }
