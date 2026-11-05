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
# Catálogo oficial de IES: nombre, sector y si es sede principal o seccional.
COLUMNAS_IES = [
    "c_digo_instituci_n", "nombre_instituci_n", "principal_seccional", "sector", "car_cter_acad_mico",
    "departamento_domicilio", "municipio_domicilio",
]


def extract_ies_catalogo(cfg: dict) -> Path:
    """Descarga el catálogo de IES (`MEN_INSTITUCIONES EDUCACIÓN SUPERIOR`)."""
    return download_socrata(
        cfg["datasets"]["snies_ies"], Path(cfg["rutas"]["raw"]) / "snies" / "ies_catalogo.csv", columnas=COLUMNAS_IES
    )


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
