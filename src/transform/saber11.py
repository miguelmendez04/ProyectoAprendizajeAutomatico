"""Transformación de Saber 11 (dataset unificado `kgxf-xxbe` de datos.gov.co).

El dataset trae todos los periodos con un mismo esquema, por lo que el mapeo de
columnas es único (la guía anticipaba un mapeo por periodo; ver ledger).
"""
from pathlib import Path

import pandas as pd

from src.homologation.divipola import normalizar_codigo

# Columna original -> nombre interno. Si falta alguna, se falla con error claro.
COLUMN_MAP: dict[str, str] = {
    "periodo": "periodo",                              # ej. 20152 -> año 2015, semestre 2
    "cole_cod_mcpio_ubicacion": "cod_mpio_colegio",    # municipio del colegio
    "punt_global": "punt_global",
    "fami_estratovivienda": "estrato_txt",             # "Estrato 1".."Estrato 6" o "Sin Estrato"
    "cole_naturaleza": "naturaleza",                   # OFICIAL / NO OFICIAL
}


def estandarizar_saber11(df: pd.DataFrame, periodo: str = "unificado") -> pd.DataFrame:
    """Devuelve anio, cod_divipola_mpio, punt_global, estrato, colegio_oficial.

    `periodo` solo se usa en el mensaje de error para identificar el lote.
    """
    faltantes = [c for c in COLUMN_MAP if c not in df.columns]
    if faltantes:
        raise KeyError(f"Saber 11 (periodo {periodo}): faltan las columnas {faltantes}")
    d = df[list(COLUMN_MAP)].rename(columns=COLUMN_MAP)
    # "Estrato 3" -> 3; "Sin Estrato" y otros textos -> nulo
    estrato = d["estrato_txt"].str.extract(r"^Estrato\s+([1-6])$")[0]
    return pd.DataFrame(
        {
            "anio": d["periodo"].astype(int) // 10,
            "cod_divipola_mpio": d["cod_mpio_colegio"].map(normalizar_codigo),
            "punt_global": pd.to_numeric(d["punt_global"], errors="coerce").astype("float64"),
            "estrato": pd.to_numeric(estrato).astype("Int64"),
            "colegio_oficial": d["naturaleza"].str.upper().eq("OFICIAL"),
        }
    ).reset_index(drop=True)


def leer_saber11(path: str | Path, chunksize: int = 500_000) -> pd.DataFrame:
    """Lee el CSV crudo por chunks, estandarizando cada uno (archivos de millones de filas)."""
    partes = [
        estandarizar_saber11(chunk)
        for chunk in pd.read_csv(path, dtype=str, chunksize=chunksize)
    ]
    return pd.concat(partes, ignore_index=True)
