"""Transformación de Saber 11 (dataset unificado `kgxf-xxbe` de datos.gov.co).

El dataset trae todos los periodos con un mismo esquema, por lo que el mapeo de
columnas es único (la guía anticipaba un mapeo por periodo; ver ledger).
"""
from pathlib import Path

import pandas as pd

from src.homologation.divipola import normalizar_codigo

# Columna original -> nombre interno. Si falta alguna, se falla con error claro.
COLUMN_MAP: dict[str, str] = {
    "estu_consecutivo": "estu_consecutivo",           # id del estudiante: la fuente repite filas (ej. 20194)
    "periodo": "periodo",                              # ej. 20152 -> año 2015, semestre 2
    "cole_cod_mcpio_ubicacion": "cod_mpio_colegio",    # municipio del colegio
    "punt_global": "punt_global",
    "fami_estratovivienda": "estrato_txt",             # "Estrato 1".."Estrato 6" o "Sin Estrato"
    "cole_naturaleza": "naturaleza",                   # OFICIAL / NO OFICIAL
}


def _estandarizar(df: pd.DataFrame, periodo: str) -> pd.DataFrame:
    """Como `estandarizar_saber11` pero conserva `estu_consecutivo` (para deduplicar entre chunks)."""
    faltantes = [c for c in COLUMN_MAP if c not in df.columns]
    if faltantes:
        raise KeyError(f"Saber 11 (periodo {periodo}): faltan las columnas {faltantes}")
    d = df[list(COLUMN_MAP)].rename(columns=COLUMN_MAP)
    # "Estrato 3" -> 3; "Sin Estrato" y otros textos -> nulo
    estrato = d["estrato_txt"].str.extract(r"^Estrato\s+([1-6])$")[0]
    return pd.DataFrame(
        {
            "estu_consecutivo": d["estu_consecutivo"],
            "anio": d["periodo"].astype(int) // 10,
            "cod_divipola_mpio": d["cod_mpio_colegio"].map(normalizar_codigo),
            "punt_global": pd.to_numeric(d["punt_global"], errors="coerce").astype("float64"),
            "estrato": pd.to_numeric(estrato).astype("Int64"),
            "colegio_oficial": d["naturaleza"].str.upper().eq("OFICIAL"),
        }
    ).reset_index(drop=True)


def estandarizar_saber11(df: pd.DataFrame, periodo: str = "unificado") -> pd.DataFrame:
    """Devuelve anio, cod_divipola_mpio, punt_global, estrato, colegio_oficial.

    `periodo` solo se usa en el mensaje de error para identificar el lote.
    """
    return _estandarizar(df, periodo).drop(columns="estu_consecutivo")


def leer_saber11(path: str | Path, chunksize: int = 500_000, notas: list[str] | None = None) -> pd.DataFrame:
    """Lee el CSV crudo por chunks (millones de filas) y elimina estudiantes repetidos.

    El dataset unificado repite filas (el periodo 20194 trae cada estudiante dos veces); se
    deduplica por `estu_consecutivo` sobre el total, porque las copias pueden caer en chunks distintos.
    """
    partes = [_estandarizar(chunk, "unificado") for chunk in pd.read_csv(path, dtype=str, chunksize=chunksize)]
    todo = pd.concat(partes, ignore_index=True)
    unico = todo.drop_duplicates("estu_consecutivo", ignore_index=True)
    if notas is not None and len(unico) < len(todo):
        notas.append(f"Saber 11: {len(todo) - len(unico)} fila(s) de estudiantes repetidos (mismo estu_consecutivo) eliminada(s)")
    return unico.drop(columns="estu_consecutivo")
