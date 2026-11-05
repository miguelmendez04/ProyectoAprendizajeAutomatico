"""Lector de los archivos exportados a mano desde la consulta pública de SPADIES.

SPADIES no ofrece descarga automática: el export se guarda en `data/raw/spadies/`
y este módulo solo lo lee, normaliza y valida.
"""
from pathlib import Path

import pandas as pd

from src.extract.common import write_metadata

COLUMNAS_OBLIGATORIAS = ["cod_ies", "anio", "tasa_desercion"]


def leer_spadies(path: Path, cfg: dict) -> pd.DataFrame:
    """Devuelve cod_ies (str), anio (int) y tasa_desercion como proporción 0-1.

    Las columnas del export se renombran según `cfg["spadies"]["columnas"]`.
    Lanza KeyError si falta una columna y ValueError si hay llaves duplicadas
    o tasas fuera de rango.
    """
    path = Path(path)
    cfg_s = cfg["spadies"]
    if path.suffix.lower() in {".xlsx", ".xls"}:
        crudo = pd.read_excel(path, dtype=str)
    else:
        crudo = pd.read_csv(path, dtype=str)

    faltantes = [c for c in cfg_s["columnas"] if c not in crudo.columns]
    if faltantes:
        raise KeyError(f"SPADIES ({path.name}): faltan las columnas {faltantes}")
    df = crudo[list(cfg_s["columnas"])].rename(columns=cfg_s["columnas"])

    df["cod_ies"] = df["cod_ies"].str.strip().str.replace(r"\.0+$", "", regex=True)
    df["anio"] = df["anio"].astype(float).astype(int)
    # El export puede traer coma decimal ("12,5")
    tasa = pd.to_numeric(df["tasa_desercion"].str.replace(",", ".", regex=False), errors="coerce")

    tope = 100 if cfg_s.get("tasa_en_porcentaje", False) else 1
    if tasa.isna().any() or ((tasa < 0) | (tasa > tope)).any():
        raise ValueError(f"SPADIES: tasa_desercion fuera de rango [0, {tope}] o no numérica")
    df["tasa_desercion"] = tasa / tope

    llave = [c for c in ("cod_ies", "cod_snies_programa", "cod_divipola_mpio", "anio") if c in df.columns]
    if df.duplicated(llave).any():
        raise ValueError(f"SPADIES: llaves duplicadas en {llave}")

    write_metadata(path, fuente="SPADIES (export manual)", url=str(path), filas=len(df))
    return df.reset_index(drop=True)
