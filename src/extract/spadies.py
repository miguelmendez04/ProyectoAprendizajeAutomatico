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

    # Solo las columnas canónicas obligatorias deben venir en el export; las demás
    # (p. ej. código de programa) del mapeo se usan si están presentes.
    faltantes = [
        orig for orig, canon in cfg_s["columnas"].items()
        if orig not in crudo.columns and canon in COLUMNAS_OBLIGATORIAS
    ]
    if faltantes:
        raise KeyError(f"SPADIES ({path.name}): faltan las columnas {faltantes}")
    presentes = {orig: canon for orig, canon in cfg_s["columnas"].items() if orig in crudo.columns}
    df = crudo[list(presentes)].rename(columns=presentes)

    df["cod_ies"] = df["cod_ies"].str.strip().str.replace(r"\.0+$", "", regex=True)
    df["anio"] = df["anio"].astype(float).astype(int)
    # El export puede traer coma decimal ("12,5")
    tasa = pd.to_numeric(df["tasa_desercion"].str.replace(",", ".", regex=False), errors="coerce")

    tope = 100 if cfg_s.get("tasa_en_porcentaje", False) else 1
    maximo = tasa.max()
    # La bandera debe ser coherente con los datos: si no, el objetivo quedaría mal escalado sin avisar
    if tope == 1 and 1 < maximo <= 100 and tasa.min() >= 0:
        raise ValueError("SPADIES: tasa_en_porcentaje=false pero hay tasas mayores que 1; ¿vienen en porcentaje?")
    if tasa.isna().any() or ((tasa < 0) | (tasa > tope)).any():
        raise ValueError(f"SPADIES: tasa_desercion fuera de rango [0, {tope}] o no numérica")
    if tope == 100 and 0 < maximo <= 1:
        raise ValueError("SPADIES: tasa_en_porcentaje=true pero todas las tasas están entre 0 y 1; ¿vienen como proporción?")
    df["tasa_desercion"] = tasa / tope

    llave = [c for c in ("cod_ies", "cod_snies_programa", "cod_divipola_mpio", "anio") if c in df.columns]
    if df.duplicated(llave).any():
        raise ValueError(f"SPADIES: llaves duplicadas en {llave}")

    write_metadata(path, fuente="SPADIES (export manual)", url=str(path), filas=len(df))
    return df.reset_index(drop=True)
