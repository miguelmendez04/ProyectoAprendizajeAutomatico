"""Lector de los exports manuales de la consulta pública de SPADIES.

Formato real: CSV con `;`, columna `TIPO` y una columna por semestre (`1998-1` ... `2024-2`),
con las filas `DESERTORES` y `MATRICULADOS`. Hay un archivo por IES, `spadies_<codigo>_<nombre>.csv`
(el código SNIES sale del nombre), y uno nacional, `spadies_nacional.csv`.

Tasa de deserción (definición de SPADIES): desertores del periodo t / matriculados del periodo t-2,
con t-2 = dos semestres antes (mismo semestre del año anterior). La anual es el promedio de los dos
semestres del año.
"""
import re
from pathlib import Path

import pandas as pd

from src.extract.common import write_metadata

PATRON_ARCHIVO = re.compile(r"^spadies_(\d+)_.+\.csv$", re.IGNORECASE)
PERIODICIDADES = {"anual", "semestre"}
REZAGO_PERIODOS = 2  # semestres entre los desertores y la matrícula del denominador
_PERIODO = re.compile(r"^(\d{4})-([12])$")


def _leer_serie(path: Path) -> pd.DataFrame:
    """Lee un export (una serie) y lo pasa a formato largo: periodo, desertores, matriculados."""
    crudo = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8-sig")
    if "TIPO" not in crudo.columns:
        raise KeyError(f"SPADIES ({path.name}): falta la columna TIPO")
    filas = crudo.set_index(crudo["TIPO"].str.strip().str.upper()).drop(columns="TIPO")
    for fila in ("DESERTORES", "MATRICULADOS"):
        if fila not in filas.index:
            raise KeyError(f"SPADIES ({path.name}): falta la fila {fila}")
    periodos = [c.strip() for c in filas.columns]
    malos = [p for p in periodos if not _PERIODO.match(p)]
    if malos:
        raise ValueError(f"SPADIES ({path.name}): encabezados de periodo inválidos {malos[:3]}")
    valores = {}
    for fila in ("DESERTORES", "MATRICULADOS"):
        num = pd.to_numeric(filas.loc[fila].str.strip(), errors="coerce")
        # un vacío es dato faltante; un texto en la celda es un error del export
        if (num.isna() & filas.loc[fila].notna() & (filas.loc[fila].str.strip() != "")).any():
            raise ValueError(f"SPADIES ({path.name}): valores no numéricos en la fila {fila}")
        valores[fila.lower()] = num.to_numpy()
    largo = pd.DataFrame({"periodo": periodos, **valores})
    # SPADIES reporta 0 desertores antes de tener datos (1998-1, 1998-2): es dato faltante, no un cero real
    primero = largo["desertores"].gt(0).idxmax() if largo["desertores"].gt(0).any() else len(largo)
    largo.loc[largo.index < primero, "desertores"] = float("nan")
    return largo


def leer_nacional(path: Path) -> pd.DataFrame:
    """Serie nacional (contexto y control de las sumas de IES): periodo, desertores, matriculados."""
    return _leer_serie(Path(path))


def _archivos_ies(directorio: Path) -> list[Path]:
    archivos = sorted(p for p in directorio.glob("*.csv") if PATRON_ARCHIVO.match(p.name))
    if not archivos:
        raise FileNotFoundError(f"SPADIES: no hay archivos spadies_<codigo>_<nombre>.csv en {directorio}")
    return archivos


def leer_spadies_ies(directorio: Path) -> pd.DataFrame:
    """Une los exports `spadies_<codigo>_<nombre>.csv` del directorio: ies, periodo, desertores, matriculados."""
    archivos = _archivos_ies(Path(directorio))
    tablas = [_leer_serie(p).assign(ies=PATRON_ARCHIVO.match(p.name).group(1)) for p in archivos]
    return pd.concat(tablas, ignore_index=True)[["ies", "periodo", "desertores", "matriculados"]]


def tasas_semestrales(largo: pd.DataFrame) -> pd.DataFrame:
    """Agrega anio, semestre y tasa_desercion = desertores_t / matriculados_(t-2) a cada fila del largo."""
    out = largo.copy()
    partes = out["periodo"].str.extract(_PERIODO)
    out["anio"], out["semestre"] = partes[0].astype(int), partes[1].astype(int)
    out["_idx"] = out["anio"] * 2 + out["semestre"] - 1
    denominador = out[["ies", "_idx", "matriculados"]].assign(_idx=lambda d: d["_idx"] + REZAGO_PERIODOS)
    denominador = denominador.rename(columns={"matriculados": "_mat_t2"})
    out = out.merge(denominador, on=["ies", "_idx"], how="left")
    out["tasa_desercion"] = out["desertores"] / out["_mat_t2"].where(out["_mat_t2"] > 0)
    return out.drop(columns=["_idx", "_mat_t2"])


def tasa_anual(semestrales: pd.DataFrame) -> pd.DataFrame:
    """Promedio de las tasas de los dos semestres de cada año; sin los dos semestres no hay tasa anual."""
    validas = semestrales.dropna(subset=["tasa_desercion"])
    g = validas.groupby(["ies", "anio"], as_index=False).agg(
        tasa_desercion=("tasa_desercion", "mean"), n=("semestre", "nunique")
    )
    g = g[g["n"] == 2].drop(columns="n").rename(columns={"ies": "cod_ies"})
    return g[["cod_ies", "anio", "tasa_desercion"]].reset_index(drop=True)


def validar_contra_nacional(largo_ies: pd.DataFrame, nacional: pd.DataFrame) -> list[str]:
    """Notas de los periodos en que la suma de las IES supera el total nacional (error del export)."""
    suma = largo_ies.groupby("periodo")[["desertores", "matriculados"]].sum(min_count=1)
    total = nacional.set_index("periodo")[["desertores", "matriculados"]]
    notas = []
    for periodo, fila in suma.iterrows():
        if periodo not in total.index:
            continue
        for col in ("desertores", "matriculados"):
            if fila[col] > total.loc[periodo, col]:
                notas.append(
                    f"SPADIES: en {periodo} la suma de {col} de las IES ({fila[col]:.0f}) "
                    f"supera el total nacional ({total.loc[periodo, col]:.0f})"
                )
    return notas


def leer_spadies(cfg: dict, notas: list[str]) -> pd.DataFrame:
    """Objetivo para el ETL: cod_ies, anio, [semestre], tasa_desercion (0-1), según `spadies.periodicidad`.

    Registra en `notas` la cobertura de IES y las inconsistencias contra el total nacional.
    """
    cfg_s = cfg["spadies"]
    periodicidad = cfg_s.get("periodicidad", "anual")
    if periodicidad not in PERIODICIDADES:
        raise ValueError(f"spadies.periodicidad inválida: {periodicidad!r}; debe ser una de {sorted(PERIODICIDADES)}")
    directorio = Path(cfg_s["directorio"])
    largo = leer_spadies_ies(directorio)
    notas.append(
        f"SPADIES: {largo['ies'].nunique()} IES leídas de {directorio} "
        f"(periodos {largo['periodo'].min()} a {largo['periodo'].max()})"
    )
    nacional = Path(cfg_s["nacional"])
    if nacional.exists():
        notas.extend(validar_contra_nacional(largo, leer_nacional(nacional)))
    else:
        notas.append(f"SPADIES: no existe {nacional}; no se validó la suma de IES contra el total nacional")

    semestrales = tasas_semestrales(largo)
    if periodicidad == "anual":
        out = tasa_anual(semestrales)
    else:
        out = semestrales.dropna(subset=["tasa_desercion"]).rename(columns={"ies": "cod_ies"})
        out = out[["cod_ies", "anio", "semestre", "tasa_desercion"]].reset_index(drop=True)
    for archivo in _archivos_ies(directorio):  # un metadata por export, junto al archivo
        write_metadata(archivo, fuente="SPADIES (export manual)", url=str(archivo), filas=int((largo["ies"] == PATRON_ARCHIVO.match(archivo.name).group(1)).sum()))
    return out
