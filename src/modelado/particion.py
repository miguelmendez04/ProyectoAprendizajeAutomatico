"""Utilidades de partición del dataset IES x año para la fase de modelado.

Dos esquemas de validación, ambos pensados para no inflar el desempeño:
- `particion_group_kfold`: validación cruzada por IES (`cod_ies`); una institución nunca está a la vez en
  entrenamiento y prueba, porque sus años consecutivos son casi la misma observación.
- `particion_temporal`: entrena con los años iniciales y prueba con los finales (2015–2019 / 2020–2021 por
  defecto). Aquí las mismas IES aparecen en ambos lados: mide la capacidad de extrapolar en el tiempo.

Limitación: el dataset no trae variables de mercado laboral (ninguna serie municipal de TerriData cubre 2015–2021).
"""
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

COLUMNAS_CATEGORICAS = ["sector", "caracter"]  # ids de SNIES: son categorías, no cantidades
_TEXTO = ["cod_ies", "cod_divipola_mpio", *COLUMNAS_CATEGORICAS]


def cargar_dataset(cfg: dict) -> pd.DataFrame:
    """Lee `data/processed/dataset_<unidad>.csv` con los códigos y las categorías como texto."""
    ruta = Path(cfg["rutas"]["processed"]) / f"dataset_{cfg['unidad_analisis']}.csv"
    cabecera = pd.read_csv(ruta, nrows=0).columns
    return pd.read_csv(ruta, dtype={c: str for c in _TEXTO if c in cabecera})


# Variables que no entran como explicativas por defecto:
# - td_poblacion_total_t1: la proyección de población cambia de base (censo 2005 -> 2018) entre los años objetivo 2018 y
#   2019 (-3 % nacional, -10 % en Bogotá) y el nivel absoluto lee ese salto como una caída. Se conservan las variables
#   relativas (% rural, % por edad) hasta tener las retroproyecciones del DANE con base 2018.
# - ratio_spadies_snies: incluye el denominador de la propia tasa (SPADIES t-2); es una variable de diagnóstico.
EXCLUIDAS_POR_DEFECTO = ["td_poblacion_total_t1", "ratio_spadies_snies"]


def columnas_explicativas(df: pd.DataFrame, objetivo: str = "tasa_desercion", excluir: list[str] | None = None) -> list[str]:
    """Variables numéricas que pueden entrar como explicativas (sin llaves, año, objetivo ni `excluir`).

    `excluir=None` aplica `EXCLUIDAS_POR_DEFECTO`; con `[]` no se excluye nada más. Las banderas (booleanas) nunca entran.
    """
    fuera = {objetivo, "anio", "semestre", *(EXCLUIDAS_POR_DEFECTO if excluir is None else excluir)}
    numericas = df.select_dtypes("number").columns
    return [c for c in numericas if c not in fuera and not c.startswith("cod_")]


def columnas_explicativas_desde_config(df: pd.DataFrame, cfg: dict, objetivo: str = "tasa_desercion") -> list[str]:
    """Como `columnas_explicativas`, con la lista `config.yaml: modelado.excluir_variables` (o la de por defecto)."""
    return columnas_explicativas(df, objetivo, cfg.get("modelado", {}).get("excluir_variables"))


def particion_group_kfold(df: pd.DataFrame, n_splits: int = 5, grupo: str = "cod_ies") -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Genera (índices de entrenamiento, índices de prueba) posicionales con GroupKFold por `grupo`."""
    n_grupos = df[grupo].nunique()
    if n_splits > n_grupos:
        raise ValueError(f"n_splits={n_splits} supera el número de IES distintas ({n_grupos})")
    yield from GroupKFold(n_splits=n_splits).split(df, groups=df[grupo])


def particion_temporal(df: pd.DataFrame, ultimo_anio_entrenamiento: int = 2019) -> tuple[np.ndarray, np.ndarray]:
    """(entrenamiento, prueba): años <= `ultimo_anio_entrenamiento` frente a los posteriores (índices posicionales)."""
    entrena = np.flatnonzero((df["anio"] <= ultimo_anio_entrenamiento).to_numpy())
    prueba = np.flatnonzero((df["anio"] > ultimo_anio_entrenamiento).to_numpy())
    if len(entrena) == 0 or len(prueba) == 0:
        raise ValueError(
            f"la partición temporal con corte {ultimo_anio_entrenamiento} deja vacío el conjunto de "
            f"{'entrenamiento' if len(entrena) == 0 else 'prueba'}"
        )
    return entrena, prueba


def particiones_desde_config(df: pd.DataFrame, cfg: dict):
    """(generador GroupKFold, partición temporal) con los parámetros de `config.yaml: particion`."""
    p = cfg.get("particion", {})
    return (
        particion_group_kfold(df, n_splits=p.get("n_splits", 5)),
        particion_temporal(df, p.get("ultimo_anio_entrenamiento", 2019)),
    )
