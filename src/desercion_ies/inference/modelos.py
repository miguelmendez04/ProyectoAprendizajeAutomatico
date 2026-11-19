"""Modelos finales listos para predecir: objeto guardado, validación de la entrada y predicción.

Un `ModeloEntrenado` envuelve el estimador del pipeline de evaluación (el mismo que construyen `ajuste.preparar` y
`evaluacion_final.construir_final`) más lo necesario para usarlo sobre filas nuevas: las columnas que espera, las categorías
conocidas y la validación de la entrada. La predicción pasa por el mismo camino que la evaluación: `preparar_x` → estimador → recorte a [0, 1].

Los modelos se guardan con joblib (pickle): **no cargue archivos de modelo de origen desconocido**. `predecir` comprueba el SHA-256 del archivo
contra el JSON de metadatos antes de cargarlo.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from desercion_ies.features.particion import preparar_x

log = logging.getLogger(__name__)

TASA = "tasa_t1"
COLUMNAS_CODIGO = {"cod_ies": str, "cod_divipola_mpio": str, "sector": str, "caracter": str}
IDENTIFICADORES = ("cod_ies", "anio", "semestre", "cod_divipola_mpio")
MAX_LISTADAS = 8


class EntradaInvalida(ValueError):
    """La entrada de `desercion predecir` no sirve; el mensaje dice qué falta y cómo corregirlo."""


class MedianaCaracterSector(RegressorMixin, BaseEstimator):
    """Línea base de la Tarea 2: mediana de la tasa por carácter × sector; sin ese grupo, la mediana global.

    Es la misma regla que `evaluation.tarea2.prediccion_mediana_grupo` (claves `caracter|sector` con los códigos del dataset).
    """

    def __init__(self, columnas: tuple[str, str] = ("caracter", "sector")):
        self.columnas = columnas

    def _clave(self, df: pd.DataFrame) -> pd.Series:
        return df[list(self.columnas)].astype(str).agg("|".join, axis=1)

    def fit(self, df: pd.DataFrame, y) -> MedianaCaracterSector:
        y = pd.Series(np.asarray(y, dtype=float), index=df.index)
        self.medianas_ = y.groupby(self._clave(df)).median()
        self.mediana_global_ = float(np.median(y))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        pred = self._clave(df).map(self.medianas_)
        return pred.fillna(self.mediana_global_).to_numpy(dtype=float)


@dataclass
class ModeloEntrenado:
    """Estimador final + contrato de entrada."""

    clave: str  # tarea1_ridge | tarea2_ridge | tarea2_mediana
    nombre: str
    tarea: int
    estimador: object
    numericas: list[str]
    categoricas: list[str]
    obligatorias: list[str]  # columnas que la entrada debe traer
    opcionales: list[str] = field(default_factory=list)  # si faltan, las imputa el propio pipeline
    categorias: dict[str, list[str]] = field(
        default_factory=dict
    )  # códigos vistos en el entrenamiento
    usa_preparar_x: bool = True  # la línea base de la Tarea 2 trabaja sobre las columnas crudas

    # ---------- entrada ----------
    def validar(self, df: pd.DataFrame) -> pd.DataFrame:
        """Devuelve la entrada lista (columnas crudas verificadas) o lanza `EntradaInvalida` con un mensaje claro."""
        if df.empty:
            raise EntradaInvalida("el archivo de entrada no tiene filas")
        faltan = [c for c in self.obligatorias if c not in df.columns]
        if faltan:
            raise EntradaInvalida(self._mensaje_faltan(faltan))
        df = df.copy()
        for c in self.numericas:
            if c in df.columns:
                df[c] = self._a_numerica(df, c)
        for c in self.categoricas or []:
            if df[c].isna().any() or (df[c].astype(str).str.strip() == "").any():
                raise EntradaInvalida(f"la columna `{c}` tiene valores vacíos")
        if self.tarea == 1:
            self._validar_tasa(df)
        for c, conocidas in self.categorias.items():
            nuevas = sorted(set(df[c].astype(str)) - set(conocidas))
            if nuevas:
                log.warning(
                    "`%s` tiene códigos que el modelo no vio al entrenar (%s): se predice con el grupo"
                    " de respaldo",
                    c,
                    ", ".join(nuevas[:MAX_LISTADAS]),
                )
        return df

    def _mensaje_faltan(self, faltan: list[str]) -> str:
        lista = ", ".join(f"`{c}`" for c in faltan[:MAX_LISTADAS]) + (
            f" y {len(faltan) - MAX_LISTADAS} más" if len(faltan) > MAX_LISTADAS else ""
        )
        if self.tarea == 1 and TASA in faltan:
            return (
                f"la Tarea 1 exige la columna `{TASA}` (tasa de deserción del año anterior de cada IES) y no está en la entrada. "
                "Si no tiene la historia de las IES, use la Tarea 2 (estimación sin historia): `desercion predecir --tarea 2 ...`"
            )
        return f"faltan columnas obligatorias para la Tarea {self.tarea}: {lista}"

    @staticmethod
    def _a_numerica(df: pd.DataFrame, c: str) -> pd.Series:
        if df[c].dtype == bool:
            return df[c].astype(int)
        convertida = pd.to_numeric(df[c], errors="coerce")
        malos = df.loc[convertida.isna() & df[c].notna() & (df[c].astype(str).str.strip() != ""), c]
        if len(malos):
            ejemplos = ", ".join(repr(v) for v in malos.astype(str).unique()[:MAX_LISTADAS])
            raise EntradaInvalida(
                f"la columna `{c}` debe ser numérica y tiene valores no numéricos: {ejemplos}"
            )
        return convertida

    def _validar_tasa(self, df: pd.DataFrame) -> None:
        tasa = df[TASA]
        if tasa.isna().any():
            n = int(tasa.isna().sum())
            raise EntradaInvalida(
                f"`{TASA}` está vacía en {n} fila(s) (filas {self._filas(tasa.isna())}). La Tarea 1 exige la tasa del año anterior; "
                "para las IES sin historia use la Tarea 2: `desercion predecir --tarea 2 ...`"
            )
        fuera = (tasa < 0) | (tasa > 1)
        if fuera.any():
            raise EntradaInvalida(
                f"`{TASA}` debe ser una proporción entre 0 y 1 (¿está en porcentaje?); fuera de rango en las filas {self._filas(fuera)}"
            )

    @staticmethod
    def _filas(mascara: pd.Series) -> str:
        filas = [
            str(i + 2) for i in np.flatnonzero(mascara.to_numpy())
        ]  # +2: encabezado y base 1 del CSV
        return ", ".join(filas[:MAX_LISTADAS]) + (" …" if len(filas) > MAX_LISTADAS else "")

    # ---------- predicción ----------
    def matriz(self, df: pd.DataFrame) -> pd.DataFrame:
        """La matriz que recibe el estimador (la que arma la evaluación: `preparar_x`); no valida."""
        if not self.usa_preparar_x:
            return df
        x = df.copy()
        if "falta_historia" in self.opcionales and "falta_historia" not in x.columns:
            x["falta_historia"] = 0  # la tasa del año anterior está presente (se validó)
        for c in self.opcionales:
            if c not in x.columns:
                x[c] = (
                    np.nan
                )  # el imputador del pipeline la completa con la mediana del entrenamiento
        return preparar_x(x, self.numericas, self.categoricas)

    def predecir(self, df: pd.DataFrame) -> np.ndarray:
        """Predicción de la tasa de deserción de cada fila, recortada a [0, 1]."""
        df = self.validar(df)
        return np.clip(np.asarray(self.estimador.predict(self.matriz(df)), dtype=float), 0, 1)
