"""Formulaciones con historia de la IES y «corrección sobre la ingenua», sobre el motor de `experimentos.py`.

Formulaciones (todas con GroupKFold por `cod_ies` y los mismos pliegues que la v1):
- `A_contexto`: las variables de siempre (sin historia).
- `B_historia_contexto`: las variables de historia (`COLUMNAS_HISTORIA`) más todas las de A.
- `B_solo_historia`: solo las variables de historia (aísla cuánto aporta el contexto encima de la historia).
- `C_correccion`: el objetivo es el cambio respecto a la ingenua, `tasa_t - base`, con `base = tasa_t1` o, si no hay, la mediana
  del entrenamiento; el modelo lo predice con historia + contexto y la predicción final es `base + cambio predicho`.
  Sin logit (el cambio puede ser negativo).
- `C_correccion_solo_historia` (adición): como C, con solo las variables de historia, para aislar el aporte del contexto dentro
  de la corrección.

Las predicciones finales se recortan a [0, 1]. Además de las métricas globales se miden por separado las filas con historia
(hay `tasa_t1`) y sin ella (primer año de la IES en el dataset, o sin fila del año anterior): la ingenua solo aplica a las primeras.
"""

import itertools
import logging
import time
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone

from desercion_ies.evaluation.experimentos import (
    BASE_INGENUA,
    BASE_MEDIANA,
    construir_pipeline,
    hash_pliegue,
    metricas,
    parametros_experimentos,
    prediccion_ingenua,
)
from desercion_ies.features.particion import (
    COLUMNAS_HISTORIA,
    COLUMNAS_HISTORIA_REZAGADA,
    cargar_dataset,
    kfold_desde_config,
    preparar_x,
    variables_modelado_desde_config,
)

FORMULACIONES = {
    "A_contexto": {"conjunto": "contexto", "correccion": False},
    "B_historia_contexto": {"conjunto": "historia_contexto", "correccion": False},
    "B_solo_historia": {"conjunto": "solo_historia", "correccion": False},
    "C_correccion": {"conjunto": "historia_contexto", "correccion": True},
    "C_correccion_solo_historia": {"conjunto": "solo_historia", "correccion": True},
}
# Variantes «historia rezagada» (escenario conservador: la tasa de t-1 aún no estaría publicada al inicio de t): no usan t-1; la
# corrección parte de la ingenua rezagada (tasa t-2, o la mediana del entrenamiento si falta).
FORMULACIONES_REZAGADAS = {
    "B_historia_rezagada_contexto": {"conjunto": "historia_rezagada_contexto", "correccion": False},
    "B_solo_historia_rezagada": {"conjunto": "solo_historia_rezagada", "correccion": False},
    "C_correccion_rezagada": {
        "conjunto": "historia_rezagada_contexto",
        "correccion": True,
        "base": "tasa_t2",
    },
    "C_correccion_solo_historia_rezagada": {
        "conjunto": "solo_historia_rezagada",
        "correccion": True,
        "base": "tasa_t2",
    },
}
TODAS_LAS_FORMULACIONES = {**FORMULACIONES, **FORMULACIONES_REZAGADAS}
COLUMNAS_RESULTADO = [
    "configuracion",
    "tipo",
    "formulacion",
    "modelo",
    "objetivo",
    "imputacion",
    "seleccion",
    "n_variables",
    "pliegue",
    "n_entrenamiento",
    "n_prueba",
    "n_con_historia",
    "n_sin_historia",
    "mae",
    "rmse",
    "r2",
    "mae_con_historia",
    "mae_sin_historia",
    "t_ajuste_s",
    "t_prediccion_s",
    "hash_prueba",
]
_DEFECTO = {
    "incluir": list(FORMULACIONES),
    "objetivo": ["ninguna", "logit"],
    "imputacion": ["mediana"],
    "seleccion": ["ninguna", "info_mutua", "pca"],
    "modelo": ["ridge", "random_forest", "xgboost"],
    "recortar_predicciones": True,
}


log = logging.getLogger(__name__)


class CorreccionSobreIngenua(RegressorMixin, BaseEstimator):
    """Predice el cambio respecto a la ingenua y devuelve `ingenua + cambio`.

    La ingenua de una fila es su columna `tasa_t1` (sin procesar, en `X`) o, si falta, la **mediana de la tasa del
    entrenamiento** (calculada en `fit`, solo con el entrenamiento). El regresor interno se ajusta sobre `y - ingenua`.
    """

    def __init__(self, regressor, columna: str = "tasa_t1"):
        self.regressor = regressor
        self.columna = columna

    def _base(self, X: pd.DataFrame) -> np.ndarray:
        t1 = X[self.columna].to_numpy(dtype=float)
        return np.where(np.isnan(t1), self.mediana_, t1)

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y, dtype=float)
        self.mediana_ = float(np.median(y))
        self.regressor_ = clone(self.regressor).fit(X, y - self._base(X))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._base(X) + np.asarray(self.regressor_.predict(X), dtype=float)


def parametros_formulaciones(cfg: dict) -> dict:
    """`modelado.formulaciones` completado con valores por defecto; los hiperparámetros salen de `modelado.experimentos`."""
    p = dict(_DEFECTO)
    p.update(cfg.get("modelado", {}).get("formulaciones") or {})
    malas = [f for f in p["incluir"] if f not in TODAS_LAS_FORMULACIONES]
    if malas:
        raise ValueError(
            f"modelado.formulaciones.incluir: {malas} no existen; las válidas son {list(TODAS_LAS_FORMULACIONES)}"
        )
    return p


def variables_formulacion(
    df: pd.DataFrame, cfg: dict, conjunto: str
) -> tuple[list[str], list[str]]:
    """(numéricas, categóricas) de cada conjunto: `contexto` | `historia_contexto` | `solo_historia` y las rezagadas
    `historia_rezagada_contexto` | `solo_historia_rezagada` (sin t-1)."""
    if conjunto == "contexto":
        return variables_modelado_desde_config(df, cfg)
    requeridas = COLUMNAS_HISTORIA_REZAGADA if "rezagada" in conjunto else COLUMNAS_HISTORIA
    falta = [c for c in requeridas if c not in df.columns]
    if falta:
        raise ValueError(
            f"el dataset no trae las columnas de historia {falta}: regenerarlo con `desercion etl` (incluir_historia: true)"
        )
    if conjunto == "solo_historia":
        return list(COLUMNAS_HISTORIA), []
    if conjunto == "historia_contexto":
        numericas, categoricas = variables_modelado_desde_config(df, cfg)
        return numericas + [c for c in COLUMNAS_HISTORIA if c not in numericas], categoricas
    if conjunto == "solo_historia_rezagada":
        return list(COLUMNAS_HISTORIA_REZAGADA), []
    if conjunto == "historia_rezagada_contexto":
        numericas, categoricas = variables_modelado_desde_config(df, cfg)
        return numericas + [
            c for c in COLUMNAS_HISTORIA_REZAGADA if c not in numericas
        ], categoricas
    raise ValueError(f"conjunto inválido: {conjunto!r}")


def grilla_formulacion(nombre: str, p: dict) -> list[dict]:
    """Configuraciones de una formulación; en la corrección el objetivo es el cambio, así que no hay logit."""
    objetivos = ["ninguna"] if TODAS_LAS_FORMULACIONES[nombre]["correccion"] else p["objetivo"]
    return [
        {"formulacion": nombre, "objetivo": o, "imputacion": i, "seleccion": s, "modelo": m}
        for o, i, s, m in itertools.product(objetivos, p["imputacion"], p["seleccion"], p["modelo"])
    ]


def nombre_configuracion(c: dict) -> str:
    return f"{c['formulacion']} | {c['modelo']} | {c['objetivo']} | {c['imputacion']} | {c['seleccion']}"


def construir_pipeline_formulacion(
    c: dict, numericas: list[str], categoricas: list[str], p_exp: dict
):
    """Pipeline de la configuración; con `correccion` lo envuelve en `CorreccionSobreIngenua`."""
    spec = TODAS_LAS_FORMULACIONES[c["formulacion"]]
    p = dict(p_exp)
    if (
        not categoricas
    ):  # SelectKBest exige k <= nº de columnas: sin categóricas, a lo sumo las numéricas
        p["k_info_mutua"] = min(p["k_info_mutua"], len(numericas))
    pipe = construir_pipeline(c, numericas, categoricas, p)
    return CorreccionSobreIngenua(pipe, spec.get("base", "tasa_t1")) if spec["correccion"] else pipe


def _mae(real, pred) -> float:
    return float(np.mean(np.abs(real - pred))) if len(real) else float("nan")


def _fila(
    nombre, tipo, c, k, df, entrena, prueba, real, pred, con_historia, t_ajuste, t_pred, n_variables
) -> dict:
    return {
        "configuracion": nombre,
        "tipo": tipo,
        "formulacion": c.get("formulacion", "linea_base"),
        "modelo": c.get("modelo", nombre),
        "objetivo": c.get("objetivo", ""),
        "imputacion": c.get("imputacion", ""),
        "seleccion": c.get("seleccion", ""),
        "n_variables": n_variables,
        "pliegue": k,
        "n_entrenamiento": len(entrena),
        "n_prueba": len(prueba),
        "n_con_historia": int(con_historia.sum()),
        "n_sin_historia": int((~con_historia).sum()),
        **metricas(real, pred),
        "mae_con_historia": _mae(real[con_historia], pred[con_historia]),
        "mae_sin_historia": _mae(real[~con_historia], pred[~con_historia]),
        "t_ajuste_s": t_ajuste,
        "t_prediccion_s": t_pred,
        "hash_prueba": hash_pliegue(df, prueba),
    }


def evaluar_lineas_base(
    df: pd.DataFrame, y: pd.Series, pliegues: Sequence[tuple[np.ndarray, np.ndarray]]
) -> list[dict]:
    con = df["tasa_t1"].notna().to_numpy()
    filas = []
    for k, (entrena, prueba) in enumerate(pliegues):
        real = y.iloc[prueba].to_numpy()
        mediana = np.full(len(prueba), float(np.median(y.iloc[entrena])))
        ingenua, _, _ = prediccion_ingenua(df, y, entrena, prueba)
        for nombre, pred in ((BASE_MEDIANA, mediana), (BASE_INGENUA, ingenua)):
            filas.append(
                _fila(
                    nombre,
                    "linea_base",
                    {"modelo": nombre.removeprefix("base_")},
                    k,
                    df,
                    entrena,
                    prueba,
                    real,
                    pred,
                    con[prueba],
                    0.0,
                    0.0,
                    0,
                )
            )
    return filas


def evaluar_configuracion(
    c: dict,
    df: pd.DataFrame,
    y: pd.Series,
    X: pd.DataFrame,
    pliegues,
    numericas: list[str],
    categoricas: list[str],
    p_exp: dict,
    recortar: bool = True,
) -> list[dict]:
    con = df["tasa_t1"].notna().to_numpy()
    filas = []
    for k, (entrena, prueba) in enumerate(pliegues):
        pipe = construir_pipeline_formulacion(c, numericas, categoricas, p_exp)
        t0 = time.perf_counter()
        pipe.fit(X.iloc[entrena], y.iloc[entrena].to_numpy())
        t_ajuste = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred = np.asarray(pipe.predict(X.iloc[prueba]), dtype=float)
        t_pred = time.perf_counter() - t0
        if recortar:
            pred = np.clip(pred, 0.0, 1.0)
        filas.append(
            _fila(
                nombre_configuracion(c),
                "modelo",
                c,
                k,
                df,
                entrena,
                prueba,
                y.iloc[prueba].to_numpy(),
                pred,
                con[prueba],
                t_ajuste,
                t_pred,
                len(numericas) + len(categoricas),
            )
        )
    return filas


def ejecutar_formulaciones(
    cfg: dict, df: pd.DataFrame | None = None, verbose: bool = False
) -> pd.DataFrame:
    """Líneas base + todas las configuraciones de las formulaciones elegidas, con los mismos pliegues."""
    p_exp, p = parametros_experimentos(cfg), parametros_formulaciones(cfg)
    if df is None:
        df = cargar_dataset(cfg)
    y = df["tasa_desercion"].astype(float)
    pliegues = list(kfold_desde_config(df, cfg))  # una sola vez
    filas = evaluar_lineas_base(df, y, pliegues)
    for nombre in p["incluir"]:
        numericas, categoricas = variables_formulacion(
            df, cfg, TODAS_LAS_FORMULACIONES[nombre]["conjunto"]
        )
        X = preparar_x(df, numericas, categoricas)
        for c in grilla_formulacion(nombre, p):
            t0 = time.perf_counter()
            filas += evaluar_configuracion(
                c, df, y, X, pliegues, numericas, categoricas, p_exp, p["recortar_predicciones"]
            )
            if verbose:
                log.info(f"{nombre_configuracion(c)}: {time.perf_counter() - t0:.1f} s")
    return pd.DataFrame(filas, columns=COLUMNAS_RESULTADO)


def resumen(resultados: pd.DataFrame) -> pd.DataFrame:
    """Media y desviación entre pliegues por configuración, ordenado por MAE; mejora % frente a la ingenua en cada subconjunto."""
    g = resultados.groupby("configuracion", sort=False)
    out = g.agg(
        tipo=("tipo", "first"),
        formulacion=("formulacion", "first"),
        modelo=("modelo", "first"),
        objetivo=("objetivo", "first"),
        imputacion=("imputacion", "first"),
        seleccion=("seleccion", "first"),
        n_variables=("n_variables", "first"),
        n_pliegues=("pliegue", "nunique"),
        mae_media=("mae", "mean"),
        mae_desv=("mae", "std"),
        rmse_media=("rmse", "mean"),
        rmse_desv=("rmse", "std"),
        r2_media=("r2", "mean"),
        r2_desv=("r2", "std"),
        mae_con_historia_media=("mae_con_historia", "mean"),
        mae_con_historia_desv=("mae_con_historia", "std"),
        mae_sin_historia_media=("mae_sin_historia", "mean"),
        mae_sin_historia_desv=("mae_sin_historia", "std"),
        t_ajuste_media_s=("t_ajuste_s", "mean"),
    ).reset_index()
    ing = out.set_index("configuracion").loc[BASE_INGENUA]
    for sufijo, col in (
        ("", "mae_media"),
        ("_con_historia", "mae_con_historia_media"),
        ("_sin_historia", "mae_sin_historia_media"),
    ):
        out[f"mejora_vs_ingenua{sufijo}_pct"] = 100 * (ing[col] - out[col]) / ing[col]
    mae = resultados.pivot(index="pliegue", columns="configuracion", values="mae")
    out["pliegues_mejor_que_ingenua"] = out["configuracion"].map(
        lambda c: int((mae[c] < mae[BASE_INGENUA]).sum())
    )
    return out.sort_values("mae_media", kind="stable").reset_index(drop=True)
