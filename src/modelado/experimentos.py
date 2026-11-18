"""Motor de experimentos: grilla de configuraciones, validación con los mismos pliegues y líneas base.

Cada configuración es un `Pipeline` de scikit-learn: preprocesador (imputación + estandarización + one-hot) →
[selección o reducción] → modelo, envuelto en `TransformedTargetRegressor` cuando el objetivo va en logit (con la
transformación y la inversa de `objetivo.py`). Todo se ajusta solo con el entrenamiento de cada pliegue; las métricas se
calculan siempre en escala de tasa (0-1), aunque se entrene en logit.

La grilla vive en `config.yaml: modelado.experimentos`. Los pliegues se generan una sola vez (`kfold_desde_config`:
GroupKFold por `cod_ies`) y se reutilizan en todas las configuraciones y líneas base.

Líneas base (mismo formato de resultados):
- `base_mediana`: la mediana de la tasa del entrenamiento del pliegue.
- `base_ingenua`: la tasa del año anterior (t-1) de la misma IES, tomada de la fila t-1 del dataset; si esa fila no
  existe, la mediana del entrenamiento. Las columnas `n_ingenua_t1` y `n_ingenua_mediana` cuentan en cuántas filas de
  prueba aplicó cada caso. La tasa t-1 de la IES es un dato ya observado al predecir t (no usa información futura); como
  GroupKFold deja todas las filas de una IES en el mismo pliegue, esa fila nunca está en el entrenamiento.
"""
import hashlib
import itertools
import time
from collections.abc import Sequence
from functools import partial

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBRegressor

from src.modelado.objetivo import invertir_objetivo, transformar_objetivo
from src.modelado.particion import (
    cargar_dataset, construir_preprocesador, kfold_desde_config, preparar_x, variables_modelado_desde_config,
)

OBJETIVOS = ("ninguna", "logit")
SELECCIONES = ("ninguna", "info_mutua", "pca")
MODELOS = ("ridge", "random_forest", "xgboost")
BASE_MEDIANA = "base_mediana"
BASE_INGENUA = "base_ingenua"
COLUMNAS_RESULTADO = [
    "configuracion", "tipo", "modelo", "objetivo", "imputacion", "seleccion", "pliegue", "n_entrenamiento", "n_prueba",
    "mae", "rmse", "r2", "t_ajuste_s", "t_prediccion_s", "n_ingenua_t1", "n_ingenua_mediana", "hash_prueba",
]
_DEFECTO = {
    "semilla": 42, "n_jobs": 1, "objetivo": list(OBJETIVOS), "imputacion": ["mediana", "knn"], "seleccion": list(SELECCIONES),
    "modelo": list(MODELOS), "k_vecinos": 5, "k_info_mutua": 15, "varianza_pca": 0.90,
    "ridge_alphas": {"exp_min": -3, "exp_max": 3, "n": 13}, "random_forest": {"n_estimators": 300},
    "xgboost": {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 4},
}


def parametros_experimentos(cfg: dict) -> dict:
    """`modelado.experimentos` de la configuración completado con los valores por defecto."""
    p = dict(_DEFECTO)
    p.update(cfg.get("modelado", {}).get("experimentos") or {})
    for k, valores in (("objetivo", OBJETIVOS), ("seleccion", SELECCIONES), ("modelo", MODELOS)):
        malos = [v for v in p[k] if v not in valores]
        if malos:
            raise ValueError(f"modelado.experimentos.{k}: valores inválidos {malos}; los válidos son {list(valores)}")
    return p


def grilla(p: dict) -> list[dict]:
    """Producto cartesiano objetivo × imputación × selección × modelo (el modelo varía más rápido)."""
    return [
        {"objetivo": o, "imputacion": i, "seleccion": s, "modelo": m}
        for o, i, s, m in itertools.product(p["objetivo"], p["imputacion"], p["seleccion"], p["modelo"])
    ]


def nombre_configuracion(c: dict) -> str:
    return f"{c['modelo']} | {c['objetivo']} | {c['imputacion']} | {c['seleccion']}"


# --- logit para TransformedTargetRegressor (funciones de módulo: se pueden serializar) ---
def _logit(y, epsilon: float = 0.001):
    y = np.asarray(y, dtype=float)
    return transformar_objetivo(pd.Series(y.ravel()), "logit", epsilon).to_numpy().reshape(y.shape)


def _inversa_logit(z):
    z = np.asarray(z, dtype=float)
    return invertir_objetivo(pd.Series(z.ravel()), "logit").to_numpy().reshape(z.shape)


def construir_modelo(nombre: str, p: dict):
    if nombre == "ridge":
        a = p["ridge_alphas"]
        return RidgeCV(alphas=np.logspace(a["exp_min"], a["exp_max"], a["n"]))
    if nombre == "random_forest":
        return RandomForestRegressor(random_state=p["semilla"], n_jobs=p["n_jobs"], **p["random_forest"])
    if nombre == "xgboost":
        return XGBRegressor(random_state=p["semilla"], n_jobs=p["n_jobs"], **p["xgboost"])
    raise ValueError(f"modelo inválido: {nombre!r}; debe ser uno de {MODELOS}")


def construir_seleccion(nombre: str, p: dict):
    """Paso de selección o reducción; `None` si es `ninguna`."""
    if nombre == "ninguna":
        return None
    if nombre == "info_mutua":
        return SelectKBest(partial(mutual_info_regression, random_state=p["semilla"]), k=p["k_info_mutua"])
    if nombre == "pca":
        return PCA(n_components=p["varianza_pca"], svd_solver="full", random_state=p["semilla"])
    raise ValueError(f"seleccion inválida: {nombre!r}; debe ser una de {SELECCIONES}")


def construir_pipeline(config: dict, numericas: list[str], categoricas: list[str], p: dict):
    """Pipeline de una configuración: preprocesador → [selección] → modelo (con logit del objetivo si se pide).

    Sin ajustar: todo (imputación, escalado, selección, modelo e inversa del logit) se ajusta en `fit` con las filas de
    entrenamiento que reciba.
    """
    pasos = [("pre", construir_preprocesador(numericas, categoricas, config["imputacion"], p["k_vecinos"]))]
    seleccion = construir_seleccion(config["seleccion"], p)
    if seleccion is not None:
        pasos.append(("sel", seleccion))
    pasos.append(("modelo", construir_modelo(config["modelo"], p)))
    pipe = Pipeline(pasos)
    if config["objetivo"] == "logit":
        return TransformedTargetRegressor(
            regressor=pipe, transformer=FunctionTransformer(_logit, _inversa_logit, check_inverse=False)
        )
    return pipe


# --- datos, pliegues y métricas ---
def preparar_datos(cfg: dict, df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, list[str], list[str]]:
    """(df, X, y, numéricas, categóricas) con las variables de `config.yaml` (excluye `NUNCA_EXPLICATIVAS`)."""
    if df is None:
        df = cargar_dataset(cfg)
    numericas, categoricas = variables_modelado_desde_config(df, cfg)
    return df, preparar_x(df, numericas, categoricas), df["tasa_desercion"].astype(float), numericas, categoricas


def hash_pliegue(df: pd.DataFrame, prueba: np.ndarray) -> str:
    """Huella corta de las IES de prueba del pliegue: dos corridas con los mismos pliegues dan la misma huella."""
    ies = ",".join(sorted(df.iloc[prueba]["cod_ies"].astype(str).unique()))
    return hashlib.sha1(ies.encode()).hexdigest()[:10]


def metricas(y_real, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_real, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_real, y_pred))),
        "r2": float(r2_score(y_real, y_pred)),
    }


def prediccion_ingenua(df: pd.DataFrame, y: pd.Series, entrena: np.ndarray, prueba: np.ndarray) -> tuple[np.ndarray, int, int]:
    """(predicción, filas con t-1, filas con la mediana): la tasa del año anterior de la misma IES o la mediana de entrenamiento.

    El año anterior se busca en todo el dataset (la fila `anio - 1` de la misma `cod_ies`), no solo en el entrenamiento.
    """
    mediana = float(np.median(y.iloc[entrena]))
    previa = pd.Series(y.to_numpy(), index=pd.MultiIndex.from_arrays([df["cod_ies"].astype(str), df["anio"]]))
    previa = previa[~previa.index.duplicated()]
    claves = pd.MultiIndex.from_arrays([df["cod_ies"].astype(str).iloc[prueba], df["anio"].iloc[prueba] - 1])
    valores = previa.reindex(claves).to_numpy()
    hay = ~np.isnan(valores)
    return np.where(hay, valores, mediana), int(hay.sum()), int((~hay).sum())


def _fila(nombre, tipo, config, k, df, entrena, prueba, y_real, y_pred, t_ajuste, t_pred, n_t1=None, n_med=None) -> dict:
    return {
        "configuracion": nombre, "tipo": tipo, "modelo": config.get("modelo", nombre), "objetivo": config.get("objetivo", ""),
        "imputacion": config.get("imputacion", ""), "seleccion": config.get("seleccion", ""), "pliegue": k,
        "n_entrenamiento": len(entrena), "n_prueba": len(prueba), **metricas(y_real, y_pred),
        "t_ajuste_s": t_ajuste, "t_prediccion_s": t_pred, "n_ingenua_t1": n_t1, "n_ingenua_mediana": n_med,
        "hash_prueba": hash_pliegue(df, prueba),
    }


def evaluar_lineas_base(df: pd.DataFrame, y: pd.Series, pliegues: Sequence[tuple[np.ndarray, np.ndarray]]) -> list[dict]:
    filas = []
    for k, (entrena, prueba) in enumerate(pliegues):
        real = y.iloc[prueba].to_numpy()
        t0 = time.perf_counter()
        pred = np.full(len(prueba), float(np.median(y.iloc[entrena])))
        filas.append(_fila(BASE_MEDIANA, "linea_base", {"modelo": "mediana"}, k, df, entrena, prueba, real, pred, 0.0,
                           time.perf_counter() - t0, 0, len(prueba)))
        t0 = time.perf_counter()
        pred, n_t1, n_med = prediccion_ingenua(df, y, entrena, prueba)
        filas.append(_fila(BASE_INGENUA, "linea_base", {"modelo": "ingenua"}, k, df, entrena, prueba, real, pred, 0.0,
                           time.perf_counter() - t0, n_t1, n_med))
    return filas


def evaluar_configuracion(
    config: dict, X: pd.DataFrame, y: pd.Series, df: pd.DataFrame, pliegues: Sequence[tuple[np.ndarray, np.ndarray]],
    numericas: list[str], categoricas: list[str], p: dict,
) -> list[dict]:
    """Una fila por pliegue: ajusta el pipeline con el entrenamiento y mide en la prueba (escala de tasa)."""
    filas = []
    for k, (entrena, prueba) in enumerate(pliegues):
        pipe = construir_pipeline(config, numericas, categoricas, p)
        t0 = time.perf_counter()
        pipe.fit(X.iloc[entrena], y.iloc[entrena].to_numpy())
        t_ajuste = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred = np.asarray(pipe.predict(X.iloc[prueba]), dtype=float)
        t_pred = time.perf_counter() - t0
        filas.append(_fila(nombre_configuracion(config), "modelo", config, k, df, entrena, prueba,
                           y.iloc[prueba].to_numpy(), pred, t_ajuste, t_pred))
    return filas


def ejecutar_experimentos(cfg: dict, df: pd.DataFrame | None = None, verbose: bool = False) -> pd.DataFrame:
    """Corre las líneas base y toda la grilla con los mismos pliegues; devuelve una fila por configuración y pliegue."""
    p = parametros_experimentos(cfg)
    df, X, y, numericas, categoricas = preparar_datos(cfg, df)
    pliegues = list(kfold_desde_config(df, cfg))  # una sola vez: los mismos pliegues para todas las configuraciones
    filas = evaluar_lineas_base(df, y, pliegues)
    configs = grilla(p)
    for i, c in enumerate(configs, 1):
        t0 = time.perf_counter()
        filas += evaluar_configuracion(c, X, y, df, pliegues, numericas, categoricas, p)
        if verbose:
            print(f"[{i}/{len(configs)}] {nombre_configuracion(c)}: {time.perf_counter() - t0:.1f} s", flush=True)
    return pd.DataFrame(filas, columns=COLUMNAS_RESULTADO)


def resumen(resultados: pd.DataFrame) -> pd.DataFrame:
    """Media y desviación entre pliegues por configuración, ordenado por MAE medio (con la mejora frente a la ingenua)."""
    g = resultados.groupby("configuracion", sort=False)
    out = g.agg(
        tipo=("tipo", "first"), modelo=("modelo", "first"), objetivo=("objetivo", "first"), imputacion=("imputacion", "first"),
        seleccion=("seleccion", "first"), n_pliegues=("pliegue", "nunique"),
        mae_media=("mae", "mean"), mae_desv=("mae", "std"), rmse_media=("rmse", "mean"), rmse_desv=("rmse", "std"),
        r2_media=("r2", "mean"), r2_desv=("r2", "std"), t_ajuste_media_s=("t_ajuste_s", "mean"),
        t_prediccion_media_s=("t_prediccion_s", "mean"),
    ).reset_index()
    ingenua = out.loc[out["configuracion"] == BASE_INGENUA, "mae_media"].iloc[0]
    out["mejora_vs_ingenua_pct"] = 100 * (ingenua - out["mae_media"]) / ingenua
    # pliegues en que la configuración tiene menor MAE que la ingenua (comparación pareada por pliegue)
    mae = resultados.pivot(index="pliegue", columns="configuracion", values="mae")
    out["pliegues_mejor_que_ingenua"] = out["configuracion"].map(lambda c: int((mae[c] < mae[BASE_INGENUA]).sum()))
    return out.sort_values("mae_media", kind="stable").reset_index(drop=True)
