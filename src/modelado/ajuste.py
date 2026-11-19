"""Ajuste de hiperparámetros con validación anidada (sin fuga).

- **Externa:** GroupKFold por `cod_ies`, 5 pliegues repetidos con las semillas 301–305 (distintas a las del cribado, la confirmación y
  la v3). Cada repetición se genera una sola vez y se usa para las configuraciones por defecto, las ajustadas y las líneas base.
- **Interna:** dentro de cada pliegue externo, `RandomizedSearchCV` de 30 iteraciones con scoring MAE y GroupKFold por IES de 3 pliegues,
  **solo sobre las filas de entrenamiento externo**: la búsqueda no ve el pliegue externo de prueba. El mejor modelo se reajusta con todo
  el entrenamiento externo (`refit`) y se mide en la prueba externa.
- **Espacios** (`ESPACIOS`): ridge `alpha` log-uniforme 1e-3–1e3 (se usa `Ridge`, no `RidgeCV`, para que la búsqueda controle el alfa);
  random forest y xgboost según el enunciado; si el preprocesamiento incluye información mutua, también se ajusta `k`.
- **Tiempos:** de ajuste (reajuste del mejor modelo) y de predicción de cada pliegue, y de búsqueda, para el análisis de complejidad.
- **Paralelismo:** `n_jobs=-1` se aplica sobre (configuración, pliegue externo); cada búsqueda interna corre en un hilo para no
  sobresuscribir los núcleos.

Comparación: por defecto (hiperparámetros de la v1) frente a ajustado en los mismos pliegues externos, y ambos frente a la línea base de
su tarea (Tarea 1: ingenua, tasa de t-1; Tarea 2: mediana por carácter × sector), con IC 95 % y t corregida de Nadeau y Bengio.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import loguniform
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, RandomizedSearchCV

from src.modelado import tarea2 as t2
from src.modelado.experimentos import hash_pliegue, metricas, parametros_experimentos
from src.modelado.formulaciones import TODAS_LAS_FORMULACIONES, construir_pipeline_formulacion, variables_formulacion
from src.modelado.particion import cargar_dataset, preparar_x
from src.modelado.protocolo import comparar_pareado, media_e_intervalo, particiones_repetidas, prediccion_base

ESPACIOS = {
    "ridge": {"alpha": loguniform(1e-3, 1e3)},
    "random_forest": {"n_estimators": [200, 500], "max_depth": [3, 5, 8, None], "min_samples_leaf": [1, 5, 10, 20], "max_features": [0.3, 0.5, "sqrt"]},
    "xgboost": {"n_estimators": [200, 500], "max_depth": [2, 3, 4, 6], "learning_rate": loguniform(0.01, 0.3), "subsample": [0.7, 1.0],
                "colsample_bytree": [0.5, 0.8, 1.0], "min_child_weight": [1, 5, 10], "reg_lambda": loguniform(0.1, 10)},
}
CONTINUOS = ("alpha", "learning_rate", "reg_lambda")  # hiperparámetros log-uniformes: se resumen con mediana y rango
BASE_INGENUA, BASE_MEDIANA, BASE_GRUPO = "base_ingenua", "base_mediana", "base_mediana_grupo"
SEMILLAS_PREVIAS = {11, 101, 102, 103, 104, 105, 201, 202, 203, 204, 205}
COLUMNAS_RESULTADO = [
    "configuracion", "tarea", "variante", "modelo", "repeticion", "semilla", "pliegue", "n_entrenamiento", "n_prueba", "mae", "rmse", "r2",
    "t_ajuste_s", "t_prediccion_s", "t_busqueda_s", "mae_interno", "hiperparametros", "hash_prueba",
]
_DEFECTO = {"semillas_externas": [301, 302, 303, 304, 305], "n_splits_externo": 5, "n_splits_interno": 3, "n_iter": 30, "semilla_busqueda": 42,
            "nivel_confianza": 0.95, "n_jobs": -1, "k_info_mutua": [5, 10, 15, 20, 25, 30]}


def parametros_ajuste(cfg: dict) -> dict:
    p = dict(_DEFECTO)
    p.update(cfg.get("modelado", {}).get("ajuste_temporal") or {})
    if set(p["semillas_externas"]) & SEMILLAS_PREVIAS:
        raise ValueError("las semillas externas deben ser distintas a las del cribado, la confirmación y la v3")
    return p


def configuraciones_ajuste(cfg: dict) -> list[dict]:
    """Tarea 1: las configuraciones fijadas en `config.yaml`. Tarea 2: la mejor configuración de cada modelo en la confirmación de la Tarea 2."""
    p = parametros_ajuste(cfg)
    specs = [{**c, "tarea": "T1"} for c in p["configuraciones_tarea1"]]
    conf = pd.read_csv(p["confirmacion_tarea2"])
    m = conf[conf["tipo"] == "modelo"].sort_values("mae_media", kind="stable")
    for _, f in m.groupby("modelo", sort=False).head(1).iterrows():
        c = {"modelo": f["modelo"], "objetivo": f["objetivo"], "imputacion": f["imputacion"], "reduccion": f["reduccion"],
             "clusters": str(f["clusters"]) == "True"}
        specs.append({"nombre": "T2 | " + t2.nombre_configuracion(c), "tarea": "T2", **c})
    return specs


# ---------- datos y estimadores ----------
def preparar(spec: dict, cfg: dict, df: pd.DataFrame):
    """(X, numéricas, categóricas) de la configuración: contexto en la Tarea 2; el conjunto de su formulación en la Tarea 1."""
    if spec["tarea"] == "T2":
        _, X, _, num, cat = t2.preparar_datos(cfg, df)
        return X, num, cat
    num, cat = variables_formulacion(df, cfg, TODAS_LAS_FORMULACIONES[spec["formulacion"]]["conjunto"])
    return preparar_x(df, num, cat), num, cat


def construir_estimador(spec: dict, numericas: list[str], categoricas: list[str], p_exp: dict, p_t2: dict):
    """Estimador con los hiperparámetros por defecto de la v1."""
    if spec["tarea"] == "T2":
        return t2.construir_pipeline({k: spec[k] for k in ("objetivo", "imputacion", "reduccion", "clusters", "modelo")}, numericas, categoricas, p_exp, p_t2)
    c = {k: spec[k] for k in ("formulacion", "modelo", "objetivo", "imputacion", "seleccion")}
    return construir_pipeline_formulacion(c, numericas, categoricas, p_exp)


def _prefijo(est) -> str:
    """Prefijo de los parámetros del pipeline interno: `TransformedTargetRegressor` y `CorreccionSobreIngenua` lo guardan en `regressor`."""
    return "regressor__" if hasattr(est, "regressor") else ""


def estimador_ajustable(est, spec: dict, k_valores: list[int]):
    """(estimador, distribuciones) para `RandomizedSearchCV`. En ridge se reemplaza `RidgeCV` por `Ridge`; si hay `SelectKBest`, se ajusta `k`."""
    est = clone(est)
    pre = _prefijo(est)
    if spec["modelo"] == "ridge":
        est.set_params(**{pre + "modelo": Ridge(alpha=1.0)})
    dist = {f"{pre}modelo__{k}": v for k, v in ESPACIOS[spec["modelo"]].items()}
    for nombre in est.get_params():
        if nombre.endswith("red__k") or nombre.endswith("sel__k"):
            dist[nombre] = list(k_valores)
    return est, dist


def buscar(est, dist: dict, X: pd.DataFrame, y: np.ndarray, grupos: np.ndarray, p: dict) -> tuple[RandomizedSearchCV, float]:
    """Búsqueda aleatoria interna (GroupKFold por IES de `n_splits_interno` pliegues) **solo con los datos que recibe**."""
    cv = GroupKFold(n_splits=p["n_splits_interno"], shuffle=True, random_state=p["semilla_busqueda"])
    busqueda = RandomizedSearchCV(est, dist, n_iter=p["n_iter"], scoring="neg_mean_absolute_error", cv=cv, random_state=p["semilla_busqueda"],
                                  n_jobs=1, refit=True, error_score="raise")
    t0 = time.perf_counter()
    busqueda.fit(X, y, groups=grupos)
    return busqueda, time.perf_counter() - t0


def _json(params: dict) -> str:
    """Hiperparámetros con el prefijo de pipeline quitado y valores nativos de Python."""
    limpio = {k.split("modelo__")[-1] if "modelo__" in k else k.split("__")[-1]: (v.item() if hasattr(v, "item") else v) for k, v in params.items()}
    return json.dumps(limpio, default=str, sort_keys=True)


def _fila(spec, variante, rep, semilla, k, entrena, prueba, real, pred, hash_p, t_aj, t_pr, t_bu=np.nan, mae_int=np.nan, hip="{}") -> dict:
    return {"configuracion": spec["nombre"], "tarea": spec["tarea"], "variante": variante, "modelo": spec["modelo"], "repeticion": rep, "semilla": int(semilla),
            "pliegue": k, "n_entrenamiento": len(entrena), "n_prueba": len(prueba), **metricas(real, pred), "t_ajuste_s": t_aj, "t_prediccion_s": t_pr,
            "t_busqueda_s": t_bu, "mae_interno": mae_int, "hiperparametros": hip, "hash_prueba": hash_p}


def ajustar_y_evaluar(spec: dict, X: pd.DataFrame, y: pd.Series, grupos: np.ndarray, entrena: np.ndarray, prueba: np.ndarray, numericas: list[str],
                      categoricas: list[str], p_exp: dict, p_t2: dict, p: dict) -> tuple[dict, dict]:
    """Un conjunto de entrenamiento y uno de prueba: (resultado por defecto, resultado ajustado), predicciones recortadas a [0, 1] y tiempos medidos.

    La búsqueda recibe únicamente `X.iloc[entrena]`, `y.iloc[entrena]` y sus IES: no tiene acceso a las filas de `prueba`.
    """
    Xtr, ytr, Xte = X.iloc[entrena], y.iloc[entrena].to_numpy(), X.iloc[prueba]
    por_defecto = construir_estimador(spec, numericas, categoricas, p_exp, p_t2)
    t0 = time.perf_counter()
    por_defecto.fit(Xtr, ytr)
    t_aj = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred_def = np.clip(np.asarray(por_defecto.predict(Xte), dtype=float), 0, 1)
    t_pr = time.perf_counter() - t0
    est, dist = estimador_ajustable(construir_estimador(spec, numericas, categoricas, p_exp, p_t2), spec, p["k_info_mutua"])
    busqueda, t_bu = buscar(est, dist, Xtr, ytr, grupos[entrena], p)
    t0 = time.perf_counter()
    pred_aj = np.clip(np.asarray(busqueda.predict(Xte), dtype=float), 0, 1)
    t_pr_aj = time.perf_counter() - t0
    real = y.iloc[prueba].to_numpy()
    return ({"pred": pred_def, "t_ajuste_s": t_aj, "t_prediccion_s": t_pr},
            {"pred": pred_aj, "t_ajuste_s": float(busqueda.refit_time_), "t_prediccion_s": t_pr_aj, "t_busqueda_s": t_bu,
             "mae_interno": float(-busqueda.best_score_), "hiperparametros": _json(busqueda.best_params_), "real": real})


def _tarea(spec, rep, semilla, k, entrena, prueba, X, y, grupos, numericas, categoricas, p_exp, p_t2, p, hash_p) -> list[dict]:
    d, a = ajustar_y_evaluar(spec, X, y, grupos, entrena, prueba, numericas, categoricas, p_exp, p_t2, p)
    real = y.iloc[prueba].to_numpy()
    return [_fila(spec, "defecto", rep, semilla, k, entrena, prueba, real, d["pred"], hash_p, d["t_ajuste_s"], d["t_prediccion_s"]),
            _fila(spec, "ajustado", rep, semilla, k, entrena, prueba, real, a["pred"], hash_p, a["t_ajuste_s"], a["t_prediccion_s"], a["t_busqueda_s"],
                  a["mae_interno"], a["hiperparametros"])]


def lineas_base(df: pd.DataFrame, y: pd.Series, repeticiones, semillas) -> list[dict]:
    """Líneas base por tarea y pliegue externo: ingenua (t-1) y mediana en la Tarea 1; mediana por carácter × sector y mediana en la Tarea 2."""
    filas = []
    for rep, (pliegues, semilla) in enumerate(zip(repeticiones, semillas)):
        for k, (entrena, prueba) in enumerate(pliegues):
            real, h = y.iloc[prueba].to_numpy(), hash_pliegue(df, prueba)
            med = np.full(len(prueba), float(np.median(y.iloc[entrena])))
            grupo, _ = t2.prediccion_mediana_grupo(df, y, entrena, prueba)
            for tarea, nombre, pred in (("T1", BASE_INGENUA, prediccion_base(df, y, entrena, prueba, "tasa_t1")), ("T1", BASE_MEDIANA, med),
                                        ("T2", BASE_GRUPO, grupo), ("T2", BASE_MEDIANA, med)):
                spec = {"nombre": f"{tarea} {nombre}", "tarea": tarea, "modelo": nombre.removeprefix("base_")}
                filas.append(_fila(spec, "linea_base", rep, semilla, k, entrena, prueba, real, pred, h, 0.0, 0.0))
    return filas


def ejecutar_ajuste(cfg: dict, df: pd.DataFrame | None = None, specs: list[dict] | None = None, verbose: bool = False,
                    parcial: str | Path | None = None) -> pd.DataFrame:
    """Validación anidada completa: por defecto, ajustado y líneas base en los mismos pliegues externos.

    Con `parcial` (un directorio) cada configuración se guarda al terminar (`parcial_<n>.csv`) y, si el archivo ya existe, se reutiliza: sirve para
    reanudar una corrida interrumpida sin repetir lo hecho. Los resultados son los mismos con o sin puntos de control."""
    p, p_exp, p_t2 = parametros_ajuste(cfg), parametros_experimentos(cfg), t2.parametros_tarea2(cfg)
    df = cargar_dataset(cfg) if df is None else df
    y = df["tasa_desercion"].astype(float)
    specs = configuraciones_ajuste(cfg) if specs is None else specs
    repeticiones = particiones_repetidas(df, p["n_splits_externo"], p["semillas_externas"])  # una sola vez
    grupos = df["cod_ies"].astype(str).to_numpy()
    filas = lineas_base(df, y, repeticiones, p["semillas_externas"])
    datos = {s["nombre"]: preparar(s, cfg, df) for s in specs}
    tareas = [(s, rep, int(semilla), k, entrena, prueba, hash_pliegue(df, prueba))
              for s in specs for rep, (pliegues, semilla) in enumerate(zip(repeticiones, p["semillas_externas"])) for k, (entrena, prueba) in enumerate(pliegues)]
    if parcial is None:
        salida = Parallel(n_jobs=p["n_jobs"], batch_size=1, verbose=10 if verbose else 0)(
            delayed(_tarea)(s, rep, semilla, k, entrena, prueba, datos[s["nombre"]][0], y, grupos, datos[s["nombre"]][1], datos[s["nombre"]][2], p_exp, p_t2, p, h)
            for s, rep, semilla, k, entrena, prueba, h in tareas)
        return pd.DataFrame(filas + [f for par in salida for f in par], columns=COLUMNAS_RESULTADO)
    carpeta = Path(parcial)
    carpeta.mkdir(parents=True, exist_ok=True)
    partes = [pd.DataFrame(filas, columns=COLUMNAS_RESULTADO)]
    for n, s in enumerate(specs):
        archivo = carpeta / f"parcial_{n}.csv"
        if archivo.exists():
            partes.append(pd.read_csv(archivo))
            continue
        t0 = time.perf_counter()
        salida = Parallel(n_jobs=p["n_jobs"], batch_size=1)(
            delayed(_tarea)(sp, rep, semilla, k, entrena, prueba, datos[sp["nombre"]][0], y, grupos, datos[sp["nombre"]][1], datos[sp["nombre"]][2], p_exp, p_t2, p, h)
            for sp, rep, semilla, k, entrena, prueba, h in tareas if sp["nombre"] == s["nombre"])
        parte = pd.DataFrame([f for par in salida for f in par], columns=COLUMNAS_RESULTADO)
        parte.to_csv(archivo, index=False)
        partes.append(parte)
        if verbose:
            print(f"[{n + 1}/{len(specs)}] {s['nombre']}: {time.perf_counter() - t0:.0f} s", flush=True)
    return pd.concat(partes, ignore_index=True)


# ---------- resúmenes ----------
def _clave(f) -> str:
    return f["configuracion"] if f["variante"] == "linea_base" else f"{f['configuracion']} [{f['variante']}]"


def matriz(res: pd.DataFrame, columna: str = "mae") -> pd.DataFrame:
    r = res.assign(clave=res.apply(_clave, axis=1))
    return r.pivot(index=["repeticion", "pliegue"], columns="clave", values=columna)


def resumen_ajuste(res: pd.DataFrame, nivel: float = 0.95) -> pd.DataFrame:
    """Por configuración y variante: MAE medio con IC 95 % (Nadeau y Bengio), comparación pareada con la línea base de su tarea y, para las
    ajustadas, con su versión por defecto; tiempos medios de ajuste, predicción y búsqueda."""
    rho = t2.factor_nb(res)
    mae = matriz(res)
    filas = []
    for (nombre, variante), g in res.groupby(["configuracion", "variante"], sort=False):
        clave = nombre if variante == "linea_base" else f"{nombre} [{variante}]"
        tarea = g["tarea"].iloc[0]
        base = f"{tarea} {BASE_INGENUA}" if tarea == "T1" else f"{tarea} {BASE_GRUPO}"
        m, lo, hi = media_e_intervalo(mae[clave], rho, nivel)
        f = {"configuracion": nombre, "tarea": tarea, "variante": variante, "modelo": g["modelo"].iloc[0], "n_ajustes": int(mae[clave].notna().sum()),
             "mae_media": m, "mae_ic_inf": lo, "mae_ic_sup": hi, "rmse_media": float(g["rmse"].mean()), "r2_media": float(g["r2"].mean()),
             "t_ajuste_media_s": float(g["t_ajuste_s"].mean()), "t_prediccion_media_s": float(g["t_prediccion_s"].mean()),
             "t_busqueda_media_s": float(g["t_busqueda_s"].mean()) if variante == "ajustado" else np.nan, "linea_base": base}
        c = comparar_pareado(mae[base] - mae[clave], rho, nivel)
        f.update({"dif_base": c["dif_media"], "dif_base_ic_inf": c["dif_ic_inf"], "dif_base_ic_sup": c["dif_ic_sup"], "mejora_base_pct": 100 * c["dif_media"] / float(mae[base].mean()),
                  "p_wilcoxon_base": c["p_wilcoxon"], "p_t_base": c["p_t_corregido"], "gana_base": c["prop_gana"]})
        if variante == "ajustado":
            d = comparar_pareado(mae[f"{nombre} [defecto]"] - mae[clave], rho, nivel)
            f.update({"dif_defecto": d["dif_media"], "dif_defecto_ic_inf": d["dif_ic_inf"], "dif_defecto_ic_sup": d["dif_ic_sup"], "p_wilcoxon_defecto": d["p_wilcoxon"],
                      "p_t_defecto": d["p_t_corregido"], "gana_defecto": d["prop_gana"], "mae_interno_medio": float(g["mae_interno"].mean())})
        filas.append(f)
    return pd.DataFrame(filas)


def resumen_hiperparametros(res: pd.DataFrame) -> pd.DataFrame:
    """Hiperparámetros elegidos en los pliegues externos: el valor más frecuente (y cuántas veces), o la mediana y el rango si son continuos."""
    filas = []
    for nombre, g in res[res["variante"] == "ajustado"].groupby("configuracion", sort=False):
        elegidos = [json.loads(x) for x in g["hiperparametros"]]
        for hp in sorted({k for e in elegidos for k in e}):
            vals = [e[hp] for e in elegidos if hp in e]
            if hp in CONTINUOS:
                v = np.asarray(vals, dtype=float)
                filas.append({"configuracion": nombre, "hiperparametro": hp, "resumen": f"mediana {np.median(v):.4g} (rango {v.min():.4g}–{v.max():.4g})", "n": len(v)})
            else:
                conteo = pd.Series([str(x) for x in vals]).value_counts()
                filas.append({"configuracion": nombre, "hiperparametro": hp, "resumen": f"{conteo.index[0]} ({int(conteo.iloc[0])} de {len(vals)})", "n": len(vals)})
    return pd.DataFrame(filas)
