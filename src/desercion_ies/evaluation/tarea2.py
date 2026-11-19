"""Tarea 2: estimación sin historia («arranque en frío») con la formulación A de contexto.

Predice la tasa de una IES solo con su contexto institucional y territorial: sin ninguna variable de historia (`COLUMNAS_HISTORIA`,
`COLUMNAS_HISTORIA_REZAGADA`) ni de `NUNCA_EXPLICATIVAS`. La validación agrupa por IES, así que cada IES de prueba es nueva para el
modelo, como lo sería una institución sin historia.

Protocolo en dos etapas (para no seleccionar sobre los mismos pliegues):
a. **Cribado:** GroupKFold por IES (5 pliegues, una semilla) sobre toda la grilla (objetivo × imputación × reducción × clusters × modelo).
b. **Confirmación:** protocolo de la v3 (GroupKFold repetido 5 × 5, IC 95 % y t corregida de Nadeau y Bengio) con semillas distintas a
   la del cribado, para las líneas base, la mejor configuración de cada modelo y la mejor de cada método de reducción del cribado.

Líneas base: `base_mediana` (mediana de la tasa del entrenamiento) y `base_mediana_grupo` (mediana por carácter × sector del
entrenamiento; si el grupo no existe en el entrenamiento, la mediana global). Las predicciones finales se recortan a [0, 1].
"""
import itertools
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

from desercion_ies.evaluation.experimentos import (
    _inversa_logit, _logit, construir_modelo, construir_seleccion, hash_pliegue, metricas, parametros_experimentos,
)
from desercion_ies.evaluation.formulaciones import variables_formulacion
from desercion_ies.models.metodos import AgregarClusters, AutoencoderPequeno, KernelPCAComoPCA
from desercion_ies.features.particion import cargar_dataset, construir_preprocesador, preparar_x
from desercion_ies.evaluation.protocolo import comparar_pareado, media_e_intervalo, particiones_repetidas

BASE_MEDIANA = "base_mediana"
BASE_GRUPO = "base_mediana_grupo"
REDUCCIONES = ("ninguna", "info_mutua", "pca", "kernel_pca", "autoencoder")
OBJETIVOS = ("ninguna", "logit")
IMPUTACIONES = ("mediana", "knn")
MODELOS = ("ridge", "random_forest", "xgboost")
FACTORES = ("modelo", "objetivo", "imputacion", "reduccion", "clusters")
COLUMNAS_RESULTADO = [
    "configuracion", "tipo", "modelo", "objetivo", "imputacion", "reduccion", "clusters", "etapa", "repeticion", "semilla", "pliegue",
    "n_entrenamiento", "n_prueba", "mae", "rmse", "r2", "k_clusters", "n_respaldo_grupo", "t_ajuste_s", "t_prediccion_s", "hash_prueba",
]
_DEFECTO = {
    "semilla_cribado": 11, "semillas_confirmacion": [201, 202, 203, 204, 205], "n_splits": 5, "nivel_confianza": 0.95, "n_jobs": -1,
    "grilla": {"objetivo": list(OBJETIVOS), "imputacion": list(IMPUTACIONES), "reduccion": list(REDUCCIONES), "clusters": [False, True],
               "modelo": list(MODELOS)},
    "kernel_pca": {"gamma": None},
    "autoencoder": {"oculta": 16, "cuello": 8, "activacion": "tanh", "max_iter": 500, "validation_fraction": 0.15, "n_iter_no_change": 15,
                    "alpha": 1e-4},
    "clusters": {"k_min": 3, "k_max": 8, "n_init": 10},
}


def parametros_tarea2(cfg: dict) -> dict:
    """`modelado.tarea2` con valores por defecto; valida la grilla y que la confirmación no reutilice la semilla del cribado."""
    p = {k: (dict(v) if isinstance(v, dict) else v) for k, v in _DEFECTO.items()}
    dado = cfg.get("modelado", {}).get("tarea2") or {}
    for k, v in dado.items():
        p[k] = {**p[k], **v} if isinstance(v, dict) and isinstance(p.get(k), dict) else v
    for factor, validos in (("objetivo", OBJETIVOS), ("imputacion", IMPUTACIONES), ("reduccion", REDUCCIONES), ("modelo", MODELOS), ("clusters", (False, True))):
        malos = [v for v in p["grilla"][factor] if v not in validos]
        if malos:
            raise ValueError(f"modelado.tarea2.grilla.{factor}: valores inválidos {malos}; los válidos son {list(validos)}")
    if p["semilla_cribado"] in p["semillas_confirmacion"]:
        raise ValueError("las semillas de la confirmación deben ser distintas a la del cribado (para no seleccionar sobre los mismos pliegues)")
    return p


def grilla(p: dict) -> list[dict]:
    g = p["grilla"]
    return [{"objetivo": o, "imputacion": i, "reduccion": r, "clusters": bool(c), "modelo": m}
            for o, i, r, c, m in itertools.product(g["objetivo"], g["imputacion"], g["reduccion"], g["clusters"], g["modelo"])]


def nombre_configuracion(c: dict) -> str:
    return f"{c['modelo']} | {c['objetivo']} | {c['imputacion']} | {c['reduccion']} | {'con clusters' if c['clusters'] else 'sin clusters'}"


# ---------- datos ----------
def preparar_datos(cfg: dict, df: pd.DataFrame | None = None):
    """(df, X, y, numéricas, categóricas) de la formulación A: solo contexto, sin historia ni `NUNCA_EXPLICATIVAS`."""
    if df is None:
        df = cargar_dataset(cfg)
    numericas, categoricas = variables_formulacion(df, cfg, "contexto")
    return df, preparar_x(df, numericas, categoricas), df["tasa_desercion"].astype(float), numericas, categoricas


# ---------- pipeline ----------
def construir_reductor(nombre: str, p_exp: dict, p: dict):
    """Paso de selección o reducción (`None` si es `ninguna`)."""
    if nombre == "ninguna":
        return None
    if nombre in ("info_mutua", "pca"):
        return construir_seleccion(nombre, p_exp)
    if nombre == "kernel_pca":
        return KernelPCAComoPCA(varianza=p_exp["varianza_pca"], gamma=p["kernel_pca"]["gamma"], random_state=p_exp["semilla"])
    if nombre == "autoencoder":
        return AutoencoderPequeno(**p["autoencoder"], random_state=p_exp["semilla"])
    raise ValueError(f"reduccion inválida: {nombre!r}; debe ser una de {REDUCCIONES}")


def construir_pipeline(c: dict, numericas: list[str], categoricas: list[str], p_exp: dict, p: dict):
    """Pipeline: preprocesador → [reducción] → modelo; con `clusters`, `FeatureUnion([base, clusters])` antes del modelo.

    Los clusters (cluster en one-hot y distancias a los centroides) se **agregan** a lo que salga de la reducción: no se reducen ni
    reemplazan nada. Con objetivo en logit se envuelve en `TransformedTargetRegressor`. Todo se ajusta solo en `fit`.
    """
    pasos = [("pre", construir_preprocesador(numericas, categoricas, c["imputacion"], p_exp["k_vecinos"]))]
    reductor = construir_reductor(c["reduccion"], p_exp, p)
    if reductor is not None:
        pasos.append(("red", reductor))
    modelo = construir_modelo(c["modelo"], {**p_exp, "n_jobs": 1})
    if c["clusters"]:
        k = p["clusters"]
        union = FeatureUnion([("base", Pipeline(pasos)),
                              ("clusters", AgregarClusters(numericas, categoricas, k["k_min"], k["k_max"], k["n_init"], p_exp["semilla"]))])
        pipe = Pipeline([("features", union), ("modelo", modelo)])
    else:
        pipe = Pipeline(pasos + [("modelo", modelo)])
    if c["objetivo"] == "logit":
        return TransformedTargetRegressor(regressor=pipe, transformer=FunctionTransformer(_logit, _inversa_logit, check_inverse=False))
    return pipe


def k_elegido(pipe) -> int | None:
    """k del K-means de un pipeline ya ajustado (`None` si no lleva clusters)."""
    interno = getattr(pipe, "regressor_", pipe)
    union = interno.named_steps.get("features")
    return int(dict(union.transformer_list)["clusters"].k_) if union is not None else None


# ---------- líneas base ----------
def prediccion_mediana_grupo(df: pd.DataFrame, y: pd.Series, entrena: np.ndarray, prueba: np.ndarray,
                             columnas: tuple[str, str] = ("caracter", "sector")) -> tuple[np.ndarray, int]:
    """(predicción, filas con respaldo): la mediana de la tasa del entrenamiento por carácter × sector; si el grupo no está en el
    entrenamiento, la mediana global del entrenamiento."""
    clave = df[list(columnas)].astype(str).agg("|".join, axis=1)
    medianas = y.iloc[entrena].groupby(clave.iloc[entrena]).median()
    pred = clave.iloc[prueba].map(medianas)
    respaldo = pred.isna()
    return pred.fillna(float(np.median(y.iloc[entrena]))).to_numpy(dtype=float), int(respaldo.sum())


def _fila(nombre, tipo, c, etapa, rep, semilla, k, entrena, prueba, real, pred, hash_p, t_ajuste=0.0, t_pred=0.0, k_cl=None, n_resp=None) -> dict:
    return {
        "configuracion": nombre, "tipo": tipo, "modelo": c.get("modelo", nombre.removeprefix("base_")), "objetivo": c.get("objetivo", ""),
        "imputacion": c.get("imputacion", ""), "reduccion": c.get("reduccion", ""), "clusters": c.get("clusters", ""), "etapa": etapa,
        "repeticion": rep, "semilla": semilla, "pliegue": k, "n_entrenamiento": len(entrena), "n_prueba": len(prueba), **metricas(real, pred),
        "k_clusters": k_cl, "n_respaldo_grupo": n_resp, "t_ajuste_s": t_ajuste, "t_prediccion_s": t_pred, "hash_prueba": hash_p,
    }


def lineas_base(df: pd.DataFrame, y: pd.Series, etapa: str, repeticiones, semillas) -> list[dict]:
    filas = []
    for rep, (pliegues, semilla) in enumerate(zip(repeticiones, semillas)):
        for k, (entrena, prueba) in enumerate(pliegues):
            real, h = y.iloc[prueba].to_numpy(), hash_pliegue(df, prueba)
            filas.append(_fila(BASE_MEDIANA, "linea_base", {}, etapa, rep, int(semilla), k, entrena, prueba, real,
                               np.full(len(prueba), float(np.median(y.iloc[entrena]))), h, n_resp=0))
            pred, n_resp = prediccion_mediana_grupo(df, y, entrena, prueba)
            filas.append(_fila(BASE_GRUPO, "linea_base", {}, etapa, rep, int(semilla), k, entrena, prueba, real, pred, h, n_resp=n_resp))
    return filas


# ---------- evaluación en paralelo ----------
def _tarea(c: dict, etapa: str, rep: int, semilla: int, k: int, entrena: np.ndarray, prueba: np.ndarray, X: pd.DataFrame, y: pd.Series,
           hash_p: str, numericas: list[str], categoricas: list[str], p_exp: dict, p: dict) -> dict:
    """Ajusta una configuración en un pliegue y mide en la prueba (escala de tasa, recortada a [0, 1]). Función de módulo: la usa joblib."""
    pipe = construir_pipeline(c, numericas, categoricas, p_exp, p)
    t0 = time.perf_counter()
    pipe.fit(X.iloc[entrena], y.iloc[entrena].to_numpy())
    t_ajuste = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = np.clip(np.asarray(pipe.predict(X.iloc[prueba]), dtype=float), 0.0, 1.0)
    t_pred = time.perf_counter() - t0
    return _fila(nombre_configuracion(c), "modelo", c, etapa, rep, int(semilla), k, entrena, prueba, y.iloc[prueba].to_numpy(), pred, hash_p,
                 t_ajuste, t_pred, k_elegido(pipe))


def evaluar(cfg: dict, configs: list[dict], etapa: str, semillas: list[int], df: pd.DataFrame | None = None, verbose: bool = False,
            grupo: str = "cod_ies") -> pd.DataFrame:
    """Líneas base y `configs` con los mismos pliegues (una repetición por semilla), en paralelo sobre (configuración, pliegue).

    `grupo` es la columna por la que se agrupan los pliegues: `cod_ies` (por defecto) o, para el arranque en frío estricto, `cod_divipola_mpio`."""
    p_exp, p = parametros_experimentos(cfg), parametros_tarea2(cfg)
    df, X, y, numericas, categoricas = preparar_datos(cfg, df)
    repeticiones = particiones_repetidas(df, p["n_splits"], semillas, grupo)  # una sola vez
    filas = lineas_base(df, y, etapa, repeticiones, semillas)
    hashes = [[hash_pliegue(df, prueba) for _, prueba in pliegues] for pliegues in repeticiones]
    tareas = [(c, rep, int(semilla), k, entrena, prueba, hashes[rep][k])
              for c in configs for rep, (pliegues, semilla) in enumerate(zip(repeticiones, semillas)) for k, (entrena, prueba) in enumerate(pliegues)]
    resultados = Parallel(n_jobs=p["n_jobs"], verbose=10 if verbose else 0)(
        delayed(_tarea)(c, etapa, rep, semilla, k, entrena, prueba, X, y, h, numericas, categoricas, p_exp, p)
        for c, rep, semilla, k, entrena, prueba, h in tareas)
    return pd.DataFrame(filas + resultados, columns=COLUMNAS_RESULTADO)


def cribado(cfg: dict, df: pd.DataFrame | None = None, verbose: bool = False) -> pd.DataFrame:
    """Etapa a: la grilla completa con GroupKFold por IES (5 pliegues, una semilla)."""
    p = parametros_tarea2(cfg)
    return evaluar(cfg, grilla(p), "cribado", [p["semilla_cribado"]], df, verbose)


def confirmacion(cfg: dict, configs: list[dict], df: pd.DataFrame | None = None, verbose: bool = False) -> pd.DataFrame:
    """Etapa b: protocolo de la v3 (5 × 5) con semillas distintas a las del cribado."""
    p = parametros_tarea2(cfg)
    return evaluar(cfg, configs, "confirmacion", list(p["semillas_confirmacion"]), df, verbose)


# ---------- resúmenes ----------
def resumen_cribado(res: pd.DataFrame) -> pd.DataFrame:
    """Media y desviación entre pliegues por configuración, ordenado por MAE medio."""
    g = res.groupby("configuracion", sort=False)
    out = g.agg(tipo=("tipo", "first"), modelo=("modelo", "first"), objetivo=("objetivo", "first"), imputacion=("imputacion", "first"),
                reduccion=("reduccion", "first"), clusters=("clusters", "first"), mae_media=("mae", "mean"), mae_desv=("mae", "std"),
                rmse_media=("rmse", "mean"), r2_media=("r2", "mean"), t_ajuste_media_s=("t_ajuste_s", "mean")).reset_index()
    return out.sort_values("mae_media", kind="stable").reset_index(drop=True)


def efecto_por_nivel(r: pd.DataFrame, factor: str) -> pd.DataFrame:
    """Por nivel de un factor: MAE medio de sus configuraciones, el mejor y su configuración (sobre las demás combinaciones)."""
    m = r[r["tipo"] == "modelo"]
    filas = []
    for nivel, g in m.groupby(factor, sort=False):
        mejor = g.iloc[0]  # `r` viene ordenado por MAE
        filas.append({"factor": factor, "nivel": nivel, "n_configuraciones": len(g), "mae_medio": g["mae_media"].mean(), "mae_mediano": g["mae_media"].median(),
                      "mae_minimo": g["mae_media"].min(), "mejor_configuracion": mejor["configuracion"]})
    return pd.DataFrame(filas).sort_values("mae_medio").reset_index(drop=True)


def efecto_contraste(r: pd.DataFrame, factor: str, referencia) -> pd.DataFrame:
    """Efecto de cada nivel frente a `referencia` con las demás decisiones iguales: diferencia de MAE (nivel − referencia) en cada
    par de configuraciones que solo difieren en ese factor; negativa = el nivel reduce el error."""
    m = r[r["tipo"] == "modelo"]
    otros = [f for f in FACTORES if f != factor]
    ref = m[m[factor] == referencia][otros + ["mae_media"]].rename(columns={"mae_media": "mae_ref"})
    filas = []
    for nivel in [n for n in m[factor].unique() if n != referencia]:
        par = m[m[factor] == nivel][otros + ["mae_media"]].merge(ref, on=otros)
        d = par["mae_media"] - par["mae_ref"]
        filas.append({"factor": factor, "nivel": nivel, "referencia": referencia, "n_pares": len(par), "dif_media": float(d.mean()),
                      "dif_mediana": float(d.median()), "prop_mejora": float((d < 0).mean())})
    return pd.DataFrame(filas)


def seleccion_confirmacion(r: pd.DataFrame) -> list[dict]:
    """La mejor configuración (por MAE medio del cribado) de cada modelo y de cada método de reducción, sin repetir."""
    m = r[r["tipo"] == "modelo"].sort_values("mae_media", kind="stable")
    elegidas = list(m.groupby("modelo", sort=False).head(1)["configuracion"]) + list(m.groupby("reduccion", sort=False).head(1)["configuracion"])
    unicas = list(dict.fromkeys(elegidas))
    cols = ["objetivo", "imputacion", "reduccion", "clusters", "modelo"]
    return [{**m[m["configuracion"] == n].iloc[0][cols].to_dict(), "clusters": bool(m[m["configuracion"] == n].iloc[0]["clusters"])} for n in unicas]


def factor_nb(res: pd.DataFrame) -> float:
    return float((res["n_prueba"] / res["n_entrenamiento"]).mean())


def matriz(res: pd.DataFrame, columna: str = "mae") -> pd.DataFrame:
    return res.pivot(index=["repeticion", "pliegue"], columns="configuracion", values=columna)


def resumen_confirmacion(res: pd.DataFrame, nivel: float = 0.95) -> pd.DataFrame:
    """Por configuración: MAE medio con IC 95 % (Nadeau y Bengio) y comparación pareada con la mediana por grupo y con la mediana."""
    rho = factor_nb(res)
    mae, rmse, r2 = matriz(res, "mae"), matriz(res, "rmse"), matriz(res, "r2")
    meta = res.groupby("configuracion").first()[["tipo", "modelo", "objetivo", "imputacion", "reduccion", "clusters"]]
    filas = []
    for nombre in dict.fromkeys(res["configuracion"]):
        m, lo, hi = media_e_intervalo(mae[nombre], rho, nivel)
        fila = {"configuracion": nombre, **meta.loc[nombre].to_dict(), "n_ajustes": int(mae[nombre].notna().sum()), "mae_media": m, "mae_ic_inf": lo,
                "mae_ic_sup": hi, "rmse_media": float(rmse[nombre].mean()), "r2_media": float(r2[nombre].mean())}
        for ref, tag in ((BASE_GRUPO, "grupo"), (BASE_MEDIANA, "mediana")):
            c = comparar_pareado(mae[ref] - mae[nombre], rho, nivel)
            fila.update({f"dif_{tag}": c["dif_media"], f"dif_{tag}_ic_inf": c["dif_ic_inf"], f"dif_{tag}_ic_sup": c["dif_ic_sup"],
                         f"mejora_{tag}_pct": 100 * c["dif_media"] / float(mae[ref].mean()), f"p_wilcoxon_{tag}": c["p_wilcoxon"],
                         f"p_t_{tag}": c["p_t_corregido"], f"gana_{tag}": c["prop_gana"]})
        filas.append(fila)
    return pd.DataFrame(filas)


def diagnostico_autoencoder(cfg: dict, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Por pliegue del cribado: iteraciones del autoencoder, si paró antes del máximo (parada temprana) y su error de reconstrucción en
    entrenamiento y prueba (unidades estandarizadas). Preprocesamiento con mediana, ajustado solo con el entrenamiento."""
    p_exp, p = parametros_experimentos(cfg), parametros_tarea2(cfg)
    df, X, _, numericas, categoricas = preparar_datos(cfg, df)
    filas = []
    for k, (entrena, prueba) in enumerate(particiones_repetidas(df, p["n_splits"], [p["semilla_cribado"]])[0]):
        pre = construir_preprocesador(numericas, categoricas, "mediana", p_exp["k_vecinos"]).fit(X.iloc[entrena])
        ztr, zte = pre.transform(X.iloc[entrena]), pre.transform(X.iloc[prueba])
        ae = AutoencoderPequeno(**p["autoencoder"], random_state=p_exp["semilla"]).fit(ztr)
        filas.append({"pliegue": k, "n_iter": ae.n_iter_, "max_iter": ae.max_iter, "paro_temprano": bool(ae.paro_temprano_),
                      "mse_entrenamiento": float(np.mean((ae.mlp_.predict(ztr) - ztr) ** 2)), "mse_prueba": float(np.mean((ae.mlp_.predict(zte) - zte) ** 2))})
    return pd.DataFrame(filas)


# ---------- clusters ----------
def ajustar_clusters(cfg: dict, df: pd.DataFrame | None = None) -> tuple[AgregarClusters, pd.DataFrame, np.ndarray]:
    """Ajusta `AgregarClusters` con **todas** las filas (solo para describir; la evaluación ajusta dentro de cada pliegue)."""
    p_exp, p = parametros_experimentos(cfg), parametros_tarea2(cfg)
    df, X, _, numericas, categoricas = preparar_datos(cfg, df)
    k = p["clusters"]
    ag = AgregarClusters(numericas, categoricas, k["k_min"], k["k_max"], k["n_init"], p_exp["semilla"]).fit(X)
    return ag, df, ag.etiquetas(X)
