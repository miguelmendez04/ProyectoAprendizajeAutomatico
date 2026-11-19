"""Protocolo de evaluación v3: GroupKFold por IES repetido, intervalos de confianza y prueba pareada contra la ingenua.

- **Pliegues:** `GroupKFold(n_splits, shuffle=True, random_state=semilla)` por `cod_ies`, repetido con varias semillas
  (5 repeticiones x 5 pliegues = 25 ajustes por configuración). Cada repetición se genera una sola vez y se usa en todas las
  configuraciones y líneas base.
- **Configuraciones:** un conjunto corto fijado de antemano en `config.yaml: modelado.protocolo_v3.configuraciones`.
- **Líneas base:** `base_ingenua` (tasa t-1 de la misma IES, o la mediana del entrenamiento si no hay), `base_ingenua_rezagada`
  (tasa t-2) y `base_mediana`. Se leen de las columnas `tasa_t1` y `tasa_t2` del dataset (que ya incluyen los años anteriores a 2015).
- **Intervalo de confianza e inferencia:** los 25 MAE por pliegue no son independientes (los conjuntos de entrenamiento se
  traslapan y las repeticiones reutilizan las mismas IES). Se usa la corrección de Nadeau y Bengio (2003): la varianza de la media
  se multiplica por `1/J + n_prueba/n_entrenamiento` en lugar de `1/J`. El IC de la media y la prueba t salen de esa varianza.
  Además se reporta la prueba de Wilcoxon de rangos con signo sobre las 25 diferencias pareadas (ingenua − configuración), que no
  corrige la dependencia y por eso es más optimista, y la proporción de pliegues en que la configuración gana.
"""
import logging
import time
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupKFold

from desercion_ies.evaluation.experimentos import hash_pliegue, metricas, parametros_experimentos
from desercion_ies.evaluation.formulaciones import (
    FORMULACIONES_REZAGADAS, TODAS_LAS_FORMULACIONES, evaluar_configuracion, variables_formulacion,
)
from desercion_ies.features.particion import cargar_dataset, preparar_x

BASE_INGENUA = "base_ingenua"
BASE_INGENUA_REZAGADA = "base_ingenua_rezagada"
BASE_MEDIANA = "base_mediana"
LINEAS_BASE = (BASE_INGENUA, BASE_INGENUA_REZAGADA, BASE_MEDIANA)
COLUMNAS_RESULTADO = [
    "configuracion", "tipo", "escenario", "formulacion", "modelo", "objetivo", "seleccion", "repeticion", "semilla", "pliegue",
    "n_entrenamiento", "n_prueba", "n_con_historia", "n_sin_historia", "mae", "rmse", "r2", "mae_con_historia", "mae_sin_historia",
    "t_ajuste_s", "t_prediccion_s", "hash_prueba",
]


log = logging.getLogger(__name__)


def parametros_protocolo(cfg: dict) -> dict:
    p = dict(cfg.get("modelado", {}).get("protocolo_v3") or {})
    if not p.get("configuraciones"):
        raise ValueError("falta modelado.protocolo_v3.configuraciones en config.yaml")
    p.setdefault("n_splits", 5)
    p.setdefault("semillas", [101, 102, 103, 104, 105])
    p.setdefault("nivel_confianza", 0.95)
    nombres = [c["nombre"] for c in p["configuraciones"]]
    if len(set(nombres)) != len(nombres) or set(nombres) & set(LINEAS_BASE):
        raise ValueError("los nombres de las configuraciones deben ser únicos y distintos de las líneas base")
    for c in p["configuraciones"]:
        if c["formulacion"] not in TODAS_LAS_FORMULACIONES:
            raise ValueError(f"formulación inválida en {c['nombre']!r}: {c['formulacion']!r}")
    return p


def escenario(formulacion: str) -> str:
    """`rezagada` si la formulación no usa la tasa de t-1; `t-1` en caso contrario."""
    return "rezagada" if formulacion in FORMULACIONES_REZAGADAS else "t-1"


def particiones_repetidas(df: pd.DataFrame, n_splits: int, semillas: Sequence[int], grupo: str = "cod_ies") -> list[list[tuple[np.ndarray, np.ndarray]]]:
    """Una lista de pliegues (entrenamiento, prueba) por semilla: GroupKFold por `grupo` con mezcla de grupos."""
    if n_splits > df[grupo].nunique():
        raise ValueError(f"n_splits={n_splits} supera el número de IES distintas")
    return [list(GroupKFold(n_splits=n_splits, shuffle=True, random_state=int(s)).split(df, groups=df[grupo])) for s in semillas]


def prediccion_base(df: pd.DataFrame, y: pd.Series, entrena: np.ndarray, prueba: np.ndarray, columna: str) -> np.ndarray:
    """La columna `columna` (`tasa_t1` o `tasa_t2`) de las filas de prueba; si falta, la mediana de la tasa de entrenamiento."""
    valor = df[columna].to_numpy(dtype=float)[prueba]
    return np.where(np.isnan(valor), float(np.median(y.iloc[entrena])), valor)


def _fila_base(nombre: str, rep: int, semilla: int, k: int, df: pd.DataFrame, y: pd.Series, entrena, prueba, pred) -> dict:
    con = df["tasa_t1"].notna().to_numpy()[prueba]
    real = y.iloc[prueba].to_numpy()
    mae_sub = lambda m: float(np.mean(np.abs(real[m] - pred[m]))) if m.any() else float("nan")  # noqa: E731
    return {
        "configuracion": nombre, "tipo": "linea_base", "escenario": "rezagada" if nombre == BASE_INGENUA_REZAGADA else "t-1",
        "formulacion": "linea_base", "modelo": nombre.removeprefix("base_"), "objetivo": "", "seleccion": "", "repeticion": rep,
        "semilla": semilla, "pliegue": k, "n_entrenamiento": len(entrena), "n_prueba": len(prueba), "n_con_historia": int(con.sum()),
        "n_sin_historia": int((~con).sum()), **metricas(real, pred), "mae_con_historia": mae_sub(con), "mae_sin_historia": mae_sub(~con),
        "t_ajuste_s": 0.0, "t_prediccion_s": 0.0, "hash_prueba": hash_pliegue(df, prueba),
    }


def evaluar_lineas_base(df: pd.DataFrame, y: pd.Series, repeticiones, semillas) -> list[dict]:
    filas = []
    for rep, (pliegues, semilla) in enumerate(zip(repeticiones, semillas)):
        for k, (entrena, prueba) in enumerate(pliegues):
            mediana = np.full(len(prueba), float(np.median(y.iloc[entrena])))
            for nombre, pred in ((BASE_INGENUA, prediccion_base(df, y, entrena, prueba, "tasa_t1")),
                                 (BASE_INGENUA_REZAGADA, prediccion_base(df, y, entrena, prueba, "tasa_t2")),
                                 (BASE_MEDIANA, mediana)):
                filas.append(_fila_base(nombre, rep, int(semilla), k, df, y, entrena, prueba, pred))
    return filas


def ejecutar_protocolo(cfg: dict, df: pd.DataFrame | None = None, verbose: bool = False) -> pd.DataFrame:
    """Líneas base y configuraciones fijadas, con todas las repeticiones de pliegues. Una fila por configuración, repetición y pliegue."""
    p, p_exp = parametros_protocolo(cfg), parametros_experimentos(cfg)
    if df is None:
        df = cargar_dataset(cfg)
    y = df["tasa_desercion"].astype(float)
    repeticiones = particiones_repetidas(df, p["n_splits"], p["semillas"])  # una sola vez: los mismos pliegues para todo
    filas = evaluar_lineas_base(df, y, repeticiones, p["semillas"])
    cache: dict[str, tuple] = {}
    for i, c in enumerate(p["configuraciones"], 1):
        t0 = time.perf_counter()
        conjunto = TODAS_LAS_FORMULACIONES[c["formulacion"]]["conjunto"]
        if conjunto not in cache:
            numericas, categoricas = variables_formulacion(df, cfg, conjunto)
            cache[conjunto] = (preparar_x(df, numericas, categoricas), numericas, categoricas)
        X, numericas, categoricas = cache[conjunto]
        config = {"formulacion": c["formulacion"], "modelo": c["modelo"], "objetivo": c["objetivo"], "imputacion": "mediana",
                  "seleccion": c["seleccion"]}
        for rep, (pliegues, semilla) in enumerate(zip(repeticiones, p["semillas"])):
            for f in evaluar_configuracion(config, df, y, X, pliegues, numericas, categoricas, p_exp, recortar=True):
                filas.append({**f, "configuracion": c["nombre"], "escenario": escenario(c["formulacion"]), "repeticion": rep, "semilla": int(semilla)})
        if verbose:
            log.info(f"[{i}/{len(p['configuraciones'])}] {c['nombre']}: {time.perf_counter() - t0:.1f} s")
    return pd.DataFrame(filas, columns=COLUMNAS_RESULTADO)


# ---------- inferencia ----------
def factor_nadeau_bengio(resultados: pd.DataFrame) -> float:
    """`n_prueba / n_entrenamiento` medio de los pliegues (0,25 con 5 pliegues)."""
    una = resultados[resultados["configuracion"] == BASE_INGENUA]
    return float((una["n_prueba"] / una["n_entrenamiento"]).mean())


def media_e_intervalo(valores, rho: float, nivel: float = 0.95) -> tuple[float, float, float]:
    """(media, límite inferior, límite superior) con la varianza corregida de Nadeau y Bengio: (1/J + rho) · s²."""
    v = np.asarray(valores, dtype=float)
    v = v[~np.isnan(v)]
    j = len(v)
    if j < 2:
        return (float(v.mean()) if j else float("nan"), float("nan"), float("nan"))
    se = np.sqrt((1.0 / j + rho) * v.var(ddof=1))
    h = stats.t.ppf((1 + nivel) / 2, j - 1) * se
    return float(v.mean()), float(v.mean() - h), float(v.mean() + h)


def comparar_pareado(dif, rho: float, nivel: float = 0.95) -> dict:
    """Inferencia sobre las diferencias pareadas por pliegue `dif = referencia − configuración` (positiva = gana la configuración).

    Devuelve la diferencia media con su IC, el p de Wilcoxon (sobre los pares), el p de la prueba t corregida y la proporción
    de pliegues en que la configuración gana (error estrictamente menor).
    """
    d = np.asarray(dif, dtype=float)
    d = d[~np.isnan(d)]
    j = len(d)
    media, lo, hi = media_e_intervalo(d, rho, nivel)
    p_t = float("nan")
    if j > 1 and d.std(ddof=1) > 0:
        t = d.mean() / np.sqrt((1.0 / j + rho) * d.var(ddof=1))
        p_t = float(2 * stats.t.sf(abs(t), j - 1))
    try:
        p_w = float(stats.wilcoxon(d).pvalue) if j > 0 and np.any(d != 0) else float("nan")
    except ValueError:
        p_w = float("nan")
    return {"n_pares": j, "dif_media": media, "dif_ic_inf": lo, "dif_ic_sup": hi, "p_wilcoxon": p_w, "p_t_corregido": p_t,
            "prop_gana": float((d > 0).mean()) if j else float("nan")}


def por_pliegue(resultados: pd.DataFrame, columna: str = "mae") -> pd.DataFrame:
    """Matriz (repetición, pliegue) × configuración del `columna` indicado."""
    return resultados.pivot(index=["repeticion", "pliegue"], columns="configuracion", values=columna)


def resumen(resultados: pd.DataFrame, nivel: float = 0.95) -> pd.DataFrame:
    """Por configuración: media e IC 95 % del MAE (total, con y sin historia) y comparación pareada con la ingenua (t-1) y con la
    ingenua rezagada, para el MAE total y para cada subconjunto."""
    rho = factor_nadeau_bengio(resultados)
    orden = list(dict.fromkeys(resultados["configuracion"]))
    meta = resultados.groupby("configuracion").first()[["tipo", "escenario", "formulacion", "modelo", "objetivo", "seleccion"]]
    filas = []
    columnas = {"": "mae", "_con_historia": "mae_con_historia", "_sin_historia": "mae_sin_historia"}
    matrices = {suf: por_pliegue(resultados, col) for suf, col in columnas.items()}
    for nombre in orden:
        fila = {"configuracion": nombre, **meta.loc[nombre].to_dict(), "n_ajustes": int(matrices[""][nombre].notna().sum())}
        for suf, m in matrices.items():
            media, lo, hi = media_e_intervalo(m[nombre], rho, nivel)
            fila.update({f"mae{suf}_media": media, f"mae{suf}_ic_inf": lo, f"mae{suf}_ic_sup": hi})
            for ref, tag in ((BASE_INGENUA, "ing"), (BASE_INGENUA_REZAGADA, "rez")):
                c = comparar_pareado(m[ref] - m[nombre], rho, nivel)
                ref_media = float(np.nanmean(m[ref]))
                fila.update({f"mejora_{tag}{suf}_pct": 100 * c["dif_media"] / ref_media if ref_media else float("nan"),
                             f"dif_{tag}{suf}": c["dif_media"], f"dif_{tag}{suf}_ic_inf": c["dif_ic_inf"], f"dif_{tag}{suf}_ic_sup": c["dif_ic_sup"],
                             f"p_wilcoxon_{tag}{suf}": c["p_wilcoxon"], f"p_t_{tag}{suf}": c["p_t_corregido"], f"gana_{tag}{suf}": c["prop_gana"],
                             f"n_pares_{tag}{suf}": c["n_pares"]})
        filas.append(fila)
    return pd.DataFrame(filas)
