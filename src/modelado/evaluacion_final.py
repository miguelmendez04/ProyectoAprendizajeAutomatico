"""Modelos finales y evaluación fuera de muestra con los mismos pliegues (base de los análisis de sesgo y de sensibilidad).

**Modelos finales** (hiperparámetros ajustados del ajuste anidado):
- Tarea 1: naive (tasa de t-1), C solo historia con ridge (ajustado) y B historia + contexto con xgboost (ajustado).
- Tarea 2: mediana por carácter × sector, ridge con información mutua (por defecto y ajustado) y xgboost con clusters (ajustado).

**Hiperparámetros finales.** En el ajuste anidado cada pliegue externo eligió los suyos (25 por configuración). Los finales son el consenso:
el valor más frecuente de cada hiperparámetro (el menor en un empate) y, para los continuos (`alpha`, `learning_rate`, `reg_lambda`), su mediana.
Esa combinación no se validó como tal en el ajuste; se evalúa aquí con los mismos pliegues (semillas 301–305), y como esos hiperparámetros salieron
de los mismos datos, el resultado es algo optimista (igual para todos los escenarios y no afecta a las líneas base).

**Evaluación.** GroupKFold por IES, 5 pliegues × las semillas 301–305. Se guardan las predicciones fuera de muestra de cada fila en cada repetición
(`oof`: modelo → matriz repetición × fila), recortadas a [0, 1], y las métricas por pliegue.
"""
import json
import time
from collections import Counter

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.modelado import ajuste as aj
from src.modelado import tarea2 as t2
from src.modelado.experimentos import hash_pliegue, metricas, parametros_experimentos
from src.modelado.protocolo import comparar_pareado, media_e_intervalo, particiones_repetidas, prediccion_base

BASES = {"T1": "T1 base_ingenua", "T2": "T2 base_mediana_grupo"}
BASE_MEDIANA = {"T1": "T1 base_mediana", "T2": "T2 base_mediana"}
FINALES = [  # (nombre del modelo final, tarea, modelo del ajuste, variante)
    ("T1 C solo historia | ridge (ajustado)", "T1", "T1 C solo historia | ridge", "ajustado"),
    ("T1 B historia+contexto | xgboost logit (ajustado)", "T1", "T1 B historia+contexto | xgboost logit", "ajustado"),
    ("T2 ridge info_mutua (defecto)", "T2", "ridge", "defecto"),
    ("T2 ridge info_mutua (ajustado)", "T2", "ridge", "ajustado"),
    ("T2 xgboost con clusters (ajustado)", "T2", "xgboost", "ajustado"),
]
COLUMNAS_RESULTADO = ["configuracion", "tarea", "tipo", "repeticion", "semilla", "pliegue", "n_entrenamiento", "n_prueba", "mae", "rmse", "r2",
                      "t_ajuste_s", "t_prediccion_s", "hash_prueba"]


def hiperparametros_finales(ajuste: pd.DataFrame) -> dict[str, dict]:
    """Consenso por configuración: moda de cada hiperparámetro (el menor en un empate) o mediana si es continuo."""
    out = {}
    for nombre, g in ajuste[ajuste["variante"] == "ajustado"].groupby("configuracion", sort=False):
        elegidos = [json.loads(x) for x in g["hiperparametros"]]
        p = {}
        for hp in sorted({k for e in elegidos for k in e}):
            vals = [e[hp] for e in elegidos if hp in e]
            if hp in aj.CONTINUOS:
                p[hp] = float(np.median(vals))
            else:
                cuenta = Counter(json.dumps(v) for v in vals)
                p[hp] = json.loads(min(cuenta, key=lambda k: (-cuenta[k], k)))
        out[nombre] = p
    return out


def modelos_finales(cfg: dict, ajuste: pd.DataFrame | None = None) -> list[dict]:
    """Los cinco modelos finales como especificaciones: `nombre`, `tarea`, `spec` (la del ajuste), `variante` e `hiperparametros`."""
    p = cfg["modelado"]["sesgo_sensibilidad"]
    ajuste = pd.read_csv(p["ajuste"]) if ajuste is None else ajuste
    hp = hiperparametros_finales(ajuste)
    specs = aj.configuraciones_ajuste(cfg)
    por_nombre = {s["nombre"]: s for s in specs}
    t2s = {s["modelo"]: s for s in specs if s["tarea"] == "T2"}
    modelos = []
    for nombre, tarea, origen, variante in FINALES:
        spec = por_nombre[origen] if tarea == "T1" else t2s[origen]
        modelos.append({"nombre": nombre, "tarea": tarea, "spec": spec, "variante": variante,
                        "hiperparametros": hp[spec["nombre"]] if variante == "ajustado" else {}})
    return modelos


def construir_final(m: dict, numericas: list[str], categoricas: list[str], p_exp: dict, p_t2: dict):
    """Estimador sin ajustar del modelo final: por defecto o con los hiperparámetros consensuados."""
    est = aj.construir_estimador(m["spec"], numericas, categoricas, p_exp, p_t2)
    if m["variante"] == "defecto":
        return est
    est, _ = aj.estimador_ajustable(est, m["spec"], [])  # en ridge reemplaza RidgeCV por Ridge
    pre = aj._prefijo(est)
    params = {}
    for k, v in m["hiperparametros"].items():
        nombre = next(n for n in est.get_params() if n.endswith("red__k") or n.endswith("sel__k")) if k == "k" else f"{pre}modelo__{k}"
        params[nombre] = v
    return est.set_params(**params)


# ---------- evaluación ----------
def _tarea(m, idx, rep, semilla, k, entrena, prueba, X, y, numericas, categoricas, p_exp, p_t2, hash_p):
    est = construir_final(m, numericas, categoricas, p_exp, p_t2)
    t0 = time.perf_counter()
    est.fit(X.iloc[entrena], y.iloc[entrena].to_numpy())
    t_aj = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = np.clip(np.asarray(est.predict(X.iloc[prueba]), dtype=float), 0, 1)
    return idx, rep, k, pred, t_aj, time.perf_counter() - t0, hash_p


def _fila(nombre, tarea, tipo, rep, semilla, k, entrena, prueba, real, pred, hash_p, t_aj=0.0, t_pr=0.0) -> dict:
    return {"configuracion": nombre, "tarea": tarea, "tipo": tipo, "repeticion": rep, "semilla": int(semilla), "pliegue": k, "n_entrenamiento": len(entrena),
            "n_prueba": len(prueba), **metricas(real, pred), "t_ajuste_s": t_aj, "t_prediccion_s": t_pr, "hash_prueba": hash_p}


def evaluar_oof(cfg: dict, df: pd.DataFrame, modelos: list[dict], verbose: bool = False) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Modelos finales y líneas base con GroupKFold por IES 5 × 5: (métricas por pliegue, predicciones fuera de muestra por modelo).

    Líneas base: ingenua y mediana en la Tarea 1; mediana por carácter × sector y mediana en la Tarea 2. La ingenua usa `tasa_t1` del dataset recibido
    (en los escenarios que filtran filas, la historia ya está calculada y no se altera).
    """
    p = cfg["modelado"]["sesgo_sensibilidad"]
    p_exp, p_t2 = parametros_experimentos(cfg), t2.parametros_tarea2(cfg)
    y = df["tasa_desercion"].astype(float)
    reps = particiones_repetidas(df, p["n_splits"], p["semillas"])  # una sola vez
    datos = {m["nombre"]: aj.preparar(m["spec"], cfg, df) for m in modelos}
    n_rep = len(reps)
    oof = {m["nombre"]: np.full((n_rep, len(df)), np.nan) for m in modelos}
    filas = []
    for rep, (pliegues, semilla) in enumerate(zip(reps, p["semillas"])):
        for k, (entrena, prueba) in enumerate(pliegues):
            real, h = y.iloc[prueba].to_numpy(), hash_pliegue(df, prueba)
            med = np.full(len(prueba), float(np.median(y.iloc[entrena])))
            grupo, _ = t2.prediccion_mediana_grupo(df, y, entrena, prueba)
            for tarea, nombre, pred in (("T1", BASES["T1"], prediccion_base(df, y, entrena, prueba, "tasa_t1")), ("T1", BASE_MEDIANA["T1"], med),
                                        ("T2", BASES["T2"], grupo), ("T2", BASE_MEDIANA["T2"], med)):
                oof.setdefault(nombre, np.full((n_rep, len(df)), np.nan))[rep, prueba] = pred
                filas.append(_fila(nombre, tarea, "linea_base", rep, semilla, k, entrena, prueba, real, pred, h))
    tareas = [(m, i, rep, int(semilla), k, entrena, prueba)
              for i, m in enumerate(modelos) for rep, (pliegues, semilla) in enumerate(zip(reps, p["semillas"])) for k, (entrena, prueba) in enumerate(pliegues)]
    salida = Parallel(n_jobs=p["n_jobs"], batch_size=1, verbose=10 if verbose else 0)(
        delayed(_tarea)(m, i, rep, semilla, k, entrena, prueba, datos[m["nombre"]][0], y, datos[m["nombre"]][1], datos[m["nombre"]][2], p_exp, p_t2,
                        hash_pliegue(df, prueba)) for m, i, rep, semilla, k, entrena, prueba in tareas)
    for (m, i, rep, semilla, k, entrena, prueba), (_, _, _, pred, t_aj, t_pr, h) in zip(tareas, salida):
        oof[m["nombre"]][rep, prueba] = pred
        filas.append(_fila(m["nombre"], m["tarea"], "modelo", rep, semilla, k, entrena, prueba, y.iloc[prueba].to_numpy(), pred, h, t_aj, t_pr))
    return pd.DataFrame(filas, columns=COLUMNAS_RESULTADO), oof


# ---------- comparación con la línea base de la tarea ----------
def clase(dif: float, p_wilcoxon: float, p_t: float) -> str:
    """`supera` o `empeora` solo si las DOS pruebas son significativas (< 0,05); si no, `sin diferencia firme`."""
    firme = not (np.isnan(p_wilcoxon) or np.isnan(p_t)) and p_wilcoxon < 0.05 and p_t < 0.05
    return ("supera" if dif > 0 else "empeora") if firme else "sin diferencia firme"


def resumen_escenario(res: pd.DataFrame, nivel: float = 0.95) -> pd.DataFrame:
    """Por modelo y línea base: MAE medio con IC 95 % (Nadeau y Bengio) y, para los modelos, la diferencia pareada con la línea base de su tarea
    (positiva = el modelo tiene menos error), t corregida, Wilcoxon, proporción de pliegues que gana y la clase de la conclusión."""
    rho = t2.factor_nb(res)
    mae = t2.matriz(res, "mae")
    meta = res.groupby("configuracion").first()[["tarea", "tipo"]]
    filas = []
    for nombre in dict.fromkeys(res["configuracion"]):
        m, lo, hi = media_e_intervalo(mae[nombre], rho, nivel)
        f = {"configuracion": nombre, **meta.loc[nombre].to_dict(), "mae": m, "mae_ic_inf": lo, "mae_ic_sup": hi, "rmse": float(res.loc[res["configuracion"] == nombre, "rmse"].mean()),
             "r2": float(res.loc[res["configuracion"] == nombre, "r2"].mean())}
        base = BASES[f["tarea"]]
        if f["tipo"] == "modelo":
            c = comparar_pareado(mae[base] - mae[nombre], rho, nivel)
            f.update({"linea_base": base, "dif_base": c["dif_media"], "dif_base_ic_inf": c["dif_ic_inf"], "dif_base_ic_sup": c["dif_ic_sup"],
                      "mejora_base_pct": 100 * c["dif_media"] / float(mae[base].mean()), "p_wilcoxon": c["p_wilcoxon"], "p_t_corregida": c["p_t_corregido"],
                      "gana_base": c["prop_gana"], "clase": clase(c["dif_media"], c["p_wilcoxon"], c["p_t_corregido"])})
        filas.append(f)
    return pd.DataFrame(filas)
