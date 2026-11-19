"""Interpretabilidad de los modelos finales: SHAP, importancia por permutación y estabilidad de los coeficientes del ridge.

- **SHAP** (`shap.TreeExplainer`) para los dos xgboost finales (Tarea 1 B y Tarea 2 con clusters), ajustados con **todo** el dataset y los hiperparámetros finales.
  Los valores están en la escala del objetivo del modelo: xgboost se entrena sobre el logit de la tasa, así que un SHAP positivo sube la deserción predicha.
  Las variables son las columnas que ve el modelo (numéricas escaladas; una columna por categoría de sector y carácter; en la Tarea 2, el cluster y las distancias).
  La dirección es el signo de la correlación de Spearman entre el valor de la variable y su SHAP.
- **Importancia por permutación** en validación: GroupKFold por IES (5 pliegues, semilla 301), se permuta cada variable **original** en el pliegue de prueba (5 repeticiones)
  y se mide el aumento del MAE; promedio entre pliegues. Para compararla con SHAP, SHAP se agrega por variable original (las categorías de sector y carácter suman; los
  clusters, que se derivan de todas las variables, se dejan aparte).
- **Coeficientes del ridge** (Tarea 2 con información mutua, y Tarea 1 C solo historia): los coeficientes sobre variables estandarizadas en cada uno de los 25 pliegues de entrenamiento
  (5 × 5), con cuántas veces entra cada variable, su media ± desviación y si el signo es constante.
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance

from desercion_ies.evaluation import ajuste as aj
from desercion_ies.evaluation import tarea2 as t2
from desercion_ies.evaluation.evaluacion_final import construir_final
from desercion_ies.evaluation.experimentos import parametros_experimentos
from desercion_ies.evaluation.protocolo import particiones_repetidas

CLUSTERS = "clusters (agregados)"


def _limpio(nombre: str) -> str:
    return nombre.split("__", 1)[-1]


def _grupo(nombre: str) -> str:
    """Variable original de una columna del modelo: sector y carácter agrupan sus categorías; los clusters van aparte."""
    if nombre.startswith("cat__sector_"):
        return "sector"
    if nombre.startswith("cat__caracter_"):
        return "caracter"
    if nombre.startswith(("cluster_", "distancia_cluster_")):
        return CLUSTERS
    return _limpio(nombre)


def matriz_y_nombres(est, X: pd.DataFrame) -> tuple[np.ndarray, list[str], object]:
    """(matriz que recibe el modelo, nombres de sus columnas, modelo final) de un estimador ya ajustado, con o sin clusters."""
    interno = getattr(est, "regressor_", est)
    union = interno.named_steps.get("features")
    if union is None:
        pre = interno.named_steps["pre"]
        return pre.transform(X), list(pre.get_feature_names_out()), interno.named_steps["modelo"]
    partes = dict(union.transformer_list)
    base, cl = partes["base"], partes["clusters"]
    nombres = (
        list(base.named_steps["pre"].get_feature_names_out())
        + [f"cluster_{i}" for i in range(cl.k_)]
        + [f"distancia_cluster_{i}" for i in range(cl.k_)]
    )
    return union.transform(X), nombres, interno.named_steps["modelo"]


def calcular_shap(est, X: pd.DataFrame):
    """(valores SHAP, matriz, nombres) del modelo de árboles de un estimador ya ajustado."""
    import shap

    M, nombres, modelo = matriz_y_nombres(est, X)
    return shap.TreeExplainer(modelo).shap_values(M), M, nombres


def _direccion(valores: np.ndarray, shap_v: np.ndarray) -> float:
    if np.std(valores) == 0 or np.std(shap_v) == 0:
        return float("nan")
    return float(spearmanr(valores, shap_v).statistic)


def tabla_shap(
    sv: np.ndarray, M: np.ndarray, nombres: list[str], modelo: str, top: int | None = None
) -> pd.DataFrame:
    """Importancia por columna (|SHAP| medio) y dirección (Spearman del valor con su SHAP), de mayor a menor."""
    t = pd.DataFrame(
        {
            "modelo": modelo,
            "variable": [_limpio(n) for n in nombres],
            "mean_abs_shap": np.abs(sv).mean(0),
            "spearman": [_direccion(M[:, j], sv[:, j]) for j in range(M.shape[1])],
        }
    )
    t = t.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    t.insert(1, "rank", np.arange(1, len(t) + 1))
    return t.head(top) if top else t


def shap_por_variable(
    sv: np.ndarray, M: np.ndarray, nombres: list[str], modelo: str
) -> pd.DataFrame:
    """SHAP agregado por variable original (suma de las columnas de cada categoría o grupo de clusters) y dirección en las numéricas."""
    grupos = pd.Series([_grupo(n) for n in nombres])
    filas = []
    for g in grupos.unique():
        cols = np.flatnonzero((grupos == g).to_numpy())
        suma = sv[:, cols].sum(1)
        filas.append(
            {
                "modelo": modelo,
                "variable": g,
                "mean_abs_shap": float(np.abs(suma).mean()),
                "spearman": _direccion(M[:, cols[0]], sv[:, cols[0]])
                if len(cols) == 1
                else float("nan"),
            }
        )
    t = pd.DataFrame(filas).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    t.insert(1, "rank_shap", np.arange(1, len(t) + 1))
    return t


# ---------- permutación ----------
def _pliegue_permutacion(
    m, X, y, entrena, prueba, numericas, categoricas, p_exp, p_t2, n_rep, semilla
):
    est = construir_final(m, numericas, categoricas, p_exp, p_t2)
    est.fit(X.iloc[entrena], y.iloc[entrena].to_numpy())
    r = permutation_importance(
        est,
        X.iloc[prueba],
        y.iloc[prueba].to_numpy(),
        scoring="neg_mean_absolute_error",
        n_repeats=n_rep,
        random_state=semilla,
        n_jobs=1,
    )
    return r.importances_mean


def permutacion(cfg: dict, df: pd.DataFrame, m: dict, n_jobs: int = -1) -> pd.DataFrame:
    """Aumento del MAE al permutar cada variable original en el pliegue de prueba (media y desviación entre los 5 pliegues)."""
    p = cfg["modelado"]["sesgo_sensibilidad"]
    p_exp, p_t2 = parametros_experimentos(cfg), t2.parametros_tarea2(cfg)
    X, num, cat = aj.preparar(m["spec"], cfg, df)
    y = df["tasa_desercion"].astype(float)
    pliegues = particiones_repetidas(df, p["n_splits"], [p["permutacion"]["semilla"]])[0]
    res = Parallel(n_jobs=n_jobs)(
        delayed(_pliegue_permutacion)(
            m,
            X,
            y,
            e,
            t,
            num,
            cat,
            p_exp,
            p_t2,
            p["permutacion"]["n_repeticiones"],
            p["permutacion"]["semilla"],
        )
        for e, t in pliegues
    )
    a = np.vstack(res)
    t = pd.DataFrame(
        {
            "modelo": m["nombre"],
            "variable": list(X.columns),
            "aumento_mae_medio": a.mean(0),
            "aumento_mae_sd": a.std(0, ddof=1),
            "n_pliegues": len(a),
        }
    )
    t = t.sort_values("aumento_mae_medio", ascending=False).reset_index(drop=True)
    t.insert(2, "rank_permutacion", np.arange(1, len(t) + 1))
    return t


# ---------- coeficientes del ridge ----------
def _coeficientes(m, X, y, entrena, num, cat, p_exp, p_t2) -> pd.DataFrame:
    est = construir_final(m, num, cat, p_exp, p_t2)
    est.fit(X.iloc[entrena], y.iloc[entrena].to_numpy())
    interno = getattr(est, "regressor_", est)
    pre = interno.named_steps["pre"]
    nombres = np.array(pre.get_feature_names_out())
    if "red" in interno.named_steps:
        nombres = nombres[interno.named_steps["red"].get_support()]
    return pd.DataFrame(
        {"variable": [_limpio(n) for n in nombres], "coef": interno.named_steps["modelo"].coef_}
    )


def coeficientes_ridge(cfg: dict, df: pd.DataFrame, m: dict, n_jobs: int = -1) -> pd.DataFrame:
    """Coeficientes estandarizados de un ridge en los 25 pliegues de entrenamiento: veces seleccionada, media ± desviación y signo constante."""
    p = cfg["modelado"]["sesgo_sensibilidad"]
    p_exp, p_t2 = parametros_experimentos(cfg), t2.parametros_tarea2(cfg)
    X, num, cat = aj.preparar(m["spec"], cfg, df)
    y = df["tasa_desercion"].astype(float)
    entrenamientos = [
        e
        for pliegues in particiones_repetidas(df, p["n_splits"], p["semillas"])
        for e, _ in pliegues
    ]
    partes = Parallel(n_jobs=n_jobs)(
        delayed(_coeficientes)(m, X, y, e, num, cat, p_exp, p_t2) for e in entrenamientos
    )
    largo = pd.concat([d.assign(pliegue=i) for i, d in enumerate(partes)], ignore_index=True)
    filas = []
    for v, g in largo.groupby("variable"):
        c = g["coef"].to_numpy()
        filas.append(
            {
                "modelo": m["nombre"],
                "variable": v,
                "n_pliegues_seleccionada": len(c),
                "de_pliegues": len(entrenamientos),
                "media": float(c.mean()),
                "desviacion": float(c.std(ddof=1)) if len(c) > 1 else float("nan"),
                "signo_constante": bool((c > 0).all() or (c < 0).all()),
            }
        )
    t = pd.DataFrame(filas)
    return t.reindex(t["media"].abs().sort_values(ascending=False).index).reset_index(drop=True)
