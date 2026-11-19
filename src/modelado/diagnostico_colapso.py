"""Diagnóstico reproducible del colapso del ridge ajustado de la Tarea 2 en la partición prepandemia (entrena 2015–2017, prueba 2018–2019).

Solo diagnóstico: no cambia ningún modelo final. Parte de los hiperparámetros que la búsqueda interna eligió **en esa partición** (`temporal_predicciones.csv`)
y calcula, para las filas de prueba:
- el z de cada variable (ya imputada y escalada, como la ve el modelo) respecto del entrenamiento 2015–2017: máximo |z|, filas con |z| mayor que el umbral y
  filas fuera del rango observado en el entrenamiento, en general y en Bogotá;
- las filas de mayor error, con su municipio y las variables que más aportan a su predicción (coeficiente × z).

Y prueba si el colapso desaparece (i) quitando `mpio_matriculados_t2`, (ii) con el alfa por defecto (`RidgeCV`) y (iii) recortando las variables escaladas a [-recorte, recorte].
"""
import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import FunctionTransformer

from src.modelado import ajuste as aj
from src.modelado import tarea2 as t2
from src.modelado import temporal as tp
from src.modelado.experimentos import metricas, parametros_experimentos
from src.modelado.particion import ETIQUETAS_CARACTER, ETIQUETAS_SECTOR, cargar_dataset

VARIABLE_SOSPECHOSA = "mpio_matriculados_t2"
BOGOTA = "11001"


def _spec_ridge(cfg: dict) -> dict:
    return next(s for s in aj.configuraciones_ajuste(cfg) if s["tarea"] == "T2" and s["modelo"] == "ridge")


def hiperparametros_de_la_particion(cfg: dict, spec: dict, particion: str) -> dict:
    """Los hiperparámetros que la búsqueda interna eligió para el ridge de la Tarea 2 en esa partición temporal."""
    ruta = cfg["modelado"]["ajuste_temporal"]["salidas"]["temporal_predicciones"]
    p = pd.read_csv(ruta, usecols=["particion", "configuracion", "variante", "hiperparametros"])
    fila = p[(p["particion"] == particion) & (p["configuracion"] == spec["nombre"]) & (p["variante"] == "ajustado")].iloc[0]
    return json.loads(fila["hiperparametros"])


def _ridge_con(spec: dict, hp: dict, numericas, categoricas, p_exp, p_t2, recorte: float | None = None, alpha_defecto: bool = False, k: int | None = None):
    """Ridge de la Tarea 2 con `hp` (alfa y k); `alpha_defecto` deja `RidgeCV` (alfa por defecto del proyecto); `recorte` agrega el recorte de las variables escaladas."""
    est = aj.construir_estimador(spec, numericas, categoricas, p_exp, p_t2)
    est, _ = aj.estimador_ajustable(est, spec, [])  # tiene Ridge en lugar de RidgeCV
    pre = aj._prefijo(est)
    nombre_k = next(n for n in est.get_params() if n.endswith("red__k"))
    params = {nombre_k: int(k if k is not None else hp["k"])}
    if alpha_defecto:
        est = aj.construir_estimador(spec, numericas, categoricas, p_exp, p_t2)  # RidgeCV
        params = {nombre_k: int(k if k is not None else hp["k"])}
    else:
        params[f"{pre}modelo__alpha"] = float(hp["alpha"])
    est.set_params(**params)
    if recorte is not None:
        interno = est.regressor
        interno.steps.insert(1, ("recorte", FunctionTransformer(np.clip, kw_args={"a_min": -recorte, "a_max": recorte})))
    return est


def _metricas(df_test: pd.DataFrame, pred: np.ndarray, n_peores: int) -> dict:
    y = df_test["tasa_desercion"].to_numpy(dtype=float)
    err = np.abs(pred - y)
    bog = (df_test["cod_divipola_mpio"].astype(str) == BOGOTA).to_numpy()
    top = np.argsort(-err)[:n_peores]
    m = metricas(y, pred)
    out = {"mae": m["mae"], "rmse": m["rmse"], "r2": m["r2"], "mae_bogota": float(err[bog].mean()), "mae_resto": float(err[~bog].mean()),
           "peores_en_bogota": int(bog[top].sum()), "n_peores": len(top)}
    for anio, g in df_test.assign(err=err).groupby("anio"):
        out[f"mae_{int(anio)}"] = float(g["err"].mean())
    return out


def ejecutar_diagnostico(cfg: dict, df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(resumen por variante, tabla de z por variable, filas de mayor error con su municipio)."""
    p = cfg["modelado"]["sesgo_sensibilidad"]["colapso"]
    df = cargar_dataset(cfg) if df is None else df
    p_exp, p_t2 = parametros_experimentos(cfg), t2.parametros_tarea2(cfg)
    spec = _spec_ridge(cfg)
    hp = hiperparametros_de_la_particion(cfg, spec, p["particion"])
    rango = cfg["modelado"]["ajuste_temporal"]["particiones_temporales"][p["particion"]]
    entrena, prueba = tp.indices_temporales(df, tuple(rango["entrena"]), tuple(rango["prueba"]))
    X, num, cat = aj.preparar(spec, cfg, df)
    y = df["tasa_desercion"].astype(float).to_numpy()
    test = df.iloc[prueba]
    nombres_var = {
        "referencia": "Referencia: ridge ajustado (alfa y k de la búsqueda de la partición)",
        "i_sin_variable": f"(i) sin `{VARIABLE_SOSPECHOSA}`",
        "ii_alfa_defecto": "(ii) alfa por defecto (RidgeCV), mismo k",
        "ii_defecto_completo": "(ii bis) configuración por defecto completa (RidgeCV, k de `tarea2`)",
        "iii_recorte": f"(iii) variables escaladas recortadas a [-{p['recorte']:g}, {p['recorte']:g}]",
        "iv_solo_2017": "(iv, adicional) entrenando solo con 2017 (único año de entrenamiento con matrícula SNIES observada)",
    }
    k_defecto = p_exp["k_info_mutua"]
    ajustes = {
        "referencia": (X, num, dict()),
        "i_sin_variable": (X.drop(columns=[VARIABLE_SOSPECHOSA]), [c for c in num if c != VARIABLE_SOSPECHOSA], dict()),
        "ii_alfa_defecto": (X, num, {"alpha_defecto": True}),
        "ii_defecto_completo": (X, num, {"alpha_defecto": True, "k": k_defecto}),
        "iii_recorte": (X, num, {"recorte": p["recorte"]}),
        "iv_solo_2017": (X, num, {"entrena": np.flatnonzero((df["anio"] == 2017).to_numpy() & np.isin(np.arange(len(df)), entrena))}),
    }
    filas, ref_est = [], None
    for clave, (Xv, numv, kw) in ajustes.items():
        kw = dict(kw)
        entrena_v = kw.pop("entrena", entrena)  # la variante (iv) entrena con un subconjunto de las filas de entrenamiento
        est = _ridge_con(spec, hp, numv, cat, p_exp, p_t2, **kw)
        est.fit(Xv.iloc[entrena_v], y[entrena_v])
        pred = np.clip(np.asarray(est.predict(Xv.iloc[prueba]), dtype=float), 0, 1)
        if clave == "referencia":
            ref_est, ref_pred = est, pred
        modelo_ajustado = est.regressor_.named_steps["modelo"]
        alfa = float(modelo_ajustado.alpha_ if hasattr(modelo_ajustado, "alpha_") else modelo_ajustado.alpha)  # RidgeCV elige el suyo; Ridge usa el fijado
        interno_v = est.regressor_
        nombres_v = np.array(interno_v.named_steps["pre"].get_feature_names_out())[interno_v.named_steps["red"].get_support()]
        esta = [i for i, n in enumerate(nombres_v) if n.endswith(VARIABLE_SOSPECHOSA)]
        filas.append({"variante": clave, "descripcion": nombres_var[clave], "alpha": alfa,
                      "n_entrenamiento": len(entrena_v), "selecciona_variable": bool(esta), "coef_variable": float(modelo_ajustado.coef_[esta[0]]) if esta else np.nan, "k": int(est.get_params()[next(n for n in est.get_params() if n.endswith("red__k"))]),
                      **_metricas(test, pred, p["n_peores"])})
    resumen = pd.DataFrame(filas)

    # z de cada variable (imputada y escalada) respecto del entrenamiento, y las filas de mayor error de la referencia
    interno = ref_est.regressor_
    pre, red, modelo = interno.named_steps["pre"], interno.named_steps["red"], interno.named_steps["modelo"]
    Z = pre.transform(X)
    nombres = np.array(pre.get_feature_names_out())
    mu, sd = Z[entrena].mean(0), Z[entrena].std(0)
    sd = np.where(sd > 0, sd, 1.0)
    zt = (Z[prueba] - mu) / sd
    bog = (test["cod_divipola_mpio"].astype(str) == BOGOTA).to_numpy()
    sel = red.get_support()
    coef = np.zeros(len(nombres))
    coef[np.flatnonzero(sel)] = modelo.coef_
    z = pd.DataFrame({
        "variable": [n.split("__", 1)[-1] for n in nombres], "seleccionada": sel, "coef": coef,
        "z_max_abs": np.abs(zt).max(0), "z_max_abs_bogota": np.abs(zt[bog]).max(0) if bog.any() else np.nan,
        "pct_filas_z_sobre_umbral": 100 * (np.abs(zt) > p["z_umbral"]).mean(0),
        "pct_filas_fuera_del_rango": 100 * ((Z[prueba] < Z[entrena].min(0)) | (Z[prueba] > Z[entrena].max(0))).mean(0),
        "pct_bogota_fuera_del_rango": 100 * ((Z[prueba][bog] < Z[entrena].min(0)) | (Z[prueba][bog] > Z[entrena].max(0))).mean(0) if bog.any() else np.nan,
        "z_max_2019": np.abs(zt[(test["anio"] == 2019).to_numpy()]).max(0)}).sort_values("z_max_abs", ascending=False).reset_index(drop=True)
    err = np.abs(ref_pred - y[prueba])
    top = np.argsort(-err)[:p["n_peores"]]
    contrib = zt * coef
    filas = []
    for rango_, i in enumerate(top, 1):
        mejores = np.argsort(-np.abs(contrib[i]))[:3]
        f = {"rango": rango_, "cod_ies": str(test.iloc[i]["cod_ies"]), "municipio": str(test.iloc[i]["cod_divipola_mpio"]), "anio": int(test.iloc[i]["anio"]), "sector": ETIQUETAS_SECTOR.get(str(test.iloc[i]["sector"]), "desconocido"),
             "caracter": ETIQUETAS_CARACTER.get(str(test.iloc[i]["caracter"]), "desconocido"), "y": float(y[prueba][i]), "pred": float(ref_pred[i]), "error": float(err[i])}
        col_sospechosa = [j for j, n in enumerate(nombres) if n.endswith(VARIABLE_SOSPECHOSA)]
        f["aporte_variable_sospechosa"] = float(contrib[i, col_sospechosa[0]]) if col_sospechosa else np.nan
        f["z_variable_sospechosa"] = float(zt[i, col_sospechosa[0]]) if col_sospechosa else np.nan
        for j, b in enumerate(mejores, 1):
            f[f"aporte_{j}_variable"], f[f"aporte_{j}_z"], f[f"aporte_{j}_coef_por_z"] = nombres[b].split("__", 1)[-1], float(zt[i, b]), float(contrib[i, b])
        filas.append(f)
    return resumen, z, pd.DataFrame(filas)
