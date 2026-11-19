"""Particiones temporales: ¿la conclusión se sostiene al predecir años futuros y en pandemia?

Particiones (`config.yaml: modelado.ajuste_temporal.particiones_temporales`):
- **prepandemia:** entrena 2015–2017, prueba 2018–2019.
- **estrés:** entrena 2015–2019, prueba 2020–2021.

Para cada partición y cada configuración se ajusta con la búsqueda interna de `ajuste.py` (RandomizedSearchCV de 30 iteraciones, GroupKFold por
IES de 3 pliegues) **solo sobre las filas de los años de entrenamiento**, y se mide en los años de prueba. También se mide la versión por
defecto. Ningún año de prueba entra al entrenamiento ni a la búsqueda.

**Tarea 1 es un pronóstico a un paso:** para predecir el año t se usa la tasa observada de t-1 (`tasa_t1`, `tasa_t2`, `tasa_media_previa`, ...),
aunque t-1 esté en el período de prueba (en la práctica está disponible cuando se predice t); esas columnas salen de los datos observados.
Lo único que se oculta al entrenar son las filas (y objetivos) de los años de prueba. La ingenua (tasa de t-1) y la corrección parten del mismo dato.
En la Tarea 2 las variables son de contexto; la mediana por carácter × sector se calcula con los años de entrenamiento.

**Incertidumbre:** IC 95 % por bootstrap sobre IES (se remuestrean IES completas con reemplazo, 1.000 réplicas, semilla fija), con el mismo remuestreo
para todas las configuraciones de una partición y tarea, de modo que las diferencias con la línea base son pareadas.
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from desercion_ies.evaluation import ajuste as aj
from desercion_ies.evaluation import tarea2 as t2
from desercion_ies.evaluation.experimentos import parametros_experimentos
from desercion_ies.evaluation.protocolo import prediccion_base
from desercion_ies.features.particion import ETIQUETAS_CARACTER, ETIQUETAS_SECTOR, cargar_dataset


def particiones(cfg: dict) -> dict[str, dict]:
    pt = cfg["modelado"]["ajuste_temporal"]["particiones_temporales"]
    return {
        nombre: {"entrena": tuple(v["entrena"]), "prueba": tuple(v["prueba"])}
        for nombre, v in pt.items()
    }


def indices_temporales(
    df: pd.DataFrame, entrena: tuple[int, int], prueba: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """(entrenamiento, prueba) posicionales: años dentro de cada rango (inclusive). Los rangos no se pueden traslapar."""
    if entrena[1] >= prueba[0]:
        raise ValueError(
            f"los años de prueba {prueba} deben ser posteriores a los de entrenamiento {entrena}"
        )
    e = np.flatnonzero(df["anio"].between(*entrena).to_numpy())
    t = np.flatnonzero(df["anio"].between(*prueba).to_numpy())
    if len(e) == 0 or len(t) == 0:
        raise ValueError("la partición temporal deja vacío el entrenamiento o la prueba")
    return e, t


def _tarea(
    spec, nombre, entrena, prueba, X, y, grupos, numericas, categoricas, p_exp, p_t2, p
) -> dict:
    d, a = aj.ajustar_y_evaluar(
        spec, X, y, grupos, entrena, prueba, numericas, categoricas, p_exp, p_t2, p
    )
    return {"particion": nombre, "spec": spec, "defecto": d, "ajustado": a}


def _filas_prediccion(
    df: pd.DataFrame,
    prueba: np.ndarray,
    particion: str,
    tarea: str,
    configuracion: str,
    variante: str,
    pred: np.ndarray,
    extra: dict,
) -> pd.DataFrame:
    t = df.iloc[prueba]
    return pd.DataFrame(
        {
            "particion": particion,
            "tarea": tarea,
            "configuracion": configuracion,
            "variante": variante,
            "cod_ies": t["cod_ies"].astype(str).to_numpy(),
            "anio": t["anio"].to_numpy(),
            "cod_divipola_mpio": t["cod_divipola_mpio"].astype(str).to_numpy(),
            "sector": t["sector"]
            .astype(str)
            .map(ETIQUETAS_SECTOR)
            .fillna("desconocido")
            .to_numpy(),
            "caracter": t["caracter"]
            .astype(str)
            .map(ETIQUETAS_CARACTER)
            .fillna("desconocido")
            .to_numpy(),
            "y": t["tasa_desercion"].to_numpy(dtype=float),
            "pred": pred,
            **extra,
        }
    )


def ejecutar_temporal(
    cfg: dict,
    df: pd.DataFrame | None = None,
    specs: list[dict] | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Predicciones por fila de prueba de cada partición, configuración, variante (defecto / ajustado) y línea base, con tiempos e hiperparámetros."""
    p, p_exp, p_t2 = (
        aj.parametros_ajuste(cfg),
        parametros_experimentos(cfg),
        t2.parametros_tarea2(cfg),
    )
    df = cargar_dataset(cfg) if df is None else df
    y = df["tasa_desercion"].astype(float)
    specs = aj.configuraciones_ajuste(cfg) if specs is None else specs
    datos = {s["nombre"]: aj.preparar(s, cfg, df) for s in specs}
    grupos = df["cod_ies"].astype(str).to_numpy()
    tareas, bases = [], []
    for nombre, rango in particiones(cfg).items():
        entrena, prueba = indices_temporales(df, rango["entrena"], rango["prueba"])
        med = np.full(len(prueba), float(np.median(y.iloc[entrena])))
        grupo, _ = t2.prediccion_mediana_grupo(df, y, entrena, prueba)
        for tarea, base, pred in (
            ("T1", aj.BASE_INGENUA, prediccion_base(df, y, entrena, prueba, "tasa_t1")),
            ("T1", aj.BASE_MEDIANA, med),
            ("T2", aj.BASE_GRUPO, grupo),
            ("T2", aj.BASE_MEDIANA, med),
        ):
            bases.append(
                _filas_prediccion(
                    df,
                    prueba,
                    nombre,
                    tarea,
                    f"{tarea} {base}",
                    "linea_base",
                    pred,
                    {
                        "t_ajuste_s": 0.0,
                        "t_prediccion_s": 0.0,
                        "t_busqueda_s": np.nan,
                        "hiperparametros": "{}",
                        "n_entrenamiento": len(entrena),
                    },
                )
            )
        tareas += [(s, nombre, entrena, prueba) for s in specs]
    salida = Parallel(n_jobs=p["n_jobs"], batch_size=1, verbose=10 if verbose else 0)(
        delayed(_tarea)(
            s,
            nombre,
            entrena,
            prueba,
            datos[s["nombre"]][0],
            y,
            grupos,
            datos[s["nombre"]][1],
            datos[s["nombre"]][2],
            p_exp,
            p_t2,
            p,
        )
        for s, nombre, entrena, prueba in tareas
    )
    por_tarea = {(s["nombre"], nombre): (entrena, prueba) for s, nombre, entrena, prueba in tareas}
    pred = list(bases)
    for r in salida:
        s, nombre = r["spec"], r["particion"]
        entrena, prueba = por_tarea[(s["nombre"], nombre)]
        for variante in ("defecto", "ajustado"):
            v = r[variante]
            pred.append(
                _filas_prediccion(
                    df,
                    prueba,
                    nombre,
                    s["tarea"],
                    s["nombre"],
                    variante,
                    v["pred"],
                    {
                        "t_ajuste_s": v["t_ajuste_s"],
                        "t_prediccion_s": v["t_prediccion_s"],
                        "t_busqueda_s": v.get("t_busqueda_s", np.nan),
                        "hiperparametros": v.get("hiperparametros", "{}"),
                        "n_entrenamiento": len(entrena),
                    },
                )
            )
    return pd.concat(pred, ignore_index=True)


# ---------- bootstrap sobre IES ----------
def _claves(g: pd.DataFrame) -> pd.Series:
    return g["configuracion"] + np.where(
        g["variante"] == "linea_base", "", " [" + g["variante"] + "]"
    )


def bootstrap_ies(
    pred: pd.DataFrame,
    replicas: int = 1000,
    semilla: int = 2026,
    nivel: float = 0.95,
    referencia: str | None = None,
) -> pd.DataFrame:
    """Métricas (MAE, RMSE, R²) con IC por bootstrap sobre IES para las filas de **una** partición y tarea.

    Se remuestrean IES completas con reemplazo (el mismo remuestreo para todas las configuraciones). Para cada configuración distinta de la
    `referencia` se calcula también la diferencia de MAE (referencia − configuración; positiva = la configuración tiene menos error) con su IC
    y la proporción de réplicas en que gana.
    """
    pred = pred.assign(clave=_claves(pred))
    ies = np.sort(pred["cod_ies"].unique())
    rng = np.random.default_rng(semilla)
    pesos = rng.multinomial(len(ies), np.full(len(ies), 1.0 / len(ies)), size=replicas).astype(
        float
    )  # (B, n_ies): veces que entra cada IES
    una = pred[pred["clave"] == pred["clave"].iloc[0]].set_index("cod_ies")
    n = una.groupby(level=0)["y"].size().reindex(ies).to_numpy(float)
    sy = una.groupby(level=0)["y"].sum().reindex(ies).to_numpy(float)
    syy = (una["y"] ** 2).groupby(level=0).sum().reindex(ies).to_numpy(float)
    q = [100 * (1 - nivel) / 2, 100 * (1 + nivel) / 2]
    filas, maes = [], {}
    for clave, g in pred.groupby("clave", sort=False):
        e = (g["pred"] - g["y"]).to_numpy()
        gi = (
            pd.DataFrame({"cod_ies": g["cod_ies"].to_numpy(), "sae": np.abs(e), "sse": e**2})
            .groupby("cod_ies")
            .sum()
            .reindex(ies)
        )
        sae, sse = gi["sae"].to_numpy(float), gi["sse"].to_numpy(float)

        def metricas_(w):
            nn = w @ n
            sst = w @ syy - (w @ sy) ** 2 / nn
            return (w @ sae) / nn, np.sqrt((w @ sse) / nn), 1 - (w @ sse) / sst  # noqa: B023  (se usa dentro de la misma iteración)

        punto = metricas_(np.ones(len(ies)))
        boot = metricas_(pesos)  # (B,) por métrica: las réplicas se calculan vectorizadas
        maes[clave] = (punto[0], boot[0])
        f = {
            "configuracion": g["configuracion"].iloc[0],
            "variante": g["variante"].iloc[0],
            "n_filas": len(g),
            "n_ies": len(ies),
        }
        for nombre, v, b in zip(("mae", "rmse", "r2"), punto, boot, strict=False):
            lo, hi = np.percentile(b, q)
            f.update(
                {nombre: float(v), f"{nombre}_ic_inf": float(lo), f"{nombre}_ic_sup": float(hi)}
            )
        for col in ("t_ajuste_s", "t_prediccion_s", "t_busqueda_s", "n_entrenamiento"):
            f[col] = float(g[col].iloc[0]) if col in g else np.nan
        f["hiperparametros"] = g["hiperparametros"].iloc[0] if "hiperparametros" in g else "{}"
        filas.append(f)
    out = pd.DataFrame(filas)
    if referencia is not None:
        ref_p, ref_b = maes[referencia]
        for i, f in out.iterrows():
            clave = (
                f["configuracion"]
                if f["variante"] == "linea_base"
                else f"{f['configuracion']} [{f['variante']}]"
            )
            p_, b_ = maes[clave]
            d = ref_b - b_
            lo, hi = np.percentile(d, q)
            out.loc[i, ["dif_ref", "dif_ref_ic_inf", "dif_ref_ic_sup", "prop_gana_ref"]] = [
                ref_p - p_,
                lo,
                hi,
                float((d > 0).mean()),
            ]
    return out


def resumen_temporal(
    pred: pd.DataFrame, replicas: int = 1000, semilla: int = 2026, nivel: float = 0.95
) -> pd.DataFrame:
    """Bootstrap por partición y tarea; la referencia es la ingenua (Tarea 1) o la mediana por carácter × sector (Tarea 2)."""
    partes = []
    for (particion, tarea), g in pred.groupby(["particion", "tarea"], sort=False):
        ref = f"{tarea} {aj.BASE_INGENUA}" if tarea == "T1" else f"{tarea} {aj.BASE_GRUPO}"
        r = bootstrap_ies(g, replicas, semilla, nivel, ref)
        partes.append(r.assign(particion=particion, tarea=tarea, referencia=ref))
    return pd.concat(partes, ignore_index=True)


def errores_por_grupo(pred: pd.DataFrame) -> pd.DataFrame:
    """MAE y número de filas por sector, carácter, año de prueba y (si hay municipio) Bogotá frente al resto, para cada partición, tarea, configuración y variante."""
    p = pred.assign(error=(pred["pred"] - pred["y"]).abs(), anio=pred["anio"].astype(str))
    variables = ["sector", "caracter", "anio"]
    if (
        "cod_divipola_mpio" in p
    ):  # Bogotá concentra una parte grande de las filas: se separa su error del resto
        p["bogota"] = np.where(p["cod_divipola_mpio"].astype(str) == "11001", "Bogotá", "resto")
        variables.append("bogota")
    filas = []
    for variable in variables:
        g = (
            p.groupby(["particion", "tarea", "configuracion", "variante", variable], sort=False)[
                "error"
            ]
            .agg(["mean", "size"])
            .reset_index()
        )
        filas.append(
            g.rename(columns={variable: "nivel", "mean": "mae", "size": "n"}).assign(
                variable=variable
            )
        )
    return pd.concat(filas, ignore_index=True)[
        ["particion", "tarea", "configuracion", "variante", "variable", "nivel", "n", "mae"]
    ]
