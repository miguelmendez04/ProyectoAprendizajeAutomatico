"""Error por subgrupo de los modelos finales frente a la línea base de su tarea, con IC 95 % por bootstrap sobre IES.

Subgrupos: sector, carácter, región (Bogotá, Medellín, Cali y resto), `es_sena`, tamaño (terciles de `log_matriculados_spadies_t2`) y `falta_historia`.

**Error de cada fila:** el promedio de sus predicciones fuera de muestra, una por cada repetición del GroupKFold por IES 5 × 5 (semillas 301–305).
**IC:** se remuestrean IES completas con reemplazo (1.000 réplicas, semilla fija); el mismo remuestreo se usa para el modelo y su línea base, de modo que
la diferencia (error de la línea base − error del modelo; positiva = el modelo es mejor) es pareada. Es la incertidumbre por la muestra de IES, que es la
relevante para un subgrupo; no incluye la variabilidad por la elección de pliegues.
"""

import numpy as np
import pandas as pd

from desercion_ies.evaluation.evaluacion_final import BASES
from desercion_ies.features.particion import ETIQUETAS_CARACTER, ETIQUETAS_SECTOR

VARIABLES = ["sector", "caracter", "region", "es_sena", "tamano", "falta_historia"]
ETIQUETA_VARIABLE = {
    "sector": "Sector",
    "caracter": "Carácter",
    "region": "Región",
    "es_sena": "SENA",
    "tamano": "Tamaño (terciles de la matrícula t-2)",
    "falta_historia": "Historia",
}
N_PEQUENO = 30  # por debajo de este número de filas el IC se marca como poco informativo
MIN_IES = 5  # con menos IES distintas el bootstrap sobre IES no da un IC válido (el subgrupo es una sola institución o casi)


def asignar_subgrupos(
    df: pd.DataFrame, ciudades: dict[str, str]
) -> tuple[pd.DataFrame, list[float]]:
    """(subgrupo de cada fila por variable, cortes de los terciles en matrícula SPADIES t-2)."""
    t, cortes = pd.qcut(
        df["log_matriculados_spadies_t2"],
        3,
        labels=["tamaño bajo", "tamaño medio", "tamaño alto"],
        retbins=True,
    )
    sub = pd.DataFrame(
        {
            "sector": df["sector"].astype(str).map(ETIQUETAS_SECTOR).fillna("desconocido"),
            "caracter": df["caracter"].astype(str).map(ETIQUETAS_CARACTER).fillna("desconocido"),
            "region": df["cod_divipola_mpio"].astype(str).map(ciudades).fillna("resto"),
            "es_sena": np.where(df["es_sena"].astype(bool), "SENA", "no SENA"),
            "tamano": t.astype(str),
            "falta_historia": np.where(
                df["falta_historia"].astype(bool), "sin historia", "con historia"
            ),
        },
        index=df.index,
    )
    return sub, [float(np.expm1(c)) for c in cortes]


def _clasificar(dif: float, lo: float, hi: float) -> str:
    if hi < 0:
        return "peor que la línea base (IC excluye 0)"
    if dif < 0:
        return "peor (sin firmeza)"
    if lo > 0:
        return "mejor (IC excluye 0)"
    return "mejor (sin firmeza)"


def analizar_subgrupos(
    df: pd.DataFrame,
    sub: pd.DataFrame,
    oof: dict[str, np.ndarray],
    modelos: list[dict],
    replicas: int = 1000,
    semilla: int = 2026,
    nivel: float = 0.95,
) -> pd.DataFrame:
    """MAE por subgrupo de cada modelo y de su línea base, y la diferencia con IC; marca los subgrupos donde el modelo es peor."""
    y = df["tasa_desercion"].to_numpy(dtype=float)
    ies_codigos, ies = np.unique(df["cod_ies"].astype(str).to_numpy(), return_inverse=True)
    n_ies = len(ies_codigos)
    pesos = (
        np.random.default_rng(semilla)
        .multinomial(n_ies, np.full(n_ies, 1.0 / n_ies), size=replicas)
        .astype(float)
    )  # (B, n_ies)
    q = [100 * (1 - nivel) / 2, 100 * (1 + nivel) / 2]

    def ic(boot):
        v = boot[~np.isnan(boot)]
        return (
            (float(np.percentile(v, q[0])), float(np.percentile(v, q[1])))
            if len(v)
            else (np.nan, np.nan)
        )

    filas = []
    for m in modelos:
        base = BASES[m["tarea"]]
        e_m = np.abs(oof[m["nombre"]] - y).mean(
            axis=0
        )  # error de cada fila: promedio de sus predicciones fuera de muestra
        e_b = np.abs(oof[base] - y).mean(axis=0)
        for variable in VARIABLES:
            for nivel_ in dict.fromkeys(sub[variable]):
                mask = (sub[variable] == nivel_).to_numpy()
                n_j = np.bincount(ies[mask], minlength=n_ies).astype(float)
                s_m, s_b = (
                    np.bincount(ies[mask], weights=e[mask], minlength=n_ies) for e in (e_m, e_b)
                )
                den = pesos @ n_j
                with np.errstate(invalid="ignore", divide="ignore"):
                    bm, bb = (
                        np.where(den > 0, (pesos @ s_m) / den, np.nan),
                        np.where(den > 0, (pesos @ s_b) / den, np.nan),
                    )
                bd = bb - bm
                mae_m, mae_b = s_m.sum() / n_j.sum(), s_b.sum() / n_j.sum()
                dif = mae_b - mae_m
                n_ies_nivel = int((n_j > 0).sum())
                evaluable = n_ies_nivel >= MIN_IES
                lo, hi = ic(bd) if evaluable else (np.nan, np.nan)
                ci_m, ci_b = (ic(bm), ic(bb)) if evaluable else ((np.nan, np.nan), (np.nan, np.nan))
                filas.append(
                    {
                        "tarea": m["tarea"],
                        "modelo": m["nombre"],
                        "linea_base": base,
                        "variable": variable,
                        "nivel": nivel_,
                        "n_filas": int(mask.sum()),
                        "n_ies": n_ies_nivel,
                        "mae_modelo": float(mae_m),
                        "mae_modelo_ic_inf": ci_m[0],
                        "mae_modelo_ic_sup": ci_m[1],
                        "mae_base": float(mae_b),
                        "mae_base_ic_inf": ci_b[0],
                        "mae_base_ic_sup": ci_b[1],
                        "dif": float(dif),
                        "dif_ic_inf": lo,
                        "dif_ic_sup": hi,
                        "prop_gana": float(np.nanmean(bd > 0)) if evaluable else np.nan,
                        "n_pequeno": bool(mask.sum() < N_PEQUENO or not evaluable),
                        "clasificacion": _clasificar(dif, lo, hi)
                        if evaluable
                        else f"no evaluable (menos de {MIN_IES} IES)",
                    }
                )
    return pd.DataFrame(filas)
