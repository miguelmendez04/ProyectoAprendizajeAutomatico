"""Arranque en frío estricto (Tarea 2): pliegues agrupados por municipio sede en lugar de por IES.

La confirmación de la Tarea 2 agrupa por `cod_ies`; en ese esquema las demás IES de la misma ciudad están en el entrenamiento y comparten
todo el contexto territorial. Aquí se repite la confirmación (5 × 5, mismas configuraciones y semillas) con GroupKFold por municipio
(`cod_divipola_mpio`): **ninguna ciudad de prueba aparece en el entrenamiento**. Bogotá concentra una parte grande de las filas, así que
los pliegues salen desbalanceados; `tamanos_pliegues` los describe.
"""

import pandas as pd

from desercion_ies.evaluation import tarea2 as t2
from desercion_ies.evaluation.protocolo import particiones_repetidas

GRUPO = "cod_divipola_mpio"
BOGOTA = "11001"


def configuraciones_confirmacion(cfg: dict, archivo: str | None = None) -> list[dict]:
    """Las configuraciones de la confirmación de la Tarea 2 (las mismas del `experimentos_tarea2_confirmacion_resumen.csv`), sin las líneas base."""
    ruta = archivo or cfg["modelado"]["ajuste_temporal"]["confirmacion_tarea2"]
    r = pd.read_csv(ruta)
    r = r[r["tipo"] == "modelo"]
    return [
        {
            "objetivo": f["objetivo"],
            "imputacion": f["imputacion"],
            "reduccion": f["reduccion"],
            "clusters": str(f["clusters"]) == "True",
            "modelo": f["modelo"],
        }
        for _, f in r.iterrows()
    ]


def tamanos_pliegues(df: pd.DataFrame, repeticiones, semillas, grupo: str = GRUPO) -> pd.DataFrame:
    """Tamaño de cada pliegue de prueba: filas, % del total, municipios, IES, y si contiene a Bogotá (y cuántas filas)."""
    filas = []
    for rep, (pliegues, semilla) in enumerate(zip(repeticiones, semillas, strict=False)):
        for k, (entrena, prueba) in enumerate(pliegues):
            t = df.iloc[prueba]
            bog = int((t[grupo].astype(str) == BOGOTA).sum())
            filas.append(
                {
                    "repeticion": rep,
                    "semilla": int(semilla),
                    "pliegue": k,
                    "n_prueba": len(prueba),
                    "pct_prueba": 100 * len(prueba) / len(df),
                    "n_entrenamiento": len(entrena),
                    "n_municipios_prueba": int(t[grupo].nunique()),
                    "n_ies_prueba": int(t["cod_ies"].nunique()),
                    "contiene_bogota": bog > 0,
                    "filas_bogota": bog,
                }
            )
    return pd.DataFrame(filas)


def ejecutar_arranque_frio(
    cfg: dict, df: pd.DataFrame | None = None, verbose: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(resultados por configuración y pliegue, tamaños de los pliegues) con GroupKFold por municipio y las semillas de la confirmación."""
    p = t2.parametros_tarea2(cfg)
    df = t2.preparar_datos(cfg, df)[0]
    semillas = list(p["semillas_confirmacion"])
    res = t2.evaluar(
        cfg,
        configuraciones_confirmacion(cfg),
        "arranque_frio_municipio",
        semillas,
        df,
        verbose,
        grupo=GRUPO,
    )
    return res, tamanos_pliegues(
        df, particiones_repetidas(df, p["n_splits"], semillas, GRUPO), semillas
    )
