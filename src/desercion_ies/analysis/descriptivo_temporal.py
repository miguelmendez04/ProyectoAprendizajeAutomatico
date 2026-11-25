"""Métricas descriptivas de las particiones temporales, juntando prepandemia y estrés (sección de robustez del paper).

Se calculan a partir de `temporal_predicciones.csv`, por configuración y variante, sobre la unión de las filas de prueba de las dos particiones.
Son **descriptivas**: no traen intervalos ni pruebas de significancia, y juntan dos modelos ajustados con entrenamientos distintos.

- MAE y error absoluto mediano de |predicción − tasa observada|.
- R² = 1 − SCE / SCT, con la media observada de las filas juntas.
- % de filas con |error| ≤ 0,02 y ≤ 0,05 (2 y 5 puntos porcentuales).
- Spearman entre predicción y tasa observada.
- Precisión en el 25 % de mayor riesgo: de las k = redondeo(0,25 · n) filas con mayor predicción, qué fracción está entre las k de mayor tasa
  observada. Los empates en el corte de la predicción (frecuentes en las líneas base) cuentan en proporción: es el valor esperado al desempatar
  al azar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

UMBRALES = (0.02, 0.05)
FRACCION_RIESGO = 0.25


def _pesos_mayores(valores: np.ndarray, k: int) -> np.ndarray:
    """Peso de cada fila en el grupo de las k mayores: 1 dentro, 0 fuera y la proporción que cabe para los empatados en el corte."""
    corte = np.sort(valores)[::-1][k - 1]
    arriba, empate = valores > corte, valores == corte
    pesos = arriba.astype(float)
    pesos[empate] = (k - arriba.sum()) / empate.sum()
    return pesos


def precision_mayor_riesgo(y, pred, fraccion: float = FRACCION_RIESGO) -> float:
    """Fracción de las filas señaladas como de mayor riesgo que están de verdad entre las de mayor tasa observada."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    k = max(1, int(round(fraccion * len(y))))
    return float((_pesos_mayores(pred, k) * _pesos_mayores(y, k)).sum() / k)


def metricas(y, pred) -> dict:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    error = np.abs(pred - y)
    return {
        "n": len(y),
        "mae": float(error.mean()),
        "error_absoluto_mediano": float(np.median(error)),
        "r2": float(1 - ((pred - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
        # la tolerancia evita que el redondeo binario saque una fila con error exactamente igual al umbral
        **{
            f"pct_error_hasta_{int(u * 100)}pp": float(100 * (error <= u + 1e-12).mean())
            for u in UMBRALES
        },
        "spearman": float(pd.Series(pred).corr(pd.Series(y), method="spearman")),
        "precision_25_mayor_riesgo": precision_mayor_riesgo(y, pred),
    }


def descriptivo_temporal(pred: pd.DataFrame) -> pd.DataFrame:
    """Una fila por tarea × configuración × variante, con las particiones temporales juntas."""
    filas = []
    for (tarea, config, variante), g in pred.groupby(
        ["tarea", "configuracion", "variante"], sort=False
    ):
        if g.duplicated(["cod_ies", "anio"]).any():
            raise ValueError(f"{config} ({variante}): hay IES × año repetidos entre particiones")
        filas.append(
            {
                "tarea": tarea,
                "configuracion": config,
                "variante": variante,
                "particiones": "+".join(sorted(g["particion"].unique())),
                **metricas(g["y"], g["pred"]),
            }
        )
    return pd.DataFrame(filas)
