"""Transformación opcional del objetivo para el modelado.

La tasa de deserción está acotada en [0, 1] y es muy asimétrica; el logit la lleva a toda la recta real y
comprime la cola derecha. Se elige en `config.yaml: modelado.transformacion_objetivo` (`ninguna` | `logit`).
Las predicciones del modelo hay que devolverlas a la escala de tasa con `invertir_objetivo` antes de evaluar.
"""
import numpy as np
import pandas as pd

METODOS = {"ninguna", "logit"}


def _validar(metodo: str) -> None:
    if metodo not in METODOS:
        raise ValueError(f"modelado.transformacion_objetivo inválida: {metodo!r}; debe ser una de {sorted(METODOS)}")


def transformar_objetivo(y: pd.Series, metodo: str = "ninguna", epsilon: float = 0.001) -> pd.Series:
    """`ninguna` devuelve la tasa; `logit` devuelve log(p / (1 - p)) con p recortada a [epsilon, 1 - epsilon]."""
    _validar(metodo)
    if metodo == "ninguna":
        return y
    p = y.clip(epsilon, 1 - epsilon)
    return np.log(p / (1 - p))


def invertir_objetivo(z: pd.Series, metodo: str = "ninguna") -> pd.Series:
    """Devuelve la escala de tasa (0-1) a partir del objetivo transformado (o de predicciones en esa escala)."""
    _validar(metodo)
    if metodo == "ninguna":
        return z
    return 1 / (1 + np.exp(-z))


def objetivo_para_modelado(df: pd.DataFrame, cfg: dict, objetivo: str = "tasa_desercion") -> pd.Series:
    """El objetivo transformado según `config.yaml: modelado` (sin sección, se deja sin transformar)."""
    m = cfg.get("modelado", {})
    return transformar_objetivo(df[objetivo], m.get("transformacion_objetivo", "ninguna"), m.get("epsilon_logit", 0.001))
