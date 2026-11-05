"""Carga y validación de la configuración del ETL."""
from pathlib import Path

import yaml

UNIDADES_VALIDAS = {"ies", "programa", "municipio"}


def load_config(path: str | Path = "config/config.yaml") -> dict:
    """Lee el YAML de configuración y valida la unidad de análisis."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg.get("unidad_analisis") not in UNIDADES_VALIDAS:
        raise ValueError(
            f"unidad_analisis inválida: {cfg.get('unidad_analisis')!r}; "
            f"debe ser una de {sorted(UNIDADES_VALIDAS)}"
        )
    return cfg
