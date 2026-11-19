"""Corre las formulaciones con historia y corrección sobre la ingenua; escribe resultados, resumen, informe y figura.

Uso: desercion evaluar v2 [--config RUTA]
Salidas: reports/tables/experimentos_v2.csv, reports/tables/experimentos_v2_resumen.csv, reports/historico/resultados_v2.md
y reports/figures/mae_formulaciones.png (rutas en config.yaml: modelado.formulaciones.salidas).
"""

import argparse
import logging
import time
from pathlib import Path

import sklearn
import xgboost

from desercion_ies.config import load_config
from desercion_ies.evaluation.experimentos import parametros_experimentos
from desercion_ies.evaluation.formulaciones import (
    FORMULACIONES,
    ejecutar_formulaciones,
    parametros_formulaciones,
    resumen,
    variables_formulacion,
)
from desercion_ies.features.particion import cargar_dataset
from desercion_ies.reporting.informe_formulaciones import escribir_informe, figura_formulaciones

SALIDAS = {
    "resultados": "reports/tables/experimentos_v2.csv",
    "resumen": "reports/tables/experimentos_v2_resumen.csv",
    "informe": "reports/historico/resultados_v2.md",
    "figura": "reports/figures/mae_formulaciones.png",
}


log = logging.getLogger(__name__)


def configurar(ap: argparse.ArgumentParser) -> None:
    """Opciones de la línea de comandos de esta etapa."""
    ap.add_argument("--config", default="configs/config.yaml", help="archivo de configuración")


def ejecutar(args: argparse.Namespace) -> None:
    """Corre la etapa con las opciones ya leídas."""
    t0 = time.perf_counter()
    cfg = load_config(args.config)
    p_exp, p = parametros_experimentos(cfg), parametros_formulaciones(cfg)
    df = cargar_dataset(cfg)
    resultados = ejecutar_formulaciones(cfg, df, verbose=True)
    res = resumen(resultados)
    total = time.perf_counter() - t0

    salidas = {k: Path(v) for k, v in {**SALIDAS, **p.get("salidas", {})}.items()}
    salidas["resultados"].parent.mkdir(parents=True, exist_ok=True)
    resultados.to_csv(salidas["resultados"], index=False)
    res.to_csv(salidas["resumen"], index=False)
    figura_formulaciones(res, salidas["figura"])
    numericas, categoricas = variables_formulacion(df, cfg, "contexto")
    meta = {
        "filas": len(df),
        "ies": int(df["cod_ies"].nunique()),
        "ventana": cfg["modelado"].get("ventana_modelado", [2015, 2021]),
        "n_pliegues": int(resultados["pliegue"].nunique()),
        "n_contexto": len(numericas) + len(categoricas),
        "k_info_mutua": p_exp["k_info_mutua"],
        "varianza_pca": p_exp["varianza_pca"],
        "semilla": p_exp["semilla"],
        "sklearn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "tiempo_total_s": total,
    }
    escribir_informe(resultados, res, meta, salidas["informe"])

    cols = [
        "configuracion",
        "mae_media",
        "mae_desv",
        "mejora_vs_ingenua_pct",
        "pliegues_mejor_que_ingenua",
    ]
    log.info("\nLíneas base:\n" + res[res["tipo"] == "linea_base"][cols].to_string(index=False))
    for f in FORMULACIONES:
        g = res[res["formulacion"] == f]
        if len(g):
            log.info(f"\nMejor de {f}:\n" + g.head(1)[cols].to_string(index=False))
    log.info(f"\nTiempo total: {total:.1f} s")
