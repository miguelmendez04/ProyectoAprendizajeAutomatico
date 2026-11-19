"""Corre el protocolo de evaluación v3 (GroupKFold repetido, IC 95 %, prueba pareada contra la ingenua).

Uso: desercion evaluar tarea1 [--config RUTA]
Salidas (rutas en config.yaml: modelado.protocolo_v3.salidas): reports/tables/experimentos_v3.csv,
reports/tables/experimentos_v3_resumen.csv, reports/02_tarea1.md y reports/figures/mae_v3_ic.png.
"""

import argparse
import logging
import time
from pathlib import Path

import sklearn
import xgboost

from desercion_ies.config import load_config
from desercion_ies.evaluation.protocolo import ejecutar_protocolo, parametros_protocolo, resumen
from desercion_ies.features.particion import cargar_dataset
from desercion_ies.reporting.informe_v3 import escribir_informe, figura_ic

SALIDAS = {
    "resultados": "reports/tables/experimentos_v3.csv",
    "resumen": "reports/tables/experimentos_v3_resumen.csv",
    "informe": "reports/02_tarea1.md",
    "figura": "reports/figures/mae_v3_ic.png",
}


log = logging.getLogger(__name__)


def configurar(ap: argparse.ArgumentParser) -> None:
    """Opciones de la línea de comandos de esta etapa."""
    ap.add_argument("--config", default="configs/config.yaml", help="archivo de configuración")


def ejecutar(args: argparse.Namespace) -> None:
    """Corre la etapa con las opciones ya leídas."""
    t0 = time.perf_counter()
    cfg = load_config(args.config)
    p = parametros_protocolo(cfg)
    df = cargar_dataset(cfg)
    resultados = ejecutar_protocolo(cfg, df, verbose=True)
    res = resumen(resultados, p["nivel_confianza"])
    total = time.perf_counter() - t0

    salidas = {k: Path(v) for k, v in {**SALIDAS, **p.get("salidas", {})}.items()}
    salidas["resultados"].parent.mkdir(parents=True, exist_ok=True)
    resultados.to_csv(salidas["resultados"], index=False)
    res.to_csv(salidas["resumen"], index=False)
    figura_ic(res, salidas["figura"])
    meta = {
        "filas": len(df),
        "ies": int(df["cod_ies"].nunique()),
        "ventana": cfg["modelado"].get("ventana_modelado", [2015, 2021]),
        "n_splits": p["n_splits"],
        "semillas": p["semillas"],
        "n_ajustes": p["n_splits"] * len(p["semillas"]),
        "n_configuraciones": len(p["configuraciones"]),
        "sklearn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "tiempo_total_s": total,
    }
    escribir_informe(resultados, res, meta, salidas["informe"])

    cols = ["configuracion", "mae_media", "mae_ic_inf", "mae_ic_sup"]
    log.info("\nLíneas base:\n" + res[res["tipo"] == "linea_base"][cols].to_string(index=False))
    log.info(
        "\nConfiguraciones:\n"
        + res[res["tipo"] == "modelo"]
        .sort_values("mae_media")[
            cols + ["mejora_ing_pct", "p_wilcoxon_ing", "mejora_rez_pct", "p_wilcoxon_rez"]
        ]
        .to_string(index=False)
    )
    log.info(f"\nTiempo total: {total:.1f} s")
