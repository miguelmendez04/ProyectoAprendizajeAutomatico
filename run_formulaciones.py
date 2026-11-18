"""Corre las formulaciones con historia y corrección sobre la ingenua; escribe resultados, resumen, informe y figura.

Uso: python run_formulaciones.py [--config RUTA]
Salidas: reports/resultados/experimentos_v2.csv, reports/resultados/experimentos_v2_resumen.csv, reports/resultados_v2.md
y reports/figuras/mae_formulaciones.png (rutas en config.yaml: modelado.formulaciones.salidas).
"""
import argparse
import time
from pathlib import Path

import sklearn
import xgboost

from src.config import load_config
from src.modelado.experimentos import parametros_experimentos
from src.modelado.formulaciones import (
    FORMULACIONES, ejecutar_formulaciones, parametros_formulaciones, resumen, variables_formulacion,
)
from src.modelado.informe_formulaciones import escribir_informe, figura_formulaciones
from src.modelado.particion import cargar_dataset

SALIDAS = {"resultados": "reports/resultados/experimentos_v2.csv", "resumen": "reports/resultados/experimentos_v2_resumen.csv",
           "informe": "reports/resultados_v2.md", "figura": "reports/figuras/mae_formulaciones.png"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()
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
        "filas": len(df), "ies": int(df["cod_ies"].nunique()), "ventana": cfg["modelado"].get("ventana_modelado", [2015, 2021]),
        "n_pliegues": int(resultados["pliegue"].nunique()), "n_contexto": len(numericas) + len(categoricas),
        "k_info_mutua": p_exp["k_info_mutua"], "varianza_pca": p_exp["varianza_pca"], "semilla": p_exp["semilla"],
        "sklearn": sklearn.__version__, "xgboost": xgboost.__version__, "tiempo_total_s": total,
    }
    escribir_informe(resultados, res, meta, salidas["informe"])

    cols = ["configuracion", "mae_media", "mae_desv", "mejora_vs_ingenua_pct", "pliegues_mejor_que_ingenua"]
    print("\nLíneas base:\n" + res[res["tipo"] == "linea_base"][cols].to_string(index=False))
    for f in FORMULACIONES:
        g = res[res["formulacion"] == f]
        if len(g):
            print(f"\nMejor de {f}:\n" + g.head(1)[cols].to_string(index=False))
    print(f"\nTiempo total: {total:.1f} s")


if __name__ == "__main__":
    main()
