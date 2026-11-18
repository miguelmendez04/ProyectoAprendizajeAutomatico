"""Corre la grilla de experimentos y escribe resultados, resumen, informe y figura.

Uso: python run_experimentos.py [--config RUTA]
Salidas (rutas en config.yaml: modelado.experimentos.salidas): reports/resultados/experimentos_v1.csv,
reports/resultados/experimentos_v1_resumen.csv, reports/resultados_v1.md y reports/figuras/mae_configuraciones.png.
"""
import argparse
import time
from pathlib import Path

import sklearn
import xgboost

from src.config import load_config
from src.modelado.experimentos import (
    BASE_INGENUA, BASE_MEDIANA, ejecutar_experimentos, parametros_experimentos, preparar_datos, resumen,
)
from src.modelado.informe_experimentos import escribir_informe, figura_mae


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()
    t0 = time.perf_counter()
    cfg = load_config(args.config)
    p = parametros_experimentos(cfg)
    df, _, _, numericas, categoricas = preparar_datos(cfg)
    resultados = ejecutar_experimentos(cfg, df, verbose=True)
    res = resumen(resultados)
    total = time.perf_counter() - t0

    salidas = {k: Path(v) for k, v in {**{"resultados": "reports/resultados/experimentos_v1.csv", "resumen": "reports/resultados/experimentos_v1_resumen.csv",
                                          "informe": "reports/resultados_v1.md", "figura": "reports/figuras/mae_configuraciones.png"},
                                       **p.get("salidas", {})}.items()}
    salidas["resultados"].parent.mkdir(parents=True, exist_ok=True)
    resultados.to_csv(salidas["resultados"], index=False)
    res.to_csv(salidas["resumen"], index=False)
    figura_mae(res, salidas["figura"])
    meta = {
        "filas": len(df), "ies": int(df["cod_ies"].nunique()), "ventana": cfg["modelado"].get("ventana_modelado", [2015, 2021]),
        "n_numericas": len(numericas), "n_categoricas": len(categoricas), "n_pliegues": int(resultados["pliegue"].nunique()),
        "k_vecinos": p["k_vecinos"], "k_info_mutua": p["k_info_mutua"], "varianza_pca": p["varianza_pca"],
        "n_estimators_rf": p["random_forest"]["n_estimators"], "n_estimators_xgb": p["xgboost"]["n_estimators"],
        "lr_xgb": p["xgboost"]["learning_rate"], "depth_xgb": p["xgboost"]["max_depth"], "semilla": p["semilla"],
        "sklearn": sklearn.__version__, "xgboost": xgboost.__version__, "tiempo_total_s": total,
    }
    escribir_informe(resultados, res, meta, salidas["informe"])

    cols = ["configuracion", "mae_media", "mae_desv", "mejora_vs_ingenua_pct", "pliegues_mejor_que_ingenua"]
    print("\nLíneas base:\n" + res[res["tipo"] == "linea_base"][cols].to_string(index=False))
    print("\nTop 5 por MAE:\n" + res[res["tipo"] == "modelo"].head(5)[cols].to_string(index=False))
    print(f"\nTiempo total: {total:.1f} s")


if __name__ == "__main__":
    main()
