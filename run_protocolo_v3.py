"""Corre el protocolo de evaluación v3 (GroupKFold repetido, IC 95 %, prueba pareada contra la ingenua).

Uso: python run_protocolo_v3.py [--config RUTA]
Salidas (rutas en config.yaml: modelado.protocolo_v3.salidas): reports/resultados/experimentos_v3.csv,
reports/resultados/experimentos_v3_resumen.csv, reports/resultados_v3.md y reports/figuras/mae_v3_ic.png.
"""
import argparse
import time
from pathlib import Path

import sklearn
import xgboost

from src.config import load_config
from src.modelado.informe_v3 import escribir_informe, figura_ic
from src.modelado.particion import cargar_dataset
from src.modelado.protocolo import ejecutar_protocolo, parametros_protocolo, resumen

SALIDAS = {"resultados": "reports/resultados/experimentos_v3.csv", "resumen": "reports/resultados/experimentos_v3_resumen.csv",
           "informe": "reports/resultados_v3.md", "figura": "reports/figuras/mae_v3_ic.png"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()
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
        "filas": len(df), "ies": int(df["cod_ies"].nunique()), "ventana": cfg["modelado"].get("ventana_modelado", [2015, 2021]),
        "n_splits": p["n_splits"], "semillas": p["semillas"], "n_ajustes": p["n_splits"] * len(p["semillas"]),
        "n_configuraciones": len(p["configuraciones"]), "sklearn": sklearn.__version__, "xgboost": xgboost.__version__, "tiempo_total_s": total,
    }
    escribir_informe(resultados, res, meta, salidas["informe"])

    cols = ["configuracion", "mae_media", "mae_ic_inf", "mae_ic_sup"]
    print("\nLíneas base:\n" + res[res["tipo"] == "linea_base"][cols].to_string(index=False))
    print("\nConfiguraciones:\n" + res[res["tipo"] == "modelo"].sort_values("mae_media")[cols + ["mejora_ing_pct", "p_wilcoxon_ing", "mejora_rez_pct", "p_wilcoxon_rez"]].to_string(index=False))
    print(f"\nTiempo total: {total:.1f} s")


if __name__ == "__main__":
    main()
