"""Ajuste de hiperparámetros con validación anidada, particiones temporales y arranque en frío estricto (por municipio).

Uso: python run_ajuste_temporal.py [--config RUTA] [--sin-informe | --solo-informe]
  (sin opciones)   calcula todo, guarda los CSV y escribe el informe y las figuras
  --sin-informe    solo calcula y guarda los CSV (y los tiempos)
  --solo-informe   no recalcula: lee los CSV y los tiempos guardados y regenera informe y figuras
  --reanudar       reutiliza las etapas ya guardadas (y las configuraciones del ajuste ya terminadas) en lugar de repetirlas
Salidas (rutas en config.yaml: modelado.ajuste_temporal.salidas): reports/resultados/ajuste.csv, temporal.csv, arranque_frio_municipio.csv (y
sus resúmenes), reports/resultados_ajuste_temporal.md, reports/figuras/ajuste_vs_defecto.png y temporal_por_anio.png.
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.modelado import ajuste as aj
from src.modelado import arranque_frio as af
from src.modelado import tarea2 as t2
from src.modelado import temporal as tp
from src.modelado.particion import cargar_dataset

SALIDAS = {"ajuste": "reports/resultados/ajuste.csv", "ajuste_resumen": "reports/resultados/ajuste_resumen.csv",
           "hiperparametros": "reports/resultados/ajuste_hiperparametros.csv", "temporal": "reports/resultados/temporal.csv",
           "temporal_predicciones": "reports/resultados/temporal_predicciones.csv", "temporal_grupos": "reports/resultados/temporal_grupos.csv",
           "arranque_frio": "reports/resultados/arranque_frio_municipio.csv", "arranque_frio_resumen": "reports/resultados/arranque_frio_municipio_resumen.csv",
           "arranque_frio_pliegues": "reports/resultados/arranque_frio_pliegues.csv", "informe": "reports/resultados_ajuste_temporal.md",
           "figura_ajuste": "reports/figuras/ajuste_vs_defecto.png", "figura_temporal": "reports/figuras/temporal_por_anio.png"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/config.yaml")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--sin-informe", action="store_true")
    g.add_argument("--solo-informe", action="store_true")
    ap.add_argument("--reanudar", action="store_true", help="reutiliza lo ya calculado (por configuración en el ajuste, por etapa en las demás)")
    args = ap.parse_args()
    t0 = time.perf_counter()
    cfg = load_config(args.config)
    p = aj.parametros_ajuste(cfg)
    s = {k: Path(v) for k, v in {**SALIDAS, **p.get("salidas", {})}.items()}
    s["ajuste"].parent.mkdir(parents=True, exist_ok=True)
    tiempos = s["ajuste"].with_name("ajuste_temporal_tiempos.json")
    boot = cfg["modelado"]["ajuste_temporal"]["bootstrap"]
    if not args.solo_informe:
        df = cargar_dataset(cfg)
        marcas = json.loads(tiempos.read_text()) if args.reanudar and tiempos.exists() else {}

        def guardar_marcas():
            tiempos.write_text(json.dumps(marcas), encoding="utf-8")

        if args.reanudar and s["ajuste"].exists():
            print("Ajuste anidado: ya guardado, se reutiliza", flush=True)
        else:
            t = time.perf_counter()
            ajuste = aj.ejecutar_ajuste(cfg, df, verbose=True, parcial=s["ajuste"].parent / "_parcial_ajuste" if args.reanudar else None)
            marcas["t_ajuste_s"] = marcas.get("t_ajuste_s", 0.0) + time.perf_counter() - t  # con reanudación solo suma lo recalculado
            ajuste.to_csv(s["ajuste"], index=False)
            aj.resumen_ajuste(ajuste, p["nivel_confianza"]).to_csv(s["ajuste_resumen"], index=False)
            aj.resumen_hiperparametros(ajuste).to_csv(s["hiperparametros"], index=False)
            guardar_marcas()
            print(f"\nAjuste anidado: {marcas['t_ajuste_s']:.0f} s", flush=True)

        if args.reanudar and s["temporal"].exists():
            print("Particiones temporales: ya guardadas, se reutilizan", flush=True)
        else:
            t = time.perf_counter()
            pred = tp.ejecutar_temporal(cfg, df, verbose=True)
            pred.to_csv(s["temporal_predicciones"], index=False)
            tp.resumen_temporal(pred, boot["replicas"], boot["semilla"], p["nivel_confianza"]).to_csv(s["temporal"], index=False)
            tp.errores_por_grupo(pred).to_csv(s["temporal_grupos"], index=False)
            marcas["t_temporal_s"] = time.perf_counter() - t
            guardar_marcas()
            print(f"Particiones temporales: {marcas['t_temporal_s']:.0f} s", flush=True)

        if args.reanudar and s["arranque_frio"].exists():
            print("Arranque en frío: ya guardado, se reutiliza", flush=True)
        else:
            t = time.perf_counter()
            frio, tam = af.ejecutar_arranque_frio(cfg, df, verbose=True)
            frio.to_csv(s["arranque_frio"], index=False)
            t2.resumen_confirmacion(frio, p["nivel_confianza"]).to_csv(s["arranque_frio_resumen"], index=False)
            tam.to_csv(s["arranque_frio_pliegues"], index=False)
            marcas["t_frio_s"] = time.perf_counter() - t
        marcas["t_total_s"] = marcas["t_ajuste_s"] + marcas["t_temporal_s"] + marcas["t_frio_s"]
        guardar_marcas()
        print(f"Arranque en frío por municipio: {marcas['t_frio_s']:.0f} s | total {marcas['t_total_s']:.0f} s", flush=True)
    if not args.sin_informe:
        from src.modelado.informe_ajuste import generar_informe

        generar_informe(cfg, {k: v for k, v in s.items()}, json.loads(tiempos.read_text()))


if __name__ == "__main__":
    main()
