"""Mide el costo computacional de los modelos finales y de los métodos de preprocesamiento, de forma controlada (un solo hilo).

Uso: python run_complejidad.py [--config RUTA] [--sin-informe | --solo-informe] [--reanudar]
  (sin opciones)   mide todo, guarda los CSV y escribe el informe y las figuras
  --sin-informe    solo mide y guarda los CSV
  --solo-informe   no mide: lee los CSV y regenera informe y figuras
  --reanudar       reutiliza los componentes ya medidos (un punto de control por componente en reports/resultados/_parcial_complejidad/)
Importante: no corra otra cosa pesada mientras mide (altera los tiempos). Las variables de hilos se fijan en 1 ANTES de importar numpy.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

from src.config import load_config  # noqa: E402
from src.modelado import complejidad as cx  # noqa: E402
from src.modelado.costo_protocolo import costo_protocolo  # noqa: E402
from src.modelado.particion import cargar_dataset  # noqa: E402

SALIDAS = {"entorno": "reports/resultados/complejidad_entorno.json", "complejidad": "reports/resultados/complejidad.csv", "escalado": "reports/resultados/complejidad_escalado.csv",
           "exponentes": "reports/resultados/complejidad_exponentes.csv", "protocolo": "reports/resultados/complejidad_protocolo.csv",
           "informe": "reports/resultados_complejidad.md", "figura_costo_error": "reports/figuras/complejidad_costo_error.png", "figura_escalado": "reports/figuras/complejidad_escalado.png"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/config.yaml")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--sin-informe", action="store_true")
    g.add_argument("--solo-informe", action="store_true")
    ap.add_argument("--reanudar", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    pc = cx.parametros_complejidad(cfg)
    s = {k: Path(v) for k, v in {**SALIDAS, **pc.get("salidas", {})}.items()}
    s["complejidad"].parent.mkdir(parents=True, exist_ok=True)
    parcial = s["complejidad"].parent / "_parcial_complejidad"

    if not args.solo_informe:
        parcial.mkdir(parents=True, exist_ok=True)
        entorno = cx.entorno()
        entorno["afinidad"] = cx.fijar_afinidad(pc["afinidad_cpu"])
        entorno["repeticiones"], entorno["calentamiento"] = pc["repeticiones"], pc["calentamiento"]
        s["entorno"].write_text(json.dumps(entorno, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Entorno:", json.dumps(entorno, ensure_ascii=False), flush=True)
        df = cargar_dataset(cfg)
        t0 = time.perf_counter()
        filas = []
        for c in cx.componentes(cfg):
            archivo = parcial / (re.sub(r"[^0-9A-Za-z_]+", "_", c["nombre"]) + ".json")
            if args.reanudar and archivo.exists():
                filas += json.loads(archivo.read_text(encoding="utf-8"))
                print(f"{c['nombre']}: ya medido", flush=True)
                continue
            t = time.perf_counter()
            medidas = cx.medir_componente(c, df, pc)
            archivo.write_text(json.dumps(medidas), encoding="utf-8")
            filas += medidas
            ultimo = medidas[-1]
            print(f"{c['nombre']}: ajuste {ultimo['ajuste_mediana_s'] * 1000:.1f} ms (n={ultimo['n_filas']}), medición {time.perf_counter() - t:.0f} s", flush=True)
        escalado = pd.DataFrame(filas)
        escalado.to_csv(s["escalado"], index=False)
        escalado[escalado["fraccion_ies"] >= 1].to_csv(s["complejidad"], index=False)
        cx.exponentes(escalado).to_csv(s["exponentes"], index=False)
        costo_protocolo(cfg).to_csv(s["protocolo"], index=False)
        print(f"Medición total: {time.perf_counter() - t0:.0f} s", flush=True)
        archivo_t = s["complejidad"].parent / "complejidad_tiempo.json"
        if not (args.reanudar and archivo_t.exists()):  # al reanudar se conserva el tiempo de la medición original
            archivo_t.write_text(json.dumps({"t_medicion_s": time.perf_counter() - t0}), encoding="utf-8")

    if not args.sin_informe:
        from src.modelado.informe_complejidad import generar_informe

        generar_informe(cfg, s)


if __name__ == "__main__":
    main()
