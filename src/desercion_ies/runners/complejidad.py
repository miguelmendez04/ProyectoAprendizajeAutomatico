"""Mide el costo computacional de los modelos finales y de los métodos de preprocesamiento, de forma controlada (un solo hilo).

Uso: desercion evaluar complejidad [--config RUTA] [--sin-informe | --solo-informe] [--reanudar]
  (sin opciones)   mide todo, guarda los CSV y escribe el informe y las figuras
  --sin-informe    solo mide y guarda los CSV
  --solo-informe   no mide: lee los CSV y regenera informe y figuras
  --reanudar       reutiliza los componentes ya medidos (un punto de control por componente en reports/tables/_parcial_complejidad/)
Importante: no corra otra cosa pesada mientras mide (altera los tiempos). Las variables de hilos se fijan en 1 ANTES de importar numpy.
"""
import logging
import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd

from desercion_ies.config import load_config
from desercion_ies.analysis import complejidad as cx
from desercion_ies.analysis.costo_protocolo import costo_protocolo
from desercion_ies.features.particion import cargar_dataset

SALIDAS = {"entorno": "reports/tables/complejidad_entorno.json", "complejidad": "reports/tables/complejidad.csv", "escalado": "reports/tables/complejidad_escalado.csv",
           "exponentes": "reports/tables/complejidad_exponentes.csv", "protocolo": "reports/tables/complejidad_protocolo.csv",
           "informe": "reports/07_complejidad.md", "figura_costo_error": "reports/figures/complejidad_costo_error.png", "figura_escalado": "reports/figures/complejidad_escalado.png"}


log = logging.getLogger(__name__)


def configurar(ap: argparse.ArgumentParser) -> None:
    """Opciones de la línea de comandos de esta etapa."""
    ap.add_argument("--config", default="configs/config.yaml", help="archivo de configuración")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--sin-informe", action="store_true")
    g.add_argument("--solo-informe", action="store_true")
    ap.add_argument("--reanudar", action="store_true")


def ejecutar(args: argparse.Namespace) -> None:
    """Corre la etapa con las opciones ya leídas."""
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
        log.info("Entorno:", json.dumps(entorno, ensure_ascii=False))
        df = cargar_dataset(cfg)
        t0 = time.perf_counter()
        filas = []
        for c in cx.componentes(cfg):
            archivo = parcial / (re.sub(r"[^0-9A-Za-z_]+", "_", c["nombre"]) + ".json")
            if args.reanudar and archivo.exists():
                filas += json.loads(archivo.read_text(encoding="utf-8"))
                log.info(f"{c['nombre']}: ya medido")
                continue
            t = time.perf_counter()
            medidas = cx.medir_componente(c, df, pc)
            archivo.write_text(json.dumps(medidas), encoding="utf-8")
            filas += medidas
            ultimo = medidas[-1]
            log.info(f"{c['nombre']}: ajuste {ultimo['ajuste_mediana_s'] * 1000:.1f} ms (n={ultimo['n_filas']}), medición {time.perf_counter() - t:.0f} s")
        escalado = pd.DataFrame(filas)
        escalado.to_csv(s["escalado"], index=False)
        escalado[escalado["fraccion_ies"] >= 1].to_csv(s["complejidad"], index=False)
        cx.exponentes(escalado).to_csv(s["exponentes"], index=False)
        costo_protocolo(cfg).to_csv(s["protocolo"], index=False)
        log.info(f"Medición total: {time.perf_counter() - t0:.0f} s")
        archivo_t = s["complejidad"].parent / "complejidad_tiempo.json"
        if not (args.reanudar and archivo_t.exists()):  # al reanudar se conserva el tiempo de la medición original
            archivo_t.write_text(json.dumps({"t_medicion_s": time.perf_counter() - t0}), encoding="utf-8")

    if not args.sin_informe:
        from desercion_ies.reporting.informe_complejidad import generar_informe

        generar_informe(cfg, s)

