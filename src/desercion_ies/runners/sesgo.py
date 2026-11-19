"""Sesgo medido en los modelos finales, sensibilidad, diagnóstico del colapso del ridge e interpretabilidad.

Uso: desercion evaluar sesgo [--config RUTA] [--sin-informe | --solo-informe] [--reanudar]
  (sin opciones)   calcula todo, guarda los CSV y escribe el informe y las figuras
  --sin-informe    solo calcula y guarda los CSV (y los tiempos)
  --solo-informe   no recalcula: lee los CSV y regenera informe y figuras
  --reanudar       reutiliza las etapas ya guardadas (base, sensibilidad —por escenario—, colapso, interpretabilidad)
Etapas: base (modelos finales, hiperparámetros, error por subgrupo) → sensibilidad (4 escenarios) → colapso → interpretabilidad (SHAP, permutación, coeficientes).
Salidas (rutas en config.yaml: modelado.sesgo_sensibilidad.salidas): reports/tables/{subgrupos,sensibilidad,colapso_ridge,shap_top}.csv (y auxiliares),
reports/06_sesgo_modelos.md y reports/figures/{error_subgrupos,shap_t1,shap_t2,coeficientes_ridge}.png.
"""
import logging
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from desercion_ies.config import load_config
from desercion_ies.evaluation import evaluacion_final as ef
from desercion_ies.analysis import interpretabilidad as it
from desercion_ies.analysis import sensibilidad as sn
from desercion_ies.analysis import subgrupos as sg
from desercion_ies.evaluation import tarea2 as t2
from desercion_ies.analysis.diagnostico_colapso import ejecutar_diagnostico
from desercion_ies.evaluation.experimentos import parametros_experimentos
from desercion_ies.features.particion import cargar_dataset

SALIDAS = {
    "subgrupos": "reports/tables/subgrupos.csv", "sensibilidad": "reports/tables/sensibilidad.csv", "sensibilidad_pliegues": "reports/tables/sensibilidad_pliegues.csv",
    "colapso": "reports/tables/colapso_ridge.csv", "colapso_z": "reports/tables/colapso_ridge_z.csv", "colapso_peores": "reports/tables/colapso_ridge_peores.csv",
    "shap_top": "reports/tables/shap_top.csv", "shap_grupos": "reports/tables/shap_grupos.csv", "permutacion": "reports/tables/permutacion.csv",
    "coeficientes": "reports/tables/coeficientes_ridge.csv", "hiperparametros": "reports/tables/hiperparametros_finales.csv",
    "informe": "reports/06_sesgo_modelos.md", "figura_subgrupos": "reports/figures/error_subgrupos.png", "figura_shap_t1": "reports/figures/shap_t1.png",
    "figura_shap_t2": "reports/figures/shap_t2.png", "figura_coeficientes": "reports/figures/coeficientes_ridge.png",
}
XGB_T1, XGB_T2 = "T1 B historia+contexto | xgboost logit (ajustado)", "T2 xgboost con clusters (ajustado)"
RIDGE_T1, RIDGE_T2 = "T1 C solo historia | ridge (ajustado)", "T2 ridge info_mutua (ajustado)"


log = logging.getLogger(__name__)


def configurar(ap: argparse.ArgumentParser) -> None:
    """Opciones de la línea de comandos de esta etapa."""
    ap.add_argument("--config", default="configs/config.yaml", help="archivo de configuración")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--sin-informe", action="store_true")
    g.add_argument("--solo-informe", action="store_true")
    ap.add_argument("--reanudar", action="store_true", help="reutiliza lo ya calculado")


def ejecutar(args: argparse.Namespace) -> None:
    """Corre la etapa con las opciones ya leídas."""
    cfg = load_config(args.config)
    p = cfg["modelado"]["sesgo_sensibilidad"]
    s = {k: Path(v) for k, v in {**SALIDAS, **p.get("salidas", {})}.items()}
    s["subgrupos"].parent.mkdir(parents=True, exist_ok=True)
    parcial = s["subgrupos"].parent / "_parcial_sesgo"
    tiempos = s["subgrupos"].with_name("sesgo_sensibilidad_tiempos.json")
    marcas = json.loads(tiempos.read_text()) if tiempos.exists() and (args.reanudar or args.solo_informe) else {}

    def etapa(nombre: str, salidas: list[str]) -> bool:
        """Si hay que calcular la etapa (no se reanuda con las salidas ya guardadas)."""
        if args.reanudar and all(s[k].exists() for k in salidas):
            log.info(f"{nombre}: ya guardada, se reutiliza")
            return False
        return True

    if not args.solo_informe:
        df = cargar_dataset(cfg)
        modelos = ef.modelos_finales(cfg)
        nombres = {m["nombre"]: m for m in modelos}
        y = df["tasa_desercion"].astype(float)

        # --- base: hiperparámetros finales y error por subgrupo ---
        if etapa("base", ["subgrupos", "hiperparametros"]) or not (parcial / "escenario_base.csv").exists():
            t = time.perf_counter()
            parcial.mkdir(parents=True, exist_ok=True)
            hp = ef.hiperparametros_finales(pd.read_csv(p["ajuste"]))  # noqa: F841
            pd.DataFrame([{"modelo": m["nombre"], "variante": m["variante"], "hiperparametros": json.dumps(m["hiperparametros"], sort_keys=True)} for m in modelos]).to_csv(s["hiperparametros"], index=False)
            res, oof = ef.evaluar_oof(cfg, df, modelos, verbose=True)
            res.to_csv(parcial / "escenario_base.csv", index=False)
            np.savez_compressed(parcial / "oof_base.npz", **{k: v for k, v in oof.items()})
            sub, cortes = sg.asignar_subgrupos(df, p["ciudades"])
            tabla = sg.analizar_subgrupos(df, sub, oof, modelos, p["bootstrap"]["replicas"], p["bootstrap"]["semilla"], p["nivel_confianza"])
            tabla.attrs["cortes"] = cortes
            tabla.to_csv(s["subgrupos"], index=False)
            marcas["cortes_tamano"] = cortes
            marcas["t_base_s"] = time.perf_counter() - t
            tiempos.write_text(json.dumps(marcas), encoding="utf-8")
            log.info(f"Base y subgrupos: {marcas['t_base_s']:.0f} s")

        # --- sensibilidad ---
        if etapa("sensibilidad", ["sensibilidad", "sensibilidad_pliegues"]):
            t = time.perf_counter()
            base_res = pd.read_csv(parcial / "escenario_base.csv")
            r, pl = sn.ejecutar_sensibilidad(cfg, df, modelos, parcial=parcial, verbose=True, base=(base_res, {}))
            r.to_csv(s["sensibilidad"], index=False)
            pl.to_csv(s["sensibilidad_pliegues"], index=False)
            marcas["t_sensibilidad_s"] = time.perf_counter() - t
            tiempos.write_text(json.dumps(marcas), encoding="utf-8")
            log.info(f"Sensibilidad: {marcas['t_sensibilidad_s']:.0f} s")

        # --- diagnóstico del colapso ---
        if etapa("colapso", ["colapso", "colapso_z", "colapso_peores"]):
            t = time.perf_counter()
            resumen, z, peores = ejecutar_diagnostico(cfg, df)
            resumen.to_csv(s["colapso"], index=False)
            z.to_csv(s["colapso_z"], index=False)
            peores.to_csv(s["colapso_peores"], index=False)
            marcas["t_colapso_s"] = time.perf_counter() - t
            tiempos.write_text(json.dumps(marcas), encoding="utf-8")
            log.info(f"Diagnóstico del colapso: {marcas['t_colapso_s']:.0f} s")

        # --- interpretabilidad ---
        if etapa("interpretabilidad", ["shap_top", "shap_grupos", "permutacion", "coeficientes"]):
            t = time.perf_counter()
            from desercion_ies.evaluation import ajuste as aj

            p_exp, p_t2 = parametros_experimentos(cfg), t2.parametros_tarea2(cfg)
            tops, grupos, perms = [], [], []
            for nombre in (XGB_T1, XGB_T2):
                m = nombres[nombre]
                X, num, cat = aj.preparar(m["spec"], cfg, df)
                est = ef.construir_final(m, num, cat, p_exp, p_t2).fit(X, y.to_numpy())
                sv, M, cols = it.calcular_shap(est, X)
                tops.append(it.tabla_shap(sv, M, cols, nombre, p["shap"]["top"]))
                g = it.shap_por_variable(sv, M, cols, nombre)
                pm = it.permutacion(cfg, df, m, p["n_jobs"])
                perms.append(pm)
                grupos.append(g.merge(pm[["variable", "aumento_mae_medio", "aumento_mae_sd", "rank_permutacion"]], on="variable", how="left"))
            pd.concat(tops).to_csv(s["shap_top"], index=False)
            pd.concat(grupos).to_csv(s["shap_grupos"], index=False)
            pd.concat(perms).to_csv(s["permutacion"], index=False)
            pd.concat([it.coeficientes_ridge(cfg, df, nombres[n], p["n_jobs"]) for n in (RIDGE_T2, RIDGE_T1)]).to_csv(s["coeficientes"], index=False)
            marcas["t_interpretabilidad_s"] = time.perf_counter() - t
            tiempos.write_text(json.dumps(marcas), encoding="utf-8")
            log.info(f"Interpretabilidad: {marcas['t_interpretabilidad_s']:.0f} s")
        marcas["t_total_s"] = sum(v for k, v in marcas.items() if k.startswith("t_") and k != "t_total_s")
        tiempos.write_text(json.dumps(marcas), encoding="utf-8")
        log.info(f"Total: {marcas['t_total_s']:.0f} s")

    if not args.sin_informe:
        from desercion_ies.reporting.informe_sesgo import generar_informe

        generar_informe(cfg, s, marcas)

