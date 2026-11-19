"""Costo del protocolo de evaluación: número de ajustes y tiempo de pared de cada etapa, a partir de los CSV e informes ya generados.

Ninguna cifra se estima: los ajustes se cuentan en los CSV de resultados (y, en la búsqueda interna, con `n_iter` × pliegues internos de `config.yaml`) y los tiempos de pared salen
de los informes (`resultados_v1/v2/v3.md`) y de los `*_tiempos.json`. Los tiempos de ajuste acumulados salen de las columnas `t_ajuste_s` (y `t_busqueda_s`) de esos CSV; se midieron
**con paralelismo** (excepto las v1–v3, que corrieron en un solo proceso), así que son una referencia, no una medición controlada.
"""
import json
import re
from pathlib import Path

import pandas as pd

R = Path("reports/resultados")


def _tiempo_md(ruta: str | Path) -> float:
    """«Tiempo total de la corrida: N s» de un informe markdown (con o sin negritas)."""
    m = re.search(r"Tiempo total de la corrida:?\*{0,2}:? *(\d+) s", Path(ruta).read_text(encoding="utf-8"))
    if not m:
        raise ValueError(f"no se encontró el tiempo total en {ruta}")
    return float(m.group(1))


def _json(ruta: str | Path) -> dict:
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


def _etapa(nombre, ajustes, configuraciones, pared, paralelismo, t_ajuste_acumulado, detalle="") -> dict:
    return {"etapa": nombre, "configuraciones": configuraciones, "ajustes": int(ajustes), "tiempo_pared_s": float(pared), "paralelismo": paralelismo,
            "suma_t_ajuste_s": float(t_ajuste_acumulado), "detalle": detalle}


def costo_protocolo(cfg: dict, resultados: str | Path = R, informes: str | Path = "reports") -> pd.DataFrame:
    """Una fila por etapa: configuraciones, número de ajustes, tiempo de pared, paralelismo y suma de los tiempos de ajuste medidos."""
    r, inf = Path(resultados), Path(informes)
    a = cfg["modelado"]["ajuste_temporal"]
    por_busqueda = a["n_iter"] * a["n_splits_interno"] + 1  # ajustes de una búsqueda: 30 × 3 pliegues internos + el reajuste del mejor (el ajuste por defecto se cuenta aparte)
    filas = []
    for nombre, csv, md in (("v1: 36 configuraciones, GroupKFold 5", "experimentos_v1.csv", "resultados_v1.md"), ("v2: formulaciones con historia, GroupKFold 5", "experimentos_v2.csv", "resultados_v2.md"),
                            ("v3: protocolo repetido 5 × 5", "experimentos_v3.csv", "resultados_v3.md")):
        d = pd.read_csv(r / csv)
        m = d[d["tipo"] == "modelo"]
        filas.append(_etapa(nombre, len(m), m["configuracion"].nunique(), _tiempo_md(inf / md), "1 proceso", m["t_ajuste_s"].sum()))
    t2j = _json(r / "experimentos_tarea2_tiempos.json")
    for nombre, csv, clave in (("Tarea 2: cribado (120 configuraciones, 1 semilla)", "experimentos_tarea2_cribado.csv", "t_cribado_s"),
                               ("Tarea 2: confirmación 5 × 5", "experimentos_tarea2_confirmacion.csv", "t_confirmacion_s")):
        d = pd.read_csv(r / csv)
        m = d[d["tipo"] == "modelo"]
        filas.append(_etapa(nombre, len(m), m["configuracion"].nunique(), t2j[clave], "16 procesos", m["t_ajuste_s"].sum()))
    at = _json(r / "ajuste_temporal_tiempos.json")
    d = pd.read_csv(r / "ajuste.csv")
    n_def, n_aj = int((d["variante"] == "defecto").sum()), int((d["variante"] == "ajustado").sum())
    filas.append(_etapa("Ajuste anidado (7 configuraciones, 5 × 5 externos)", n_def + n_aj * por_busqueda, d.loc[d["variante"] != "linea_base", "configuracion"].nunique(), at["t_ajuste_s"], "16 procesos",
                        d["t_ajuste_s"].sum() + d["t_busqueda_s"].sum(), f"{n_def} ajustes por defecto + {n_aj} búsquedas internas de {por_busqueda} ajustes ({a['n_iter']} iteraciones × {a['n_splits_interno']} pliegues + el reajuste)"))
    p = pd.read_csv(r / "temporal_predicciones.csv", usecols=["particion", "configuracion", "variante", "t_ajuste_s", "t_busqueda_s"])
    g = p[p["variante"].isin(["defecto", "ajustado"])].drop_duplicates(["particion", "configuracion", "variante"])
    t_def, t_aj = int((g["variante"] == "defecto").sum()), int((g["variante"] == "ajustado").sum())
    filas.append(_etapa("Particiones temporales (2 particiones × 7 configuraciones)", t_def + t_aj * por_busqueda, g["configuracion"].nunique(), at["t_temporal_s"], "16 procesos",
                        g["t_ajuste_s"].sum() + g["t_busqueda_s"].sum(), f"{t_def} ajustes por defecto + {t_aj} búsquedas internas de {por_busqueda} ajustes"))
    f = pd.read_csv(r / "arranque_frio_municipio.csv")
    fm = f[f["tipo"] == "modelo"]
    filas.append(_etapa("Arranque en frío por municipio 5 × 5", len(fm), fm["configuracion"].nunique(), at["t_frio_s"], "16 procesos", fm["t_ajuste_s"].sum()))
    sj = _json(r / "sesgo_sensibilidad_tiempos.json")
    s = pd.read_csv(r / "sensibilidad_pliegues.csv")
    sm = s[s["tipo"] == "modelo"]
    filas.append(_etapa("Sesgo y sensibilidad (5 escenarios × 5 modelos finales × 25 pliegues)", len(sm), sm["configuracion"].nunique(), sj["t_total_s"], "16 procesos", sm["t_ajuste_s"].sum(),
                        "el tiempo de pared incluye además subgrupos, el diagnóstico del colapso, SHAP, permutación y coeficientes"))
    out = pd.DataFrame(filas)
    out["tiempo_pared_min"] = out["tiempo_pared_s"] / 60
    return out
