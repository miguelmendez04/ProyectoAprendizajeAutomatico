"""Sensibilidad de las conclusiones: se re-entrena todo con el mismo protocolo (GroupKFold por IES 5 × 5, semillas 301–305) en dataset filtrados.

Escenarios (filtran el dataset **ya construido**; no se re-corre el ETL):
- `base`: el dataset completo (referencia).
- `sin_bogota`: sin las filas de Bogotá, en entrenamiento y en prueba.
- `umbral_200`: solo filas con denominador (matrícula SPADIES t-2) de al menos 200 (`mitigacion.filtrar_denominador_minimo`).
- `sin_sena`: sin las filas del SENA (`es_sena`).
- `desde_2017`: solo los años 2017–2021, con matrícula SNIES observada (en 2015–2016 falta y se imputa), para verificar la hipótesis sobre el SNIES imputado.

Las columnas de historia no se recalculan: se conservan las del dataset (calculadas con todos los años). Los modelos usan los hiperparámetros finales **sin volver a
ajustarlos** en cada escenario (re-ajustar cada uno costaría ~7 min por escenario); lo que cambia es el entrenamiento, el conjunto de prueba y las líneas base.
Una conclusión «cambia» cuando la clase del modelo frente a su línea base (`supera`, `empeora` o `sin diferencia firme`, con significancia en las dos pruebas)
difiere de la del escenario base.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.modelado import evaluacion_final as ef
from src.modelado.mitigacion import filtrar_denominador_minimo

ESCENARIOS = ["base", "sin_bogota", "umbral_200", "sin_sena", "desde_2017"]
DESCRIPCION = {"base": "Dataset completo", "sin_bogota": "Sin Bogotá (entrenamiento y prueba)", "umbral_200": "Denominador mínimo 200 (en lugar de 100)",
               "sin_sena": "Sin SENA", "desde_2017": "Solo 2017–2021 (matrícula SNIES observada)"}


def escenarios(df: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    p = cfg["modelado"]["sesgo_sensibilidad"]["sensibilidad"]
    out = {
        "base": df,
        "sin_bogota": df[df["cod_divipola_mpio"].astype(str) != "11001"],
        "umbral_200": filtrar_denominador_minimo(df, p["umbral_denominador"], cfg.get("min_matriculados_t2", 100)),
        "sin_sena": df[~df["es_sena"].astype(bool)],
        "desde_2017": df[df["anio"] >= p["anio_minimo_snies"]],
    }
    return {k: v.reset_index(drop=True) for k, v in out.items()}


def ejecutar_sensibilidad(cfg: dict, df: pd.DataFrame, modelos: list[dict], parcial: str | Path | None = None, verbose: bool = False,
                          base: tuple[pd.DataFrame, dict] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(resumen por escenario y modelo, métricas por pliegue de todos los escenarios).

    Con `parcial` (un directorio) cada escenario se guarda al terminar y se reutiliza si el archivo existe (reanudar). `base` permite pasar la evaluación
    ya hecha del dataset completo `(res, oof)` para no repetirla.
    """
    p = cfg["modelado"]["sesgo_sensibilidad"]
    carpeta = Path(parcial) if parcial else None
    if carpeta:
        carpeta.mkdir(parents=True, exist_ok=True)
    resumenes, pliegues = [], []
    for nombre, d in escenarios(df, cfg).items():
        archivo = carpeta / f"escenario_{nombre}.csv" if carpeta else None
        if archivo is not None and archivo.exists():
            res = pd.read_csv(archivo)
        elif nombre == "base" and base is not None:
            res = base[0]
        else:
            res, _ = ef.evaluar_oof(cfg, d, modelos, verbose)
        if archivo is not None and not archivo.exists():
            res.to_csv(archivo, index=False)
        r = ef.resumen_escenario(res, p["nivel_confianza"]).assign(escenario=nombre, n_filas=len(d), n_ies=int(d["cod_ies"].nunique()))
        resumenes.append(r)
        pliegues.append(res.assign(escenario=nombre))
        if verbose:
            print(f"escenario {nombre}: {len(d)} filas, {d['cod_ies'].nunique()} IES", flush=True)
    r = pd.concat(resumenes, ignore_index=True)
    clase_base = r[r["escenario"] == "base"].set_index("configuracion")["clase"]
    r["clase_base"] = r["configuracion"].map(clase_base)
    r["cambia_conclusion"] = (r["tipo"] == "modelo") & (r["clase"] != r["clase_base"])
    return r, pd.concat(pliegues, ignore_index=True)
