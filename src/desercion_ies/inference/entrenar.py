"""`desercion entrenar`: entrena con TODO el dataset los modelos finales y los guarda en `models/` (joblib + JSON de metadatos).

Modelos (los finales del análisis, con los hiperparámetros de consenso del ajuste anidado):
- Tarea 1 (con historia): `tarea1_ridge` = C solo historia con ridge (corrige la ingenua con la historia de la IES).
- Tarea 2 (sin historia): `tarea2_ridge` = ridge con información mutua (logit, imputación por mediana) y
  `tarea2_mediana` = mediana por carácter × sector (línea base).

Cada modelo se entrena por el mismo camino que la evaluación (`ajuste.preparar` + `evaluacion_final.construir_final`); no hay semillas nuevas.
Cada `.joblib` lleva un `.json` con fecha, commit, SHA-256 del dataset y del modelo, versiones, hiperparámetros, columnas esperadas y el MAE de
validación publicado en los informes. Salidas por defecto en `models/` (rutas relativas a la raíz del proyecto).
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd

from desercion_ies import __version__, paths
from desercion_ies.config import load_config
from desercion_ies.evaluation import ajuste as aj
from desercion_ies.evaluation import evaluacion_final as ef
from desercion_ies.evaluation import tarea2 as t2
from desercion_ies.evaluation.experimentos import parametros_experimentos
from desercion_ies.features.particion import cargar_dataset
from desercion_ies.inference.modelos import TASA, MedianaCaracterSector, ModeloEntrenado
from desercion_ies.reproducibilidad import sha256_archivo

log = logging.getLogger(__name__)

# clave del modelo guardado -> (nombre del modelo final en la evaluación, descripción)
FINALES = {
    "tarea1_ridge": (
        "T1 C solo historia | ridge (ajustado)",
        "Tarea 1: ridge de corrección sobre la ingenua, solo historia",
    ),
    "tarea2_ridge": (
        "T2 ridge info_mutua (ajustado)",
        "Tarea 2: ridge con información mutua, solo contexto",
    ),
}
MEDIANA = (
    "tarea2_mediana",
    "T2 base_mediana_grupo",
    "Tarea 2: mediana por carácter × sector (línea base)",
)
CLAVES = [*FINALES, MEDIANA[0]]


def configurar(ap: argparse.ArgumentParser) -> None:
    """Opciones de `desercion entrenar`."""
    ap.add_argument("--config", default="configs/config.yaml", help="archivo de configuración")
    ap.add_argument(
        "--ajuste",
        default=None,
        help="CSV del ajuste anidado de donde salen los hiperparámetros (por defecto el de la configuración)",
    )
    ap.add_argument(
        "--salida", default="models", help="carpeta de los modelos (se crea si no existe)"
    )


def _git(*args: str) -> str | None:
    try:
        r = subprocess.run(
            ["git", *args],
            cwd=paths.PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
        return r.stdout.strip()
    except Exception:  # noqa: BLE001  (sin git o fuera de un repositorio: el commit queda como desconocido)
        return None


def _versiones() -> dict:
    import numpy
    import scipy
    import sklearn
    import xgboost

    return {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "numpy": numpy.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "joblib": joblib.__version__,
        "desercion_ies": __version__,
    }


def entrenar_modelo(m: dict, cfg: dict, df: pd.DataFrame) -> ModeloEntrenado:
    """Entrena un modelo final (especificación de `evaluacion_final.modelos_finales`) con las filas de `df`."""
    X, numericas, categoricas = aj.preparar(m["spec"], cfg, df)
    est = ef.construir_final(
        m, numericas, categoricas, parametros_experimentos(cfg), t2.parametros_tarea2(cfg)
    )
    est.fit(X, df["tasa_desercion"].astype(float).to_numpy())
    if m["tarea"] == "T1":
        obligatorias, opcionales = [TASA], [c for c in numericas if c != TASA]
    else:
        obligatorias, opcionales = [*numericas, *categoricas], []
    clave = next(k for k, (nombre, _) in FINALES.items() if nombre == m["nombre"])
    return ModeloEntrenado(
        clave=clave,
        nombre=m["nombre"],
        tarea=int(m["tarea"][1]),
        estimador=est,
        numericas=numericas,
        categoricas=categoricas,
        obligatorias=obligatorias,
        opcionales=opcionales,
        categorias={c: sorted(df[c].astype(str).unique()) for c in categoricas},
    )


def entrenar_mediana(df: pd.DataFrame) -> ModeloEntrenado:
    """La línea base de la Tarea 2: mediana por carácter × sector (columnas crudas del dataset)."""
    est = MedianaCaracterSector().fit(df, df["tasa_desercion"].astype(float).to_numpy())
    columnas = ["sector", "caracter"]
    return ModeloEntrenado(
        clave=MEDIANA[0],
        nombre="T2 mediana carácter × sector",
        tarea=2,
        estimador=est,
        numericas=[],
        categoricas=columnas,
        obligatorias=columnas,
        categorias={c: sorted(df[c].astype(str).unique()) for c in columnas},
        usa_preparar_x=False,
    )


def _validacion(
    resumen: pd.DataFrame | None,
    configuracion: str,
    variante: str,
    linea_base: str | None,
    fuente: str,
) -> dict | None:
    """MAE de validación publicado (ajuste anidado, GroupKFold por IES 5 × 5) del modelo y de su línea base."""
    if resumen is None:
        return None
    f = resumen[(resumen["configuracion"] == configuracion) & (resumen["variante"] == variante)]
    if f.empty:
        return None
    f = f.iloc[0]
    out = {
        "fuente": fuente,
        "protocolo": "validación anidada, GroupKFold por IES 5 × 5 (semillas 301-305), IC 95 % de Nadeau y Bengio",
        "mae": float(f["mae_media"]),
        "mae_ic95": [float(f["mae_ic_inf"]), float(f["mae_ic_sup"])],
        "n_ajustes": int(f["n_ajustes"]),
    }
    if linea_base:
        b = resumen[resumen["configuracion"] == linea_base]
        if not b.empty:
            out.update(
                linea_base=linea_base,
                mae_linea_base=float(b.iloc[0]["mae_media"]),
                mejora_frente_a_la_linea_base_pct=float(f["mejora_base_pct"]),
                p_wilcoxon=float(f["p_wilcoxon_base"]),
                p_t_corregida=float(f["p_t_base"]),
            )
    return out


def _guardar(
    modelo: ModeloEntrenado,
    descripcion: str,
    hp: dict,
    df: pd.DataFrame,
    dataset: Path | None,
    validacion: dict | None,
    destino: Path,
) -> Path:
    destino.mkdir(parents=True, exist_ok=True)
    archivo = destino / f"{modelo.clave}.joblib"
    joblib.dump(modelo, archivo)
    estado = _git("status", "--porcelain", "--untracked-files=no")
    meta = {
        "modelo": modelo.clave,
        "nombre": modelo.nombre,
        "descripcion": descripcion,
        "tarea": modelo.tarea,
        "fecha_entrenamiento_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": _git("rev-parse", "HEAD"),
        "arbol_sin_cambios": (estado == "") if estado is not None else None,
        "dataset": {
            "ruta": dataset.as_posix() if dataset else None,
            "sha256": sha256_archivo(dataset) if dataset and dataset.exists() else None,
            "filas": len(df),
            "ies": int(df["cod_ies"].nunique()),
        },
        "versiones": _versiones(),
        "hiperparametros": hp,
        "columnas": {
            "obligatorias": modelo.obligatorias,
            "opcionales": modelo.opcionales,
            "numericas": modelo.numericas,
            "categoricas": modelo.categoricas,
        },
        "categorias_conocidas": modelo.categorias,
        "validacion": validacion,
        "archivo": archivo.name,
        "sha256_modelo": sha256_archivo(archivo),
    }
    (destino / f"{modelo.clave}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return archivo


def entrenar_modelos(
    cfg: dict,
    df: pd.DataFrame,
    ajuste: pd.DataFrame,
    destino: Path,
    dataset: Path | None = None,
    resumen: pd.DataFrame | None = None,
    fuente: str = "",
) -> dict[str, Path]:
    """Entrena y guarda los tres modelos; devuelve clave -> archivo."""
    finales = {m["nombre"]: m for m in ef.modelos_finales(cfg, ajuste)}
    salida = {}
    for clave, (nombre, descripcion) in FINALES.items():
        m = finales[nombre]
        modelo = entrenar_modelo(m, cfg, df)
        base = ef.BASES[m["tarea"]]
        validacion = _validacion(resumen, m["spec"]["nombre"], "ajustado", base, fuente)
        salida[clave] = _guardar(
            modelo, descripcion, m["hiperparametros"], df, dataset, validacion, destino
        )
        log.info("%s: entrenado con %d filas (%s)", clave, len(df), salida[clave])
    clave, linea, descripcion = MEDIANA
    validacion = _validacion(resumen, linea, "linea_base", None, fuente)
    salida[clave] = _guardar(
        entrenar_mediana(df), descripcion, {}, df, dataset, validacion, destino
    )
    log.info("%s: entrenado con %d filas (%s)", clave, len(df), salida[clave])
    return salida


def ejecutar(args: argparse.Namespace) -> int:
    """Entrena los modelos finales con el dataset completo y los guarda en `--salida`."""
    cfg = load_config(args.config)
    df = cargar_dataset(cfg)
    ajuste = pd.read_csv(args.ajuste or cfg["modelado"]["sesgo_sensibilidad"]["ajuste"])
    ruta_resumen = Path(cfg["modelado"]["ajuste_temporal"]["salidas"]["ajuste_resumen"])
    resumen = pd.read_csv(ruta_resumen) if ruta_resumen.exists() else None
    dataset = Path(cfg["rutas"]["processed"]) / f"dataset_{cfg['unidad_analisis']}.csv"
    destino = Path(args.salida)
    archivos = entrenar_modelos(cfg, df, ajuste, destino, dataset, resumen, ruta_resumen.as_posix())
    log.info("%d modelos guardados en %s", len(archivos), destino)
    return 0
