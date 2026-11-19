"""`desercion predecir`: predice la tasa de deserción de filas nuevas con un modelo guardado por `desercion entrenar`.

  desercion predecir --tarea 1 --entrada historia.csv --salida predicciones.csv
  desercion predecir --tarea 2 --entrada contexto.csv --salida predicciones.csv [--modelo {ridge,mediana}]

La entrada es un CSV con una fila por IES × año y las columnas que espera el modelo (el JSON de `models/` las lista; las del dataset publicado
`data/processed/dataset_ies.csv` sirven de ejemplo). Se valida antes de predecir y los errores dicen qué falta.
- **Tarea 1** (con historia): exige `tasa_t1` (tasa de deserción del año anterior, entre 0 y 1). Las demás columnas de historia son opcionales.
- **Tarea 2** (sin historia): exige el contexto de la IES (`sector`, `caracter`, `log_matriculados_spadies_t2`, indicadores de Saber 11 y TerriData…).
La salida trae los identificadores que haya en la entrada (`cod_ies`, `anio`, …), `modelo` y `tasa_predicha` (proporción entre 0 y 1).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import pandas as pd

from desercion_ies.inference.modelos import (
    COLUMNAS_CODIGO,
    IDENTIFICADORES,
    EntradaInvalida,
    ModeloEntrenado,
)
from desercion_ies.reproducibilidad import sha256_archivo

log = logging.getLogger(__name__)

MODELOS_T2 = {"ridge": "tarea2_ridge", "mediana": "tarea2_mediana"}


def configurar(ap: argparse.ArgumentParser) -> None:
    """Opciones de `desercion predecir`."""
    ap.add_argument(
        "--tarea",
        type=int,
        choices=[1, 2],
        required=True,
        help="1: con historia (exige tasa_t1); 2: sin historia (solo contexto)",
    )
    ap.add_argument("--entrada", required=True, help="CSV de entrada")
    ap.add_argument("--salida", required=True, help="CSV de salida con las predicciones")
    ap.add_argument(
        "--modelo",
        choices=sorted(MODELOS_T2),
        default="mediana",
        help="solo Tarea 2: mediana por carácter × sector (por defecto) o ridge con información mutua (sin mejora firme sobre la mediana; ver models/model_card.md)",
    )
    ap.add_argument(
        "--modelos-dir",
        default="models",
        help="carpeta con los modelos guardados por `desercion entrenar`",
    )


def clave_modelo(tarea: int, modelo: str = "mediana") -> str:
    """Nombre del modelo guardado para la tarea pedida."""
    return "tarea1_ridge" if tarea == 1 else MODELOS_T2[modelo]


def cargar_modelo(clave: str, carpeta: str | Path = "models") -> tuple[ModeloEntrenado, dict]:
    """Carga un modelo y sus metadatos tras comprobar el SHA-256 del archivo (joblib es pickle: no se carga lo que no coincide)."""
    carpeta = Path(carpeta)
    meta_json, archivo = carpeta / f"{clave}.json", carpeta / f"{clave}.joblib"
    if not meta_json.exists() or not archivo.exists():
        raise EntradaInvalida(
            f"no hay un modelo guardado `{clave}` en {carpeta}: entrénelo con `desercion entrenar`"
        )
    meta = json.loads(meta_json.read_text(encoding="utf-8"))
    if sha256_archivo(archivo) != meta["sha256_modelo"]:
        raise EntradaInvalida(
            f"{archivo.name} no coincide con el SHA-256 de {meta_json.name}: el archivo cambió o está dañado; vuelva a entrenarlo con `desercion entrenar`"
        )
    import sklearn

    if meta["versiones"]["scikit-learn"] != sklearn.__version__:
        log.warning(
            "el modelo se entrenó con scikit-learn %s y aquí hay %s: las predicciones podrían diferir",
            meta["versiones"]["scikit-learn"],
            sklearn.__version__,
        )
    return joblib.load(archivo), meta


def leer_entrada(ruta: str | Path) -> pd.DataFrame:
    """Lee el CSV con los códigos (IES, municipio, sector, carácter) como texto."""
    ruta = Path(ruta)
    if not ruta.exists():
        raise EntradaInvalida(f"no existe el archivo de entrada: {ruta}")
    try:
        return pd.read_csv(
            ruta, dtype={c: t for c, t in COLUMNAS_CODIGO.items()}, encoding="utf-8-sig"
        )
    except pd.errors.EmptyDataError as e:
        raise EntradaInvalida(f"el archivo de entrada está vacío: {ruta}") from e
    except (pd.errors.ParserError, UnicodeDecodeError) as e:
        raise EntradaInvalida(
            f"no se pudo leer {ruta} como CSV (UTF-8, separado por comas): {e}"
        ) from e


def predecir_tabla(modelo: ModeloEntrenado, df: pd.DataFrame, etiqueta: str) -> pd.DataFrame:
    """Tabla de salida: identificadores presentes + `modelo` + `tasa_predicha`."""
    pred = modelo.predecir(df)
    ids = [c for c in IDENTIFICADORES if c in df.columns]
    salida = df[ids].copy() if ids else pd.DataFrame(index=df.index)
    salida["modelo"] = etiqueta
    salida["tasa_predicha"] = pred
    return salida.reset_index(drop=True)


ADVERTENCIA_RIDGE = (
    "el ridge de la Tarea 2 no mejora con firmeza a la mediana por carácter × sector y es inestable "
    "en la validación temporal (prepandemia); ver models/model_card.md"
)


def ejecutar(args: argparse.Namespace) -> int:
    """Valida la entrada, predice y escribe el CSV; devuelve 2 (sin traceback) si la entrada no es válida."""
    if args.tarea == 2 and args.modelo == "ridge":
        log.warning(ADVERTENCIA_RIDGE)
    try:
        modelo, meta = cargar_modelo(clave_modelo(args.tarea, args.modelo), args.modelos_dir)
        salida = predecir_tabla(modelo, leer_entrada(args.entrada), meta["nombre"])
    except EntradaInvalida as e:
        log.error("entrada no válida: %s", e)
        return 2
    destino = Path(args.salida)
    destino.parent.mkdir(parents=True, exist_ok=True)
    salida.to_csv(destino, index=False)
    log.info("%d predicciones (%s) escritas en %s", len(salida), meta["nombre"], destino)
    return 0
