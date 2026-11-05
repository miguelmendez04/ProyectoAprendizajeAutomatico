"""Orquestador del ETL: extract -> transform -> integrate -> validate.

Uso: python pipeline.py [--config RUTA] [--saltar-extract]
"""
import argparse
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.integrate.build import construir_dataset, hay_exports_spadies
from src.validate.dictionary import describir_columnas, generar_diccionario
from src.validate.schema import validar

# Todo lo que sigue a esta marca en etl_log.md lo regenera el pipeline; lo anterior es manual.
MARCA_AUTOMATICA = "<!-- pipeline:inicio -->"


def _tabla_md(filas: list[dict]) -> str:
    """Tabla markdown a partir de una lista de dicts con las mismas claves."""
    columnas = list(filas[0])
    encabezado = "| " + " | ".join(columnas) + " |\n|" + "---|" * len(columnas)
    cuerpo = ["| " + " | ".join(str(f[c]) for c in columnas) + " |" for f in filas]
    return "\n".join([encabezado, *cuerpo])


def escribir_etl_log(destino: Path, log: list[dict], reportes: list[pd.DataFrame], notas: list[str],
                     unidad: str, filas: int) -> None:
    """Escribe la parte automática de la bitácora conservando las notas manuales previas."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    previo = destino.read_text(encoding="utf-8") if destino.exists() else "# Bitácora ETL\n"
    manual = previo.split(MARCA_AUTOMATICA)[0].rstrip() + "\n\n"

    bloque = [MARCA_AUTOMATICA, "", "## Última corrida del pipeline", "",
              f"Unidad de análisis: `{unidad}` · Filas del dataset: {filas}", "",
              "### Pérdidas por join (filas de la base sin pareja)", "", _tabla_md(log), ""]
    no_vacios = [r for r in reportes if not r.empty]
    if no_vacios:
        todo = pd.concat(no_vacios, ignore_index=True)
        bloque += [f"### Emparejamientos no exactos de municipios ({len(todo)})", "",
                   _tabla_md(todo.head(50).to_dict("records")), ""]
    if notas:
        bloque += ["### Observaciones de calidad", "", *[f"- {n}" for n in notas], ""]
    destino.write_text(manual + "\n".join(bloque), encoding="utf-8")


def main(config_path: str = "config/config.yaml", saltar_extract: bool = False) -> Path:
    """Ejecuta el pipeline completo y devuelve la ruta del dataset final."""
    cfg = load_config(config_path)
    unidad = cfg["unidad_analisis"]

    if not saltar_extract:
        from src.extract.dane import extract_divipola
        from src.extract.saber11 import extract_saber11
        from src.extract.snies import extract_ies_catalogo, extract_snies

        extract_divipola(cfg)
        extract_snies(cfg)
        extract_ies_catalogo(cfg)
        extract_saber11(cfg)

    reportes: list[pd.DataFrame] = []
    notas: list[str] = []
    df, log = construir_dataset(cfg, reportes, notas)

    # Sin export de SPADIES la tasa es nula: se valida el resto, no el objetivo
    hay_objetivo = hay_exports_spadies(Path(cfg["spadies"]["directorio"]))
    validar(df, unidad, cfg["max_nulos_pct"], objetivo_obligatorio=hay_objetivo)

    processed = Path(cfg["rutas"]["processed"])
    processed.mkdir(parents=True, exist_ok=True)
    salida = processed / f"dataset_{unidad}.csv"
    df.to_csv(salida, index=False)

    reports = Path(cfg["rutas"].get("reports", "reports"))
    fuentes, descripciones = describir_columnas(list(df.columns))
    generar_diccionario(df, fuentes, descripciones, reports / "data_dictionary.md")
    escribir_etl_log(reports / "etl_log.md", log, reportes, notas, unidad, len(df))
    return salida


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--saltar-extract", action="store_true")
    args = parser.parse_args()
    print(main(args.config, args.saltar_extract))
