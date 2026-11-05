"""Construcción del dataset integrado a la unidad de análisis configurada.

Reglas de diseño:
- Objetivo (tasa de deserción SPADIES) en el año t; todas las variables explicativas con año
  entran rezagadas (t-1) para evitar fuga de datos. Las variables estáticas (sector, población,
  NBI) no llevan sufijo.
- Cada join registra las filas de la base sin pareja en el log.
"""
from pathlib import Path

import pandas as pd

from src.integrate.join import rezagar_features, unir_con_perdida

LLAVE_MPIO = ["cod_divipola_mpio", "anio"]


def _esqueleto(unidad: str, snies_mpio: pd.DataFrame, snies_programa: pd.DataFrame) -> pd.DataFrame:
    """Llaves de la unidad con objetivo nulo (para cuando SPADIES aún no se ha exportado)."""
    if unidad == "municipio":
        base = snies_mpio[LLAVE_MPIO]
    elif unidad == "ies":
        base = snies_programa[["cod_ies", "anio"]]
    else:
        base = snies_programa[["cod_ies", "cod_snies_programa", "anio"]]
    base = base.drop_duplicates().reset_index(drop=True)
    base["tasa_desercion"] = float("nan")
    return base


def _unir_rezagada(base, tabla, llave, cols, nombre, log):
    """Une `tabla` con sus columnas `cols` desplazadas un año (t-1)."""
    lag = rezagar_features(tabla, llave, cols, rezago=1)
    return unir_con_perdida(base, lag, llave, nombre, log)


def construir_dataset_desde_tablas(
    unidad: str,
    anios: list[int],
    saber_mpio: pd.DataFrame,
    snies_mpio: pd.DataFrame,
    snies_programa: pd.DataFrame,
    catalogo: pd.DataFrame,
    objetivo: pd.DataFrame | None = None,
    dane: pd.DataFrame | None = None,
    terridata: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Devuelve (dataset, log de joins) para la unidad `ies | programa | municipio`.

    Si `objetivo` es None se genera un esqueleto con `tasa_desercion` nula.
    """
    log: list[dict] = []
    base = _esqueleto(unidad, snies_mpio, snies_programa) if objetivo is None else objetivo.copy()
    base = base[base["anio"].isin(anios)].reset_index(drop=True)

    sede_ies = catalogo.drop_duplicates("cod_ies")[["cod_ies", "cod_divipola_mpio", "sector", "caracter"]]

    if unidad == "ies":
        base = unir_con_perdida(base, sede_ies, ["cod_ies"], "catalogo_ies", log)
        mat = snies_programa.groupby(["cod_ies", "anio"], as_index=False)["matriculados"].sum()
        mat = mat.rename(columns={"matriculados": "matriculados_ies"})
        base = _unir_rezagada(base, mat, ["cod_ies", "anio"], ["matriculados_ies"], "matricula_ies", log)
    elif unidad == "programa":
        if "cod_snies_programa" not in base.columns:
            raise ValueError(
                "unidad 'programa' requiere cod_snies_programa en el objetivo (export de SPADIES por programa)"
            )
        prog = catalogo[["cod_ies", "cod_snies_programa", "cod_divipola_mpio_programa", "sector", "caracter"]]
        prog = prog.rename(columns={"cod_divipola_mpio_programa": "cod_divipola_mpio"})
        base = unir_con_perdida(base, prog, ["cod_ies", "cod_snies_programa"], "catalogo_programa", log)
        mat = snies_programa.rename(columns={"matriculados": "matriculados_programa"})
        llave = ["cod_ies", "cod_snies_programa", "anio"]
        base = _unir_rezagada(base, mat, llave, ["matriculados_programa"], "matricula_programa", log)
    else:  # municipio
        if "cod_divipola_mpio" not in base.columns:
            # la tasa de cada IES se asigna a su municipio sede
            base = unir_con_perdida(base, sede_ies[["cod_ies", "cod_divipola_mpio"]], ["cod_ies"], "sede_ies", log)
            base = base.dropna(subset=["cod_divipola_mpio"])
            n_ies = ("cod_ies", "nunique")
        else:
            n_ies = ("tasa_desercion", "size")
        base = base.groupby(LLAVE_MPIO, as_index=False).agg(tasa_desercion=("tasa_desercion", "mean"), n_ies=n_ies)

    # Variables municipales (todas rezagadas t-1)
    cols_saber = [c for c in saber_mpio.columns if c not in LLAVE_MPIO]
    base = _unir_rezagada(base, saber_mpio, LLAVE_MPIO, cols_saber, "saber11", log)
    mpio = snies_mpio.rename(columns={"matriculados": "mpio_matriculados", "ies_con_oferta": "mpio_ies_con_oferta"})
    base = _unir_rezagada(base, mpio, LLAVE_MPIO, ["mpio_matriculados", "mpio_ies_con_oferta"], "snies_municipio", log)
    if terridata is not None:
        cols_td = [c for c in terridata.columns if c not in LLAVE_MPIO]
        base = _unir_rezagada(base, terridata, LLAVE_MPIO, cols_td, "terridata", log)
    if dane is not None:
        base = unir_con_perdida(base, dane, ["cod_divipola_mpio"], "dane", log)
    return base.reset_index(drop=True), log


def codigos_fuera_de_divipola(df: pd.DataFrame, divipola: pd.DataFrame) -> int:
    """Cuenta filas con código de municipio no nulo que no existe en el catálogo DIVIPOLA vigente."""
    codigos = df["cod_divipola_mpio"].dropna()
    return int((~codigos.isin(divipola["cod_divipola_mpio"])).sum())


def construir_dataset(
    cfg: dict,
    reportes: list[pd.DataFrame] | None = None,
    notas: list[str] | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Lee `data/raw/`, transforma cada fuente (guardando `data/interim/`) y construye el dataset.

    `reportes` recibe los emparejamientos no exactos de municipios; `notas` recibe observaciones
    de calidad (códigos fuera de DIVIPOLA, SPADIES ausente) para la bitácora.
    """
    from src.extract.dane import leer_manual
    from src.extract.spadies import leer_spadies
    from src.homologation.divipola import cargar_divipola
    from src.integrate.aggregate import agregar_saber11_municipio
    from src.transform.dane import transformar_dane, transformar_terridata
    from src.transform.saber11 import leer_saber11
    from src.transform.snies import (
        transformar_snies_catalogo,
        transformar_snies_matricula,
        transformar_snies_matricula_programa,
    )

    notas = notas if notas is not None else []
    raw, interim = Path(cfg["rutas"]["raw"]), Path(cfg["rutas"]["interim"])
    interim.mkdir(parents=True, exist_ok=True)

    divipola = cargar_divipola(raw / "dane" / "divipola.csv")
    saber = agregar_saber11_municipio(leer_saber11(raw / "saber11" / "saber11.csv"))
    crudo_mpio = pd.read_csv(raw / "snies" / "matricula_mpio.csv", dtype=str)
    crudo_prog = pd.read_csv(raw / "snies" / "matricula_programa.csv", dtype=str)
    snies_mpio = transformar_snies_matricula(crudo_mpio)
    snies_programa = transformar_snies_matricula_programa(crudo_prog)
    catalogo = transformar_snies_catalogo(crudo_prog)

    crudo_dane = leer_manual(cfg["manual"]["dane"])
    dane = transformar_dane(crudo_dane, divipola, reportes) if crudo_dane is not None else None
    crudo_td = leer_manual(cfg["manual"]["terridata"])
    terridata = transformar_terridata(crudo_td) if crudo_td is not None else None
    for nombre, tabla in (("DANE", dane), ("TerriData", terridata)):
        if tabla is None:
            notas.append(f"{nombre}: no hay archivo manual en data/raw; la fuente no se integró")

    for nombre, tabla in (("saber11_municipio", saber), ("snies_matricula_mpio", snies_mpio)):
        fuera = codigos_fuera_de_divipola(tabla, divipola)
        if fuera:
            notas.append(f"{nombre}: {fuera} filas con código de municipio que no está en DIVIPOLA vigente")
        tabla.to_csv(interim / f"{nombre}.csv", index=False)
    snies_programa.to_csv(interim / "snies_matricula_programa.csv", index=False)
    catalogo.to_csv(interim / "snies_catalogo.csv", index=False)

    archivo_spadies = Path(cfg["spadies"]["archivo"])
    if archivo_spadies.exists():
        objetivo = leer_spadies(archivo_spadies, cfg)
        objetivo.to_csv(interim / "spadies.csv", index=False)
    else:
        objetivo = None
        notas.append(f"SPADIES: no existe {archivo_spadies}; dataset preliminar con tasa_desercion nula")

    return construir_dataset_desde_tablas(
        cfg["unidad_analisis"], cfg["anios"], saber, snies_mpio, snies_programa, catalogo,
        objetivo=objetivo, dane=dane, terridata=terridata,
    )
