"""Construcción del dataset integrado a la unidad de análisis configurada.

Reglas de diseño:
- Objetivo (tasa de deserción SPADIES) en el año t; todas las variables explicativas con año
  entran rezagadas (t-1) para evitar fuga de datos. Las variables estáticas (sector, población,
  NBI) no llevan sufijo.
- Cada join registra las filas de la base sin pareja en el log.
"""
from pathlib import Path

import pandas as pd

from src.integrate.banderas import calcular_banderas
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


def sede_por_ies(catalogo: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Municipio sede de cada IES (el que concentra más programas) y las IES con más de una sede.

    SNIES registra un único código por institución; las seccionales pueden venir agregadas bajo
    la sede principal, de modo que asignar toda la IES a un municipio es una aproximación.
    """
    cat = catalogo.dropna(subset=["cod_divipola_mpio"])
    conteo = cat.groupby(["cod_ies", "cod_divipola_mpio", "sector", "caracter"], as_index=False).size()
    conteo = conteo.sort_values(["cod_ies", "size", "cod_divipola_mpio"], ascending=[True, False, True])
    sede = conteo.drop_duplicates("cod_ies")[["cod_ies", "cod_divipola_mpio", "sector", "caracter"]]
    multisede = cat.groupby("cod_ies")["cod_divipola_mpio"].nunique()
    return sede.reset_index(drop=True), sorted(multisede[multisede > 1].index)


def _unir_rezagada(base, tabla, llave, cols, nombre, log, rezago=1):
    """Une `tabla` con sus columnas `cols` desplazadas `rezago` años (t-1 por defecto)."""
    lag = rezagar_features(tabla, llave, cols, rezago=rezago)
    return unir_con_perdida(base, lag, llave, nombre, log)


def construir_dataset_desde_tablas(
    unidad: str,
    anios: list[int],
    saber_mpio: pd.DataFrame,
    snies_mpio: pd.DataFrame,
    snies_programa: pd.DataFrame,
    catalogo: pd.DataFrame,
    objetivo: pd.DataFrame | None = None,
    terridata: pd.DataFrame | None = None,
    rezago_matricula: int = 2,
    excluir_inconsistencias: bool = True,
    excluir_regimen_especial: bool = True,
    excluir_sena: bool = False,
    umbral_ratio: float = 3.0,
    registro: dict | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Devuelve (dataset, log de joins) para la unidad `ies | programa | municipio`.

    Las variables de matrícula SNIES se rezagan `rezago_matricula` años (2 = el denominador de la tasa
    de SPADIES); Saber 11 y TerriData, un año.

    Si `objetivo` es None se genera un esqueleto con `tasa_desercion` nula.

    Si el objetivo trae `matriculados_t2_spadies` se calculan las banderas `inconsistencia_fuentes`,
    `regimen_especial` y `es_sena` y el `ratio_spadies_snies` (ver `banderas.py`); las filas marcadas se excluyen según
    `excluir_inconsistencias`, `excluir_regimen_especial` y `excluir_sena` (esta última, apagada por defecto), antes de
    agregar por municipio. La tabla completa
    con las banderas se deja en `registro["banderas"]`. El denominador no entra al dataset.
    """
    log: list[dict] = []
    base = _esqueleto(unidad, snies_mpio, snies_programa) if objetivo is None else objetivo.copy()
    base = base[base["anio"].isin(anios)].reset_index(drop=True)
    if objetivo is not None and {"cod_ies", "matriculados_t2_spadies"} <= set(base.columns):
        base = calcular_banderas(base, snies_programa, rezago_matricula, umbral_ratio)
        if registro is not None:
            registro["banderas"] = base.copy()
        sale = pd.Series(False, index=base.index)
        if excluir_inconsistencias:
            sale |= base["inconsistencia_fuentes"]
        if excluir_regimen_especial:
            sale |= base["regimen_especial"]
        if excluir_sena:
            sale |= base["es_sena"]
        base = base[~sale].drop(columns=["matriculados_t2_spadies", "matriculados_t2_snies"]).reset_index(drop=True)

    sede_ies, _ = sede_por_ies(catalogo)

    if unidad == "ies":
        base = unir_con_perdida(base, sede_ies, ["cod_ies"], "catalogo_ies", log)
        mat = snies_programa.groupby(["cod_ies", "anio"], as_index=False)["matriculados"].sum()
        mat = mat.rename(columns={"matriculados": "matriculados_ies"})
        base = _unir_rezagada(base, mat, ["cod_ies", "anio"], ["matriculados_ies"], "matricula_ies", log, rezago_matricula)
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
        base = _unir_rezagada(base, mat, llave, ["matriculados_programa"], "matricula_programa", log, rezago_matricula)
    else:  # municipio
        if "cod_divipola_mpio" not in base.columns:
            # la tasa de cada IES se asigna a su municipio sede
            base = unir_con_perdida(base, sede_ies[["cod_ies", "cod_divipola_mpio"]], ["cod_ies"], "sede_ies", log)
            base = base.dropna(subset=["cod_divipola_mpio"])
        # No se conserva el número de IES: se calcula con el objetivo del año t y sería fuga
        llave = LLAVE_MPIO + [c for c in ("semestre",) if c in base.columns]
        base = base.groupby(llave, as_index=False).agg(tasa_desercion=("tasa_desercion", "mean"))

    # Variables municipales (todas rezagadas t-1)
    cols_saber = [c for c in saber_mpio.columns if c not in LLAVE_MPIO]
    base = _unir_rezagada(base, saber_mpio, LLAVE_MPIO, cols_saber, "saber11", log)
    mpio = snies_mpio.rename(columns={"matriculados": "mpio_matriculados", "ies_con_oferta": "mpio_ies_con_oferta"})
    base = _unir_rezagada(base, mpio, LLAVE_MPIO, ["mpio_matriculados", "mpio_ies_con_oferta"], "snies_municipio", log, rezago_matricula)
    if terridata is not None:
        cols_td = [c for c in terridata.columns if c not in LLAVE_MPIO]
        base = _unir_rezagada(base, terridata, LLAVE_MPIO, cols_td, "terridata", log)
    return base.reset_index(drop=True), log


def _cargar_terridata(cfg: dict, interim: Path, notas: list[str]) -> pd.DataFrame | None:
    """Lee los zips de TerriData (con caché en `interim`) y devuelve la tabla ancha municipio x año, o None."""
    from src.extract.terridata import leer_terridata
    from src.transform import terridata as catalogo_td
    from src.transform.terridata import transformar_terridata

    directorio = Path(cfg["terridata"]["directorio"])
    zips = sorted(directorio.glob("TerriData_Dim*.txt.zip")) if directorio.is_dir() else []
    if not zips:
        notas.append(f"TerriData: no hay zips TerriData_Dim<N>.txt.zip en {directorio}; la fuente no se integró")
        return None
    cache = interim / "terridata_largo.csv"
    # la lectura pesa ~1 min (la dimensión 2 son ~1 GB): se reutiliza mientras ni los zips ni el catálogo cambien
    vigente = cache.exists() and cache.stat().st_mtime > max(
        [z.stat().st_mtime for z in zips] + [Path(catalogo_td.__file__).stat().st_mtime]
    )
    if vigente:
        largo = pd.read_csv(cache, dtype={"cod_divipola_mpio": str})
    else:
        largo = leer_terridata(directorio, notas=notas)
        largo.to_csv(cache, index=False)
    if largo.empty:
        notas.append("TerriData: ningún indicador del catálogo con datos municipales; la fuente no se integró")
        return None
    return transformar_terridata(largo, cfg["anios"])


def depurar_terridata(df: pd.DataFrame, terridata: pd.DataFrame | None, unidad: str, max_nulos_pct: float,
                      notas: list[str]) -> pd.DataFrame:
    """Descarta las variables `td_*` con más nulos que el tope y anota las unidades sin municipio en TerriData.

    Una variable con tanta falta no aporta y haría fallar la validación. Se informa cuántas IES (o municipios)
    tienen un municipio sede que no existe en TerriData.
    """
    if terridata is None:
        return df
    td_cols = [c for c in df.columns if c.startswith("td_")]
    nulos = df[td_cols].isna().mean()
    altos = nulos[nulos > max_nulos_pct]
    if len(altos):
        detalle = ", ".join(f"{c} ({v:.0%})" for c, v in altos.items())
        notas.append(
            f"TerriData: {len(altos)} de {len(td_cols)} variable(s) descartada(s) por superar {max_nulos_pct:.0%} de nulos "
            f"tras el rezago: {detalle}"
        )
        df = df.drop(columns=list(altos.index))
    conocidos = set(terridata["cod_divipola_mpio"])
    unidad_col = "cod_ies" if unidad == "ies" else "cod_divipola_mpio"
    etiqueta = "IES" if unidad == "ies" else "municipios"
    if "cod_divipola_mpio" in df.columns and unidad_col in df.columns:
        por_unidad = df.drop_duplicates(unidad_col).set_index(unidad_col)["cod_divipola_mpio"]
        sin = por_unidad[~por_unidad.isin(conocidos)]
        detalle = ""
        if len(sin):
            detalle = " (" + ", ".join(f"{k}→{v}" for k, v in list(sin.items())[:15]) + (", …" if len(sin) > 15 else "") + ")"
        notas.append(
            f"TerriData: {len(sin)} de {len(por_unidad)} {etiqueta} sin municipio sede en TerriData{detalle}; "
            f"quedan con sus variables td_* nulas"
        )
    return df


def escribir_tasas_altas(cfg: dict, banderas: pd.DataFrame | None, destino: Path, notas: list[str]) -> pd.DataFrame:
    """Lista las IES-año con tasa > `umbral_tasa_alta` tras el denominador mínimo y las escribe en `destino`.

    Columnas: código, nombre (catálogo oficial), año, tasa, matrícula t-2 de SPADIES (denominador) y de SNIES, ratio,
    último periodo reportado, `posible_cierre`, `inconsistencia_fuentes` y `regimen_especial`. La lista se hace antes
    de excluir las filas marcadas, para poder verlas; `banderas` es la tabla de `construir_dataset_desde_tablas`.
    """
    from src.extract.spadies import leer_diferenciado, tabla_tasas_altas, tasas_semestrales

    umbral = cfg.get("umbral_tasa_alta", 0.5)
    anios = cfg["anios"]
    largo = leer_diferenciado(Path(cfg["spadies"]["archivo"]))
    semestrales = tasas_semestrales(largo)
    semestrales.loc[semestrales["tasa_desercion"] > 1, "tasa_desercion"] = float("nan")
    tabla = tabla_tasas_altas(semestrales, largo, umbral, cfg.get("min_matriculados_t2", 100),
                              cfg["spadies"].get("periodicidad", "anual"))
    tabla = tabla[tabla["anio"].between(min(anios), max(anios))]

    catalogo = Path(cfg["rutas"]["raw"]) / "snies" / "ies_catalogo.csv"
    if catalogo.exists():
        nombres = pd.read_csv(catalogo, dtype=str).rename(columns={"c_digo_instituci_n": "cod_ies", "nombre_instituci_n": "nombre"})
        tabla = tabla.merge(nombres[["cod_ies", "nombre"]].drop_duplicates("cod_ies"), on="cod_ies", how="left")
    else:
        tabla["nombre"] = pd.NA
    llave = [c for c in ("cod_ies", "anio", "semestre") if c in tabla.columns]
    extra = ["matriculados_t2_snies", "ratio_spadies_snies", "inconsistencia_fuentes", "regimen_especial", "es_sena"]
    if banderas is not None:
        tabla = tabla.merge(banderas[llave + extra], on=llave, how="left")
    primeras = ["cod_ies", "nombre", "anio"] + (["semestre"] if "semestre" in tabla.columns else [])
    resto = [c for c in tabla.columns if c not in primeras]
    tabla = tabla[primeras + resto]
    destino.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(destino, index=False, encoding="utf-8-sig")
    marcas = {c: tabla[c].fillna(False).astype(bool) for c in ("posible_cierre", "inconsistencia_fuentes", "regimen_especial", "es_sena") if c in tabla}
    union = pd.concat(marcas.values(), axis=1).any(axis=1) if marcas else pd.Series(False, index=tabla.index)
    opciones = {"inconsistencia_fuentes": cfg.get("excluir_inconsistencias", True), "regimen_especial": cfg.get("excluir_regimen_especial", True),
                "posible_cierre": cfg.get("excluir_posible_cierre", True), "es_sena": cfg.get("excluir_sena", False)}
    notas.append(
        f"SPADIES: {len(tabla)} IES-año de {tabla['cod_ies'].nunique()} IES con tasa > {umbral:.0%} tras el denominador mínimo "
        f"(lista en {destino.as_posix()}); marcadas: "
        + ", ".join(f"{c} {int(m.sum())}" for c, m in marcas.items())
        + f"; con alguna bandera {int(union.sum())}; se eliminan según la configuración "
        + f"{int(pd.concat([m for c, m in marcas.items() if opciones[c]], axis=1).any(axis=1).sum()) if any(opciones[c] for c in marcas) else 0}"
    )
    return tabla


def _anotar_banderas(cfg: dict, banderas: pd.DataFrame, nombres: pd.Series | None, destino: Path, notas: list[str]) -> None:
    """Escribe las IES-año marcadas (con su motivo) y anota cuántas se excluyen por cada bandera."""
    umbral = cfg.get("umbral_ratio_inconsistencia", 3.0)
    excl_inc = cfg.get("excluir_inconsistencias", True)
    excl_reg = cfg.get("excluir_regimen_especial", True)
    excl_sena = cfg.get("excluir_sena", False)
    marcadas = banderas[banderas["inconsistencia_fuentes"] | banderas["regimen_especial"] | banderas["es_sena"]].copy()
    if nombres is not None:
        marcadas.insert(1, "nombre", marcadas["cod_ies"].map(nombres))
    marcadas["excluida"] = ((marcadas["inconsistencia_fuentes"] & excl_inc) | (marcadas["regimen_especial"] & excl_reg)
                            | (marcadas["es_sena"] & excl_sena))
    destino.parent.mkdir(parents=True, exist_ok=True)
    marcadas.to_csv(destino, index=False, encoding="utf-8-sig")
    n = len(banderas)
    ratio = banderas["ratio_spadies_snies"]
    sin_snies = int(banderas["matriculados_t2_snies"].isna().sum())
    notas.append(
        f"Inconsistencia SPADIES-SNIES (ratio > {umbral:g} o < 1/{umbral:g}, o SNIES sin dato dentro de su cobertura): "
        f"{int(banderas['inconsistencia_fuentes'].sum())} de {n} IES-año marcadas ({int((ratio > umbral).sum())} con ratio alto, "
        f"{int((ratio < 1 / umbral).sum())} con ratio bajo; {sin_snies} sin matrícula SNIES t-{cfg.get('rezago_matricula', 2)}, "
        f"de las cuales las de años anteriores a la cobertura de SNIES no se marcan); "
        f"{'se excluyen' if excl_inc else 'no se excluyen'}"
    )
    reg = banderas[banderas["regimen_especial"]]
    notas.append(
        f"Régimen especial (escuelas militares y policiales): {len(reg)} IES-año de {reg['cod_ies'].nunique()} IES "
        f"({', '.join(sorted(reg['cod_ies'].unique()))}); {'se excluyen' if excl_reg else 'no se excluyen'}"
    )
    sena = banderas[banderas["es_sena"]]
    notas.append(
        f"SENA (9110): {len(sena)} IES-año marcados con es_sena; {'se excluyen' if excl_sena else 'no se excluyen (excluir_sena = false)'}"
    )
    quitadas = int(marcadas["excluida"].sum())
    notas.append(f"Exclusiones por calidad (detalle en {destino.as_posix()}): {quitadas} IES-año de {n} (unión de las banderas activas)")


def hay_export_spadies(cfg: dict) -> bool:
    """True si existe el export de SPADIES configurado (`spadies.archivo`)."""
    return Path(cfg["spadies"]["archivo"]).is_file()


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
    from src.extract.spadies import leer_spadies
    from src.homologation.divipola import cargar_divipola
    from src.homologation.divipola import aplicar_cambios_codigo
    from src.integrate.aggregate import agregar_saber11_municipio, excluir_anios_incompletos
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
    # Códigos de municipio que cambiaron: se aplican a cada fuente antes de agregar/unir
    cambios = {str(k).zfill(5): str(v).zfill(5) for k, v in cfg.get("cambios_codigo", {}).items()}

    def con_cambios(tabla, col="cod_divipola_mpio"):
        return aplicar_cambios_codigo(tabla, col, cambios) if cambios and tabla is not None else tabla

    saber_est = con_cambios(leer_saber11(raw / "saber11" / "saber11.csv", notas=notas))
    sin_codigo = int(saber_est["cod_divipola_mpio"].isna().sum())
    if sin_codigo:
        notas.append(f"Saber 11: {sin_codigo} fila(s) sin código de municipio válido descartada(s)")
    saber = excluir_anios_incompletos(agregar_saber11_municipio(saber_est), notas)
    crudo_mpio = pd.read_csv(raw / "snies" / "matricula_mpio.csv", dtype=str)
    crudo_prog = pd.read_csv(raw / "snies" / "matricula_programa.csv", dtype=str)
    snies_mpio = con_cambios(transformar_snies_matricula(crudo_mpio, notas))
    snies_programa = transformar_snies_matricula_programa(crudo_prog, notas)
    catalogo = transformar_snies_catalogo(crudo_prog)
    catalogo = con_cambios(con_cambios(catalogo), "cod_divipola_mpio_programa")

    terridata = con_cambios(_cargar_terridata(cfg, interim, notas))

    for nombre, tabla in (("saber11_municipio", saber), ("snies_matricula_mpio", snies_mpio)):
        fuera = codigos_fuera_de_divipola(tabla, divipola)
        if fuera:
            notas.append(f"{nombre}: {fuera} filas con código de municipio que no está en DIVIPOLA vigente")
        tabla.to_csv(interim / f"{nombre}.csv", index=False)
    snies_programa.to_csv(interim / "snies_matricula_programa.csv", index=False)
    catalogo.to_csv(interim / "snies_catalogo.csv", index=False)

    if hay_export_spadies(cfg):
        objetivo = leer_spadies(cfg, notas)
        objetivo.to_csv(interim / "spadies.csv", index=False)
        _, multisede = sede_por_ies(catalogo)
        con_tasa = sorted(set(objetivo["cod_ies"]) & set(multisede))
        if con_tasa:
            notas.append(
                f"IES con programas en más de un municipio ({len(con_tasa)}: {', '.join(con_tasa)}): SPADIES las "
                "reporta agregadas y se asignan al municipio con más programas; el contexto territorial es una aproximación"
            )
    else:
        objetivo = None
        notas.append(f"SPADIES: no existe {cfg['spadies']['archivo']}; dataset preliminar con tasa_desercion nula")

    catalogo_ies = raw / "snies" / "ies_catalogo.csv"
    nombres_ies = None  # solo para escribir los nombres en los reportes
    if catalogo_ies.exists():
        nombres_ies = pd.read_csv(catalogo_ies, dtype=str).drop_duplicates("c_digo_instituci_n").set_index("c_digo_instituci_n")["nombre_instituci_n"]
    registro: dict = {}
    reports = Path(cfg["rutas"].get("reports", "reports"))
    df, log = construir_dataset_desde_tablas(
        cfg["unidad_analisis"], cfg["anios"], saber, snies_mpio, snies_programa, catalogo,
        objetivo=objetivo, terridata=terridata, rezago_matricula=cfg.get("rezago_matricula", 2),
        excluir_inconsistencias=cfg.get("excluir_inconsistencias", True),
        excluir_regimen_especial=cfg.get("excluir_regimen_especial", True), excluir_sena=cfg.get("excluir_sena", False),
        umbral_ratio=cfg.get("umbral_ratio_inconsistencia", 3.0), registro=registro,
    )
    df = depurar_terridata(df, terridata, cfg["unidad_analisis"], cfg["max_nulos_pct"], notas)
    if "banderas" in registro:
        _anotar_banderas(cfg, registro["banderas"], nombres_ies, reports / "exclusiones_calidad.csv", notas)
        escribir_tasas_altas(cfg, registro["banderas"], reports / "tasas_altas.csv", notas)
    return df, log
