"""Lector de los exports de la consulta pública de SPADIES.

Fuente principal: `spadies_diferenciado.csv` (Analíticas → Deserción Anual → tipo de cálculo IES →
IES: Diferenciado). CSV con `;` y BOM UTF-8, columnas `IES;TIPO;1998-1;…;2024-2` y dos filas por IES
(`DESERTORES`, `MATRICULADOS`); el código SNIES de la IES viene en la columna `IES`. El total nacional
(`spadies_nacional.csv`, sin columna IES) solo sirve de contexto y de control de las sumas.

Tasa de deserción (definición de SPADIES): desertores del periodo t / matriculados del periodo t-2,
con t-2 = dos semestres antes (mismo semestre del año anterior). La anual es la razón de las sumas de
los dos semestres, es decir, el promedio de las tasas semestrales ponderado por su denominador.
"""
from pathlib import Path

import pandas as pd

from src.extract.common import write_metadata

PERIODICIDADES = {"anual", "semestre"}
REZAGO_PERIODOS = 2  # semestres entre los desertores y la matrícula del denominador
_PATRON_PERIODO = r"^(\d{4})-([12])$"
_FILAS = ("DESERTORES", "MATRICULADOS")


def _serie_a_largo(filas: pd.DataFrame, nombre: str) -> pd.DataFrame:
    """`filas` tiene índice DESERTORES/MATRICULADOS y una columna por periodo (una sola serie)."""
    for fila in _FILAS:
        if fila not in filas.index:
            raise KeyError(f"SPADIES ({nombre}): falta la fila {fila}")
    if filas.index.duplicated().any():
        raise ValueError(f"SPADIES ({nombre}): filas duplicadas")
    periodos = [str(c).strip() for c in filas.columns]
    malos = [p for p in periodos if not pd.Series([p]).str.match(_PATRON_PERIODO)[0]]
    if malos:
        raise ValueError(f"SPADIES ({nombre}): encabezados de periodo inválidos {malos[:3]}")
    valores = {}
    for fila in _FILAS:
        crudo = filas.loc[fila].astype("string").str.strip()
        num = pd.to_numeric(crudo, errors="coerce")
        # un vacío es dato faltante; un texto en la celda es un error del export
        if (num.isna() & crudo.notna() & (crudo != "")).any():
            raise ValueError(f"SPADIES ({nombre}): valores no numéricos en la fila {fila}")
        valores[fila.lower()] = num.to_numpy(dtype=float)
    largo = pd.DataFrame({"periodo": periodos, **valores})
    # SPADIES reporta 0 desertores antes de tener datos (1998-1, 1998-2): es dato faltante, no un cero real
    con_datos = largo["desertores"].gt(0)
    primero = con_datos.to_numpy().argmax() if con_datos.any() else len(largo)
    largo.loc[largo.index < primero, "desertores"] = float("nan")
    return largo


def leer_nacional(path: Path) -> pd.DataFrame:
    """Serie nacional (contexto y control de las sumas de IES): periodo, desertores, matriculados."""
    path = Path(path)
    crudo = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8-sig")
    if "TIPO" not in crudo.columns:
        raise KeyError(f"SPADIES ({path.name}): falta la columna TIPO")
    return _serie_a_largo(crudo.set_index(crudo["TIPO"].str.strip().str.upper()).drop(columns="TIPO"), path.name)


def leer_diferenciado(path: Path) -> pd.DataFrame:
    """Export "IES: Diferenciado" en formato largo: ies, periodo, desertores, matriculados."""
    path = Path(path)
    crudo = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8-sig")
    for col in ("IES", "TIPO"):
        if col not in crudo.columns:
            raise KeyError(f"SPADIES ({path.name}): falta la columna {col}")
    crudo["IES"] = crudo["IES"].str.strip().str.replace(r"\.0+$", "", regex=True)
    crudo["TIPO"] = crudo["TIPO"].str.strip().str.upper()
    tablas = []
    for ies, grupo in crudo.groupby("IES", sort=False):
        filas = grupo.set_index("TIPO").drop(columns="IES")
        try:
            tablas.append(_serie_a_largo(filas, f"{path.name}, IES {ies}").assign(ies=ies))
        except KeyError as e:
            raise KeyError(f"SPADIES: la IES {ies} no tiene todas las filas ({e.args[0]})") from e
    return pd.concat(tablas, ignore_index=True)[["ies", "periodo", "desertores", "matriculados"]]


def tasas_semestrales(largo: pd.DataFrame) -> pd.DataFrame:
    """Agrega anio, semestre, matriculados_t2 y tasa_desercion = desertores_t / matriculados_(t-2)."""
    out = largo.copy()
    partes = out["periodo"].str.extract(_PATRON_PERIODO)
    out["anio"], out["semestre"] = partes[0].astype(int), partes[1].astype(int)
    out["_idx"] = out["anio"] * 2 + out["semestre"] - 1
    denominador = out[["ies", "_idx", "matriculados"]].assign(_idx=lambda d: d["_idx"] + REZAGO_PERIODOS)
    denominador = denominador.rename(columns={"matriculados": "matriculados_t2"})
    out = out.merge(denominador, on=["ies", "_idx"], how="left")
    out["matriculados_t2"] = out["matriculados_t2"].where(out["matriculados_t2"] > 0)
    out["tasa_desercion"] = out["desertores"] / out["matriculados_t2"]
    return out.drop(columns="_idx")


def _anual_completo(semestrales: pd.DataFrame) -> pd.DataFrame:
    """Tasa anual con su denominador medio y, si las filas traen la bandera, `posible_cierre` (cualquier semestre)."""
    validas = semestrales.dropna(subset=["tasa_desercion"])
    agregados = dict(desertores=("desertores", "sum"), denominador=("matriculados_t2", "sum"), n=("semestre", "nunique"))
    if "posible_cierre" in validas.columns:
        agregados["posible_cierre"] = ("posible_cierre", "any")
    g = validas.groupby(["ies", "anio"], as_index=False).agg(**agregados)
    g = g[g["n"] == 2].rename(columns={"ies": "cod_ies"})
    g["tasa_desercion"] = g["desertores"] / g["denominador"]
    g["matriculados_t2_spadies"] = g["denominador"] / 2
    return g.drop(columns=["desertores", "denominador", "n"]).reset_index(drop=True)


def tasa_anual(semestrales: pd.DataFrame, min_denominador: float = 0) -> pd.DataFrame:
    """Tasa anual = suma de desertores / suma de matriculados t-2 de los dos semestres del año.

    Equivale al promedio de las tasas semestrales ponderado por su denominador. Sin los dos
    semestres no hay tasa anual. Con `min_denominador` se excluyen los años cuyo denominador
    anual (promedio de los dos semestres) es menor. Si las filas traen `posible_cierre`, el año
    queda marcado cuando alguno de sus semestres lo está.
    """
    g = _anual_completo(semestrales)
    g = g[g["matriculados_t2_spadies"] >= min_denominador]
    columnas = ["cod_ies", "anio", "tasa_desercion", "matriculados_t2_spadies"] + (["posible_cierre"] if "posible_cierre" in g.columns else [])
    return g[columnas].reset_index(drop=True)


VENTANA_CIERRE = 2  # semestres previos al último reporte que se consideran sospechosos


def _indice(anio, semestre):
    """Número correlativo de semestre (2015-1 -> 4030, 2015-2 -> 4031)."""
    return anio * 2 + semestre - 1


def _periodo_texto(idx: int) -> str:
    return f"{idx // 2}-{idx % 2 + 1}"


def ultimo_reporte(largo: pd.DataFrame) -> tuple[pd.Series, int]:
    """(último semestre con matrícula de cada IES, último semestre del archivo), como índices correlativos."""
    partes = largo["periodo"].str.extract(_PATRON_PERIODO).astype(int)
    idx = _indice(partes[0], partes[1])
    reportado = largo["matriculados"] > 0
    ultimo = idx[reportado].groupby(largo.loc[reportado, "ies"]).max()
    return ultimo, int(idx.max())


def marcar_posible_cierre(semestrales: pd.DataFrame, ultimo: pd.Series, global_ultimo: int) -> pd.DataFrame:
    """Agrega `posible_cierre`: el semestre cae entre los `VENTANA_CIERRE` previos al último reporte de la IES o es ese reporte.

    Solo cuenta si la IES dejó de reportar antes del final del archivo: una IES que sigue activa no se marca.
    """
    out = semestrales.copy()
    idx = _indice(out["anio"], out["semestre"])
    fin = out["ies"].map(ultimo)
    out["posible_cierre"] = ((fin < global_ultimo) & (idx >= fin - VENTANA_CIERRE) & (idx <= fin)).fillna(False).astype(bool)
    return out


def tabla_tasas_altas(semestrales: pd.DataFrame, largo: pd.DataFrame, umbral: float = 0.5,
                      min_matriculados_t2: float = 0, periodicidad: str = "anual") -> pd.DataFrame:
    """Filas con tasa mayor que `umbral` (tras el denominador mínimo), con su último reporte y la bandera de cierre."""
    ultimo, global_ultimo = ultimo_reporte(largo)
    marcadas = marcar_posible_cierre(semestrales, ultimo, global_ultimo)
    if periodicidad == "anual":
        tabla = _anual_completo(marcadas)
        llave = ["cod_ies", "anio"]
    else:
        tabla = marcadas.dropna(subset=["tasa_desercion"]).rename(columns={"ies": "cod_ies", "matriculados_t2": "matriculados_t2_spadies"})
        llave = ["cod_ies", "anio", "semestre"]
    tabla = tabla[(tabla["tasa_desercion"] > umbral) & (tabla["matriculados_t2_spadies"] >= min_matriculados_t2)].copy()
    tabla["ultimo_periodo"] = tabla["cod_ies"].map(ultimo).map(_periodo_texto)
    columnas = llave + ["tasa_desercion", "matriculados_t2_spadies", "ultimo_periodo", "posible_cierre"]
    return tabla[columnas].sort_values(llave).reset_index(drop=True)


MOTIVO_ANTIGUA = "dejó de reportar antes de {desde}"
MOTIVO_NUEVA = "empezó a reportar tan tarde que su primera tasa anual es posterior a {hasta}"
MOTIVO_SIN_TASA = "sin ninguna tasa anual (menos de dos semestres con denominador)"
MOTIVO_HUECO = "tiene tasa anual solo fuera de la ventana, con un vacío dentro de ella"


def clasificar_ies_sin_tasa(semestrales: pd.DataFrame, anios: list[int]) -> dict[str, list[str]]:
    """Motivo por el que cada IES de SPADIES no tiene ninguna tasa anual dentro de `anios` (sin mínimo de matrícula)."""
    desde, hasta = min(anios), max(anios)
    anual = tasa_anual(semestrales)
    con_tasa = set(anual.loc[anual["anio"].between(desde, hasta), "cod_ies"])
    motivos: dict[str, list[str]] = {}
    for ies in sorted(set(semestrales["ies"]) - con_tasa):
        anios_ies = anual.loc[anual["cod_ies"] == ies, "anio"]
        if anios_ies.empty:
            motivo = MOTIVO_SIN_TASA
        elif anios_ies.max() < desde:
            motivo = MOTIVO_ANTIGUA.format(desde=desde)
        elif anios_ies.min() > hasta:
            motivo = MOTIVO_NUEVA.format(hasta=hasta)
        else:
            motivo = MOTIVO_HUECO
        motivos.setdefault(motivo, []).append(ies)
    return motivos


def validar_contra_nacional(largo_ies: pd.DataFrame, nacional: pd.DataFrame) -> list[str]:
    """Notas de los periodos en que la suma de las IES supera el total nacional (error del export)."""
    suma = largo_ies.groupby("periodo")[["desertores", "matriculados"]].sum(min_count=1)
    total = nacional.set_index("periodo")[["desertores", "matriculados"]]
    notas = []
    for periodo, fila in suma.iterrows():
        if periodo not in total.index:
            continue
        for col in ("desertores", "matriculados"):
            if fila[col] > total.loc[periodo, col]:
                notas.append(
                    f"SPADIES: en {periodo} la suma de {col} de las IES ({fila[col]:.0f}) "
                    f"supera el total nacional ({total.loc[periodo, col]:.0f})"
                )
    return notas


def leer_spadies(cfg: dict, notas: list[str]) -> pd.DataFrame:
    """Objetivo para el ETL: cod_ies, anio, [semestre], tasa_desercion (0-1), matriculados_t2_spadies (denominador) y
    posible_cierre, según `spadies.periodicidad`. El denominador sirve para las banderas de calidad y no entra al dataset.

    Registra en `notas` la cobertura de IES, las inconsistencias contra el total nacional y las
    tasas semestrales mayores que 1 (desertores superiores a la matrícula t-2).
    """
    cfg_s = cfg["spadies"]
    periodicidad = cfg_s.get("periodicidad", "anual")
    if periodicidad not in PERIODICIDADES:
        raise ValueError(f"spadies.periodicidad inválida: {periodicidad!r}; debe ser una de {sorted(PERIODICIDADES)}")
    archivo = Path(cfg_s["archivo"])
    largo = leer_diferenciado(archivo)
    write_metadata(archivo, fuente="SPADIES (export manual, IES: Diferenciado)", url=str(archivo), filas=len(largo))
    notas.append(
        f"SPADIES: {largo['ies'].nunique()} IES leídas de {archivo.name} "
        f"(periodos {largo['periodo'].min()} a {largo['periodo'].max()})"
    )
    nacional = Path(cfg_s["nacional"])
    if nacional.exists():
        notas.extend(validar_contra_nacional(largo, leer_nacional(nacional)))
    else:
        notas.append(f"SPADIES: no existe {nacional}; no se validó la suma de IES contra el total nacional")

    semestrales = tasas_semestrales(largo)
    excesivas = int((semestrales["tasa_desercion"] > 1).sum())
    if excesivas:
        notas.append(f"SPADIES: {excesivas} tasa(s) semestral(es) mayor que 1 (desertores > matrícula t-2); se descartan")
        semestrales.loc[semestrales["tasa_desercion"] > 1, "tasa_desercion"] = float("nan")

    anios = cfg.get("anios")
    if anios:
        motivos = clasificar_ies_sin_tasa(semestrales, anios)
        total = largo["ies"].nunique()
        n_sin = sum(len(v) for v in motivos.values())
        if n_sin:
            detalle = "; ".join(f"{m} ({len(c)}: {', '.join(c)})" for m, c in motivos.items())
            notas.append(f"SPADIES: {n_sin} de {total} IES de SPADIES no tienen tasa anual en {min(anios)}–{max(anios)}: {detalle}")

    minimo = cfg.get("min_matriculados_t2", 100)
    ultimo, global_ultimo = ultimo_reporte(largo)
    semestrales = marcar_posible_cierre(semestrales, ultimo, global_ultimo)

    excluir_cierre = cfg.get("excluir_posible_cierre", True)

    def filas(minimo_t2: float, quitar_cierre: bool = True) -> pd.DataFrame:
        if periodicidad == "anual":
            fil = tasa_anual(semestrales, minimo_t2)
        else:
            fil = semestrales.dropna(subset=["tasa_desercion"])
            fil = fil[fil["matriculados_t2"] >= minimo_t2].rename(columns={"ies": "cod_ies", "matriculados_t2": "matriculados_t2_spadies"})
            fil = fil[["cod_ies", "anio", "semestre", "tasa_desercion", "matriculados_t2_spadies", "posible_cierre"]]
        if quitar_cierre and excluir_cierre:
            fil = fil[~fil["posible_cierre"]]
        return fil.reset_index(drop=True)

    def en_ventana(tabla: pd.DataFrame) -> pd.DataFrame:
        return tabla[tabla["anio"].between(min(anios), max(anios))] if anios else tabla

    # los efectos se cuentan en orden: primero el denominador mínimo y después los posibles cierres
    out = filas(minimo)
    if minimo > 0:
        antes, despues = en_ventana(filas(0, quitar_cierre=False)), en_ventana(filas(minimo, quitar_cierre=False))
        quitadas = len(antes) - len(despues)
        sin_filas = sorted(set(antes["cod_ies"]) - set(despues["cod_ies"]))
        notas.append(
            f"SPADIES: min_matriculados_t2={minimo} excluye {quitadas} IES-año ({quitadas / max(len(antes), 1):.1%} de {len(antes)}) "
            f"con denominador menor; {len(sin_filas)} IES quedan sin ninguna fila"
            + (f" ({', '.join(sin_filas[:15])}{', …' if len(sin_filas) > 15 else ''})" if sin_filas else "")
        )
    marcadas = en_ventana(filas(minimo, quitar_cierre=False))
    marcadas = marcadas[marcadas["posible_cierre"]]
    accion = "excluye" if excluir_cierre else "marca (no excluye)"
    sin_cierre = sorted(set(en_ventana(filas(minimo, quitar_cierre=False))["cod_ies"]) - set(en_ventana(out)["cod_ies"]))
    notas.append(
        f"SPADIES: posible cierre ({VENTANA_CIERRE} semestres previos al último reporte de una IES que dejó de reportar): "
        f"{accion} {len(marcadas)} fila(s) de {marcadas['cod_ies'].nunique()} IES"
        + (f" ({', '.join(sorted(marcadas['cod_ies'].unique())[:15])})" if len(marcadas) else "")
        + (f"; {len(sin_cierre)} IES quedan sin ninguna fila ({', '.join(sin_cierre[:15])})" if sin_cierre else "")
    )
    return out
