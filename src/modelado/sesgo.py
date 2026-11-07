"""Análisis de sesgo del dataset IES x año. Todo se calcula sobre copias: el dataset y los valores por defecto no cambian.

Cada función devuelve un DataFrame (o un valor simple) que usan `notebooks/02_sesgo.ipynb` y `reports/analisis_sesgo.md`.
Bloques: a) selección, b) supervivencia/panel, c) concentración geográfica, d) desbalance, e) faltantes, f) medición,
g) fuga temporal.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.modelado.geografia import CIUDADES_PRINCIPALES, RESTO, agregar_geografia, denominador


# --------------------------------------------------------------------------------------------- composición / contrastes
def distribucion(df: pd.DataFrame, columna: str) -> pd.DataFrame:
    """Filas, IES distintas y su porcentaje por categoría de `columna`."""
    g = df.groupby(columna, observed=True, dropna=False).agg(filas=("cod_ies", "size"), ies=("cod_ies", "nunique"))
    g["pct_filas"] = g["filas"] / g["filas"].sum()
    g["pct_ies"] = g["ies"] / df["cod_ies"].nunique()
    return g.reset_index()


def resumen_tasa(df: pd.DataFrame, por: str, valor: str = "tasa_desercion") -> pd.DataFrame:
    """Filas, IES y distribución de `valor` (mediana, media, desviación, p10, p90) por categoría de `por`."""
    g = df.groupby(por, observed=True, dropna=False)[valor]
    out = pd.DataFrame({
        "filas": g.size(), "ies": df.groupby(por, observed=True, dropna=False)["cod_ies"].nunique(),
        "mediana": g.median(), "media": g.mean(), "desv": g.std(), "p10": g.quantile(0.1), "p90": g.quantile(0.9),
    })
    return out.reset_index()


def contraste(a, b) -> dict:
    """Mann-Whitney entre dos muestras: p-valor y P(a > b) (probabilidad de superioridad; 0,5 = sin diferencia).

    Las filas de una misma IES no son independientes, así que el p-valor es solo orientativo.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return {"p_valor": np.nan, "prob_superioridad": np.nan}
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {"p_valor": float(p), "prob_superioridad": float(u / (len(a) * len(b)))}


# ------------------------------------------------------------------------------------------------ a) selección / cobertura
def universo_spadies(cfg: dict) -> pd.DataFrame:
    """Todas las IES-año con tasa anual de SPADIES en la ventana, sin aplicar ningún filtro.

    Trae `tasa_desercion`, `matriculados_t2_spadies` (denominador) y `posible_cierre`. Es el punto de partida de los
    filtros del pipeline (2.148 IES-año de 321 IES con la configuración vigente).
    """
    from src.extract.spadies import _anual_completo, leer_diferenciado, marcar_posible_cierre, tasas_semestrales, ultimo_reporte

    largo = leer_diferenciado(Path(cfg["spadies"]["archivo"]))
    semestrales = tasas_semestrales(largo)
    semestrales.loc[semestrales["tasa_desercion"] > 1, "tasa_desercion"] = np.nan  # igual que leer_spadies
    ultimo, global_ultimo = ultimo_reporte(largo)
    anual = _anual_completo(marcar_posible_cierre(semestrales, ultimo, global_ultimo))
    anios = cfg["anios"]
    return anual[anual["anio"].between(min(anios), max(anios))].reset_index(drop=True)


ETAPAS = ["incluida", "bajo denominador mínimo", "posible cierre", "inconsistencia de fuentes", "régimen especial"]


def clasificar_exclusion(universo: pd.DataFrame, exclusiones: pd.DataFrame, dataset: pd.DataFrame, minimo: float = 100) -> pd.DataFrame:
    """Agrega `estado` a cada IES-año del universo: incluida o el primer filtro que la saca, en el orden del pipeline.

    Orden: denominador mínimo, posible cierre y, por último, las banderas de calidad de `exclusiones_calidad.csv`
    (inconsistencia antes que régimen especial cuando están las dos). Verifica que las incluidas coincidan con `dataset`.
    """
    out = universo.copy()
    q = exclusiones[exclusiones["excluida"]].set_index(["cod_ies", "anio"])
    llaves = pd.MultiIndex.from_frame(out[["cod_ies", "anio"]])
    en_q = llaves.isin(q.index)
    estado = np.full(len(out), "incluida", dtype=object)
    inc = np.zeros(len(out), bool)
    reg = np.zeros(len(out), bool)
    if en_q.any():
        buscado = q.reindex(llaves[en_q])
        inc[en_q] = buscado["inconsistencia_fuentes"].to_numpy(dtype=bool)
        reg[en_q] = buscado["regimen_especial"].to_numpy(dtype=bool)
    estado[reg] = "régimen especial"
    estado[inc] = "inconsistencia de fuentes"
    estado[out["posible_cierre"].to_numpy(dtype=bool)] = "posible cierre"
    estado[(out["matriculados_t2_spadies"] < minimo).to_numpy()] = "bajo denominador mínimo"
    out["estado"] = estado
    incluidas = set(map(tuple, out.loc[out["estado"] == "incluida", ["cod_ies", "anio"]].to_numpy()))
    final = set(map(tuple, dataset[["cod_ies", "anio"]].to_numpy()))
    if incluidas != final:
        raise ValueError(
            f"las IES-año incluidas ({len(incluidas)}) no coinciden con el dataset ({len(final)}): "
            "revise que exclusiones_calidad.csv y el dataset sean de la misma corrida"
        )
    return out


def adjuntar_atributos(universo: pd.DataFrame, catalogo: pd.DataFrame) -> pd.DataFrame:
    """Agrega sector, carácter, municipio sede (con `sede_por_ies`), geografía y tramo de tamaño a cada IES-año."""
    from src.integrate.build import sede_por_ies

    sede, _ = sede_por_ies(catalogo)
    out = universo.merge(sede, on="cod_ies", how="left")
    out["log_matriculados_spadies_t2"] = np.log1p(out["matriculados_t2_spadies"])
    return agregar_geografia(out)


def cobertura_por_etapa(universo: pd.DataFrame, columna: str) -> pd.DataFrame:
    """% de filas e IES de cada categoría de `columna` en el universo, tras mínimo+cierre y en el dataset final."""
    etapas = {
        "sin filtro": universo,
        "tras mínimo y cierre": universo[~universo["estado"].isin(["bajo denominador mínimo", "posible cierre"])],
        "dataset final": universo[universo["estado"] == "incluida"],
    }
    partes = []
    for nombre, parte in etapas.items():
        d = distribucion(parte, columna)
        d.insert(0, "etapa", nombre)
        partes.append(d)
    return pd.concat(partes, ignore_index=True)


# ---------------------------------------------------------------------------------------------- b) supervivencia / panel
def patron_panel(df: pd.DataFrame, anios: tuple[int, int]) -> pd.DataFrame:
    """Por IES: años observados y patrón (completo, entra tarde, sale antes, entra tarde y sale antes, intermitente)."""
    inicio, fin = anios
    g = df.groupby("cod_ies")["anio"].agg(primero="min", ultimo="max", n="nunique")
    esperados = fin - inicio + 1
    seguido = (g["ultimo"] - g["primero"] + 1) == g["n"]
    patron = np.select(
        [g["n"] == esperados, ~seguido, (g["primero"] > inicio) & (g["ultimo"] < fin), g["primero"] > inicio, g["ultimo"] < fin],
        ["completo", "intermitente", "entra tarde y sale antes", "entra tarde", "sale antes"], default="otro",
    )
    g["patron"] = patron
    g["panel_completo"] = g["n"] == esperados
    return g.reset_index()


def comparar_panel(df: pd.DataFrame, anios: tuple[int, int]) -> pd.DataFrame:
    """IES, filas y tasa (mediana, media) de las IES con panel completo frente a las incompletas, y por patrón."""
    pan = patron_panel(df, anios)
    unido = df.merge(pan[["cod_ies", "n", "patron", "panel_completo"]], on="cod_ies")
    unido["grupo"] = np.where(unido["panel_completo"], "panel completo", "panel incompleto")
    return pd.concat([resumen_tasa(unido, "grupo"), resumen_tasa(unido[~unido["panel_completo"]], "patron")], ignore_index=True)


# -------------------------------------------------------------------------------------------- c) concentración geográfica
def concentracion_ciudades(df: pd.DataFrame) -> pd.DataFrame:
    """% de IES y de filas en Bogotá, Medellín, Cali y el resto del país (por municipio sede)."""
    d = distribucion(agregar_geografia(df), "ciudad_principal")
    orden = [*CIUDADES_PRINCIPALES.values(), RESTO]
    return d.set_index("ciudad_principal").reindex(orden).reset_index()


def dispersion_territorial(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """Para cada variable territorial: desviación dentro de las tres ciudades principales frente al resto del país.

    Cada municipio aporta un valor por año a todas sus IES-año: dentro de una ciudad la variable casi no varía entre IES.
    Se informa también cuántos municipios distintos hay en cada lado (la información territorial independiente).
    """
    d = agregar_geografia(df)
    ciudades = d["ciudad_principal"] != RESTO
    filas = []
    for c in columnas:
        dentro, fuera = d.loc[ciudades, c].dropna(), d.loc[~ciudades, c].dropna()
        filas.append({
            "variable": c, "desv_ciudades": dentro.std(), "desv_resto": fuera.std(),
            "razon_desv": dentro.std() / fuera.std() if fuera.std() else np.nan,
            "valores_distintos_ciudades": dentro.nunique(), "valores_distintos_resto": fuera.nunique(),
        })
    out = pd.DataFrame(filas)
    out.attrs["municipios_ciudades"] = int(d.loc[ciudades, "cod_divipola_mpio"].nunique())
    out.attrs["municipios_resto"] = int(d.loc[~ciudades, "cod_divipola_mpio"].nunique())
    return out


# ------------------------------------------------------------------------------------------------------- e) faltantes
def nulos_por_grupo(df: pd.DataFrame, columnas: list[str], grupo: str) -> pd.DataFrame:
    """% de nulos de cada columna por categoría de `grupo` (filas = categorías)."""
    return df.groupby(grupo, observed=True)[columnas].apply(lambda g: g.isna().mean())


def contraste_nulos(df: pd.DataFrame, columna: str, grupo: str) -> dict:
    """¿El nulo de `columna` depende de `grupo`? Chi-cuadrado sobre los años en que la variable existe.

    Se descartan los años sin ningún dato (bloques estructurales, que explican casi todos los nulos): lo que queda es la
    falta que no se explica por el año. Devuelve filas usadas, % de nulos y p-valor (orientativo: filas de una IES se repiten).
    """
    con_dato = df.groupby("anio")[columna].transform(lambda s: s.notna().any())
    parte = df[con_dato]
    nulo = parte[columna].isna()
    tabla = pd.crosstab(parte[grupo], nulo)
    if tabla.shape[0] < 2 or tabla.shape[1] < 2:
        return {"filas": len(parte), "pct_nulos": float(nulo.mean()) if len(parte) else np.nan, "p_valor": np.nan}
    return {"filas": len(parte), "pct_nulos": float(nulo.mean()), "p_valor": float(stats.chi2_contingency(tabla)[1])}


# --------------------------------------------------------------------------------------------------------- f) medición
def dispersion_por_tamano(df: pd.DataFrame) -> pd.DataFrame:
    """Dispersión de la tasa por tramo de denominador, frente a la desviación binomial de referencia sqrt(p(1-p)/n).

    La referencia supone que cada matriculado de t-2 es una prueba independiente con probabilidad p (la tasa media del
    tramo): es solo un orden de magnitud. `razon` = desviación observada / referencia; muy por encima de 1 indica
    variación real entre IES además del ruido del tamaño.
    """
    d = agregar_geografia(df)
    n = denominador(d)
    d = d.assign(_n=n)
    filas = []
    for tramo, parte in d.groupby("tramo_tamano", observed=True):
        p = parte["tasa_desercion"].mean()
        ref = float(np.sqrt(p * (1 - p) / parte["_n"]).mean())
        obs = float(parte["tasa_desercion"].std())
        q1, q3 = parte["tasa_desercion"].quantile([0.25, 0.75])
        filas.append({"tramo_tamano": tramo, "filas": len(parte), "ies": parte["cod_ies"].nunique(), "mediana": parte["tasa_desercion"].median(),
                      "desv": obs, "iqr": float(q3 - q1), "desv_binomial_ref": ref, "razon": obs / ref,
                      "pct_sobre_50": float((parte["tasa_desercion"] > 0.5).mean())})
    return pd.DataFrame(filas)


def ratio_fuera_de_rango(exclusiones: pd.DataFrame, dataset: pd.DataFrame, umbral: float = 3.0) -> pd.DataFrame:
    """Resumen de la inconsistencia SPADIES/SNIES: marcadas por tipo (ratio alto, bajo, sin SNIES) y la cola del ratio en el dataset."""
    marcadas = exclusiones[exclusiones["inconsistencia_fuentes"]]
    tipo = np.select([marcadas["ratio_spadies_snies"] > umbral, marcadas["ratio_spadies_snies"] < 1 / umbral],
                     ["ratio > umbral", "ratio < 1/umbral"], default="SNIES sin dato dentro de su cobertura")
    filas = [{"grupo": t, "filas": int((tipo == t).sum()), "mediana_tasa": float(marcadas.loc[tipo == t, "tasa_desercion"].median())}
             for t in pd.unique(tipo)]
    r = dataset["ratio_spadies_snies"].dropna() if "ratio_spadies_snies" in dataset else pd.Series(dtype=float)
    cola = dataset[(dataset["ratio_spadies_snies"] > umbral / 1.5) | (dataset["ratio_spadies_snies"] < 1.5 / umbral)] if len(r) else dataset.iloc[:0]
    filas.append({"grupo": f"en el dataset, ratio > {umbral / 1.5:.1f} o < {1.5 / umbral:.2f} (cerca del umbral)", "filas": len(cola),
                  "mediana_tasa": float(cola["tasa_desercion"].median()) if len(cola) else np.nan})
    return pd.DataFrame(filas)


# ------------------------------------------------------------------------------------------------------ g) fuga temporal
def _lag(df: pd.DataFrame, tabla: pd.DataFrame, columna_tabla: str, rezago: int, llaves: list[str]) -> pd.Series:
    """Valor de `columna_tabla` en el año `anio - rezago` para cada fila de `df` (por `llaves` sin año)."""
    t = tabla[[*llaves, "anio", columna_tabla]].rename(columns={"anio": "_origen", columna_tabla: "_valor"})
    pedido = df[[*llaves, "anio"]].assign(_origen=df["anio"] - rezago)
    return pedido.merge(t, on=[*llaves, "_origen"], how="left")["_valor"].set_axis(df.index)


def verificar_rezagos(
    df: pd.DataFrame, saber: pd.DataFrame, snies_mpio: pd.DataFrame, snies_programa: pd.DataFrame,
    terridata: pd.DataFrame | None = None, objetivo: pd.DataFrame | None = None, rezago_matricula: int = 2,
) -> pd.DataFrame:
    """Recalcula cada explicativa desde su fuente con el rezago permitido y cuenta las filas que no coinciden.

    Permitido: Saber 11 y TerriData en t-1 (`_t1`), matrícula SNIES (IES y municipio) en t-`rezago_matricula` y el
    denominador de SPADIES en la misma fila (ya es matrícula de t-2). Una explicativa que usara un año posterior (o el
    mismo año t) tendría discrepancias. Devuelve (columna, fuente, rezago, comparadas, discrepancias).
    """
    filas = []

    def comparar(columna, valores, fuente, rezago):
        if columna not in df.columns:
            return
        obs, esp = df[columna].astype(float), pd.Series(valores, index=df.index).astype(float)
        ambos = obs.notna() | esp.notna()
        malo = ambos & ~np.isclose(obs.fillna(-9e99), esp.fillna(-9e99), rtol=1e-9, atol=1e-9)
        filas.append({"columna": columna, "fuente": fuente, "rezago": rezago, "comparadas": int(ambos.sum()), "discrepancias": int(malo.sum())})

    for c in [c for c in saber.columns if c not in ("cod_divipola_mpio", "anio")]:
        comparar(f"{c}_t1", _lag(df, saber, c, 1, ["cod_divipola_mpio"]), "Saber 11", 1)
    for c, nombre in (("matriculados", "mpio_matriculados"), ("ies_con_oferta", "mpio_ies_con_oferta")):
        comparar(f"{nombre}_t{rezago_matricula}", _lag(df, snies_mpio, c, rezago_matricula, ["cod_divipola_mpio"]), "SNIES municipio", rezago_matricula)
    mat = snies_programa.groupby(["cod_ies", "anio"], as_index=False)["matriculados"].sum()
    comparar(f"matriculados_ies_t{rezago_matricula}", _lag(df, mat, "matriculados", rezago_matricula, ["cod_ies"]), "SNIES programas", rezago_matricula)
    if terridata is not None:
        for c in [c for c in terridata.columns if c not in ("cod_divipola_mpio", "anio")]:
            comparar(f"{c}_t1", _lag(df, terridata, c, 1, ["cod_divipola_mpio"]), "TerriData", 1)
    if objetivo is not None:
        den = df[["cod_ies", "anio"]].merge(objetivo[["cod_ies", "anio", "matriculados_t2_spadies"]], on=["cod_ies", "anio"], how="left")
        comparar("log_matriculados_spadies_t2", np.log1p(den["matriculados_t2_spadies"].to_numpy()), "SPADIES (denominador, t-2)", 2)
    return pd.DataFrame(filas)


POBLACION = ("poblacion_total", "pct_poblacion_rural", "pct_pob_15_19", "pct_pob_20_24")


def celdas_rellenadas_hacia_atras(largo: pd.DataFrame, anios: list[int], df: pd.DataFrame) -> pd.DataFrame:
    """Celdas municipio-año de población que `df` usa y que TerriData rellena con un valor de un año POSTERIOR.

    `transformar_terridata` arrastra hacia atrás las variables de población (`bfill`); si un municipio no tuviera dato en
    el año de origen del rezago, tomaría el de un año posterior. Devuelve, por variable, las celdas usadas, las que quedan
    nulas con solo arrastrar hacia adelante y las que el arrastre hacia atrás llena (0 = no hay fuga por este camino).
    """
    from src.transform.terridata import _derivar_poblacion

    ancho = largo.pivot_table(index=["cod_divipola_mpio", "anio"], columns="columna", values="valor", aggfunc="mean")
    ancho.columns.name = None
    ancho = _derivar_poblacion(ancho)
    cols = [c for c in POBLACION if c in ancho.columns]
    origen = range(min(anios) - 1, max(anios))
    malla = pd.MultiIndex.from_product(
        [ancho.index.get_level_values(0).unique(), sorted(set(ancho.index.get_level_values(1)) | set(origen))], names=["cod_divipola_mpio", "anio"]
    )
    ancho = ancho.reindex(malla).reset_index().sort_values(["cod_divipola_mpio", "anio"])
    adelante = ancho.copy()
    adelante[cols] = adelante.groupby("cod_divipola_mpio")[cols].ffill()
    completo = adelante.copy()
    completo[cols] = completo.groupby("cod_divipola_mpio")[cols].bfill()
    usadas = df[["cod_divipola_mpio", "anio"]].drop_duplicates().assign(anio=lambda d: d["anio"] - 1)
    filas = []
    for c in cols:
        m = usadas.merge(adelante[["cod_divipola_mpio", "anio", c]], how="left").merge(
            completo[["cod_divipola_mpio", "anio", c]].rename(columns={c: "_atras"}), how="left")
        filas.append({"variable": c, "celdas_usadas": len(m), "nulas_hacia_adelante": int(m[c].isna().sum()),
                      "rellenadas_con_anio_posterior": int((m[c].isna() & m["_atras"].notna()).sum())})
    return pd.DataFrame(filas)


PATRON_REZAGO = re.compile(r"_t(\d+)$")
# Explicativas sin sufijo de rezago que son legítimas: derivadas de variables ya rezagadas o indicadores de bloque de años
SIN_REZAGO_PERMITIDAS = {"falta_saber11", "falta_mdm", "falta_snies", "panel_incompleto"}


def explicativas_sin_rezago(columnas: list[str]) -> list[str]:
    """Explicativas cuyo nombre no declara un rezago `_t1`/`_t2` ni están en la lista de permitidas (candidatas a fuga)."""
    return [c for c in columnas if not PATRON_REZAGO.search(c) and c not in SIN_REZAGO_PERMITIDAS]


# ------------------------------------------------------------------------------------- sensibilidad al umbral (reconstruye)
def sensibilidad_umbral(cfg: dict, umbrales: tuple[float, ...] = (50, 100, 200)) -> pd.DataFrame:
    """Reconstruye el dataset en memoria con distintos `min_matriculados_t2` y resume cada uno. No escribe nada.

    Usa las tablas de `data/interim/` y el export de SPADIES. Con 100 debe reproducir el dataset publicado.
    """
    from src.extract.spadies import leer_spadies
    from src.integrate.build import construir_dataset_desde_tablas, depurar_terridata
    from src.transform.terridata import transformar_terridata

    interim = Path(cfg["rutas"]["interim"])
    cod = {"cod_divipola_mpio": str, "cod_ies": str, "sector": str, "caracter": str}
    saber = pd.read_csv(interim / "saber11_municipio.csv", dtype=cod)
    snies_mpio = pd.read_csv(interim / "snies_matricula_mpio.csv", dtype=cod)
    snies_prog = pd.read_csv(interim / "snies_matricula_programa.csv", dtype=cod)
    catalogo = pd.read_csv(interim / "snies_catalogo.csv", dtype={**cod, "cod_divipola_mpio_programa": str})
    td = transformar_terridata(pd.read_csv(interim / "terridata_largo.csv", dtype=cod), cfg["anios"])
    filas = []
    for u in umbrales:
        c = {**cfg, "min_matriculados_t2": u}
        objetivo = leer_spadies(c, [])
        df, _ = construir_dataset_desde_tablas(
            cfg["unidad_analisis"], cfg["anios"], saber, snies_mpio, snies_prog, catalogo, objetivo=objetivo, terridata=td,
            rezago_matricula=cfg.get("rezago_matricula", 2), excluir_inconsistencias=cfg.get("excluir_inconsistencias", True),
            excluir_regimen_especial=cfg.get("excluir_regimen_especial", True), excluir_sena=cfg.get("excluir_sena", False),
            umbral_ratio=cfg.get("umbral_ratio_inconsistencia", 3.0),
        )
        df = depurar_terridata(df, td, cfg["unidad_analisis"], cfg["max_nulos_pct"], [])
        d = agregar_geografia(df)
        filas.append({
            "umbral": u, "filas": len(df), "ies": df["cod_ies"].nunique(),
            "pct_ies_oficiales": float((d.drop_duplicates("cod_ies")["sector"] == "1").mean()),
            "pct_filas_bogota": float((d["ciudad_principal"] == "Bogotá").mean()),
            "mediana": df["tasa_desercion"].median(), "media": df["tasa_desercion"].mean(),
            "p95": df["tasa_desercion"].quantile(0.95), "desv": df["tasa_desercion"].std(),
            "filas_sobre_50": int((df["tasa_desercion"] > 0.5).sum()),
        })
    return pd.DataFrame(filas)
