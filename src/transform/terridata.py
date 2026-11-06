"""Transformación de TerriData (DNP): catálogo de indicadores y tabla ancha municipio x año.

Cada indicador seleccionado se identifica por (dimensión, subcategoría, indicador). Hay tres modos de
llevarlo al año t de la fila:
- `anual`: el valor de ese año; si no hay dato queda nulo (no se arrastran valores viejos).
- `censo`: el último censo ya levantado (NBI e IPM existen solo en 2005 y 2018): se arrastra hacia adelante.
- `poblacion`: proyección DANE con base en el censo 2018, disponible desde 2018; para años anteriores se
  completa hacia atrás con el primer dato (la población municipal cambia lento, pero es una aproximación).
Las columnas salen con prefijo `td_`; el rezago t-1 lo aplica la integración.
"""
import pandas as pd

from src.homologation.divipola import normalizar_texto


def _ind(columna, dimension, subcategoria, indicador, descripcion, modo="anual", prefijo=False):
    return {"columna": columna, "dimension": dimension, "subcategoria": subcategoria, "indicador": indicador,
            "descripcion": descripcion, "modo": modo, "prefijo": prefijo}


INDICADORES = [
    # Población (Dim 2)
    _ind("poblacion_hombres", 2, "Población desagregada por sexo", "Población total de hombres", "Población de hombres", "poblacion"),
    _ind("poblacion_mujeres", 2, "Población desagregada por sexo", "Población total de mujeres", "Población de mujeres", "poblacion"),
    _ind("pct_poblacion_rural", 2, "Población desagregada por área", "Porcentaje población rural", "Porcentaje de población rural", "poblacion"),
    _ind("pct_pob_15_19", 2, "Población total por grupos quinquenales de edad", "Porcentaje de población entre los 15 y 19 años",
         "Porcentaje de población de 15 a 19 años", "poblacion", True),
    _ind("pct_pob_20_24", 2, "Población total por grupos quinquenales de edad", "Porcentaje de población entre los 20 y 24 años",
         "Porcentaje de población de 20 a 24 años", "poblacion", True),
    # Educación (Dim 4)
    _ind("cob_bruta_total", 4, "Acceso a la educación", "Cobertura bruta en educación - Total", "Cobertura bruta en educación (preescolar a media), %"),
    _ind("cob_bruta_media", 4, "Acceso a la educación", "Cobertura bruta en educación media", "Cobertura bruta en educación media, %"),
    _ind("cob_neta_media", 4, "Acceso a la educación", "Cobertura neta en educación media", "Cobertura neta en educación media, %"),
    _ind("cob_superior", 4, "Acceso a la educación", "Cobertura en educación superior", "Cobertura en educación superior, %"),
    _ind("transito_superior", 4, "Acceso a la educación", "Tasa de tránsito inmediato a la educación superior",
         "Tasa de tránsito inmediato a la educación superior, %"),
    _ind("desercion_escolar", 4, "Permanencia y rezago", "Tasa de deserción intra-anual en educación",
         "Deserción escolar intra-anual (transición a once), %", prefijo=True),
    _ind("repitencia_oficial", 4, "Permanencia y rezago", "Tasa de repitencia del sector oficial en educación",
         "Repitencia del sector oficial en básica y media, %", prefijo=True),
    # Finanzas públicas / desempeño fiscal (Dim 7)
    _ind("idf", 7, "Desempeño fiscal", "Indicador de desempeño fiscal", "Indicador de desempeño fiscal (DNP), puntos"),
    _ind("idf_transferencias", 7, "Desempeño fiscal", "% de ingresos que corresponden a transferencias", "Porcentaje de ingresos por transferencias"),
    _ind("idf_recursos_propios", 7, "Desempeño fiscal", "% de ingresos corrientes que corresponden a recursos propios",
         "Porcentaje de ingresos corrientes por recursos propios"),
    _ind("idf_inversion", 7, "Desempeño fiscal", "% del gasto total destinado a inversión", "Porcentaje del gasto destinado a inversión"),
    _ind("idf_ahorro", 7, "Desempeño fiscal", "Capacidad de ahorro", "Capacidad de ahorro (desempeño fiscal)"),
    _ind("idf_deuda", 7, "Desempeño fiscal", "Respaldo de la deuda", "Respaldo de la deuda (desempeño fiscal)"),
    # Desempeño municipal (Dim 8)
    _ind("mdm", 8, "Resumen MDM", "MDM", "Medición de desempeño municipal, puntos"),
    _ind("mdm_gestion", 8, "Resumen MDM", "Componente de gestión", "MDM: componente de gestión, puntos"),
    _ind("mdm_resultados", 8, "Resumen MDM", "Componente de resultados", "MDM: componente de resultados, puntos"),
    _ind("mdm_educacion", 8, "Componente de resultados", "Educación", "MDM: resultados en educación, puntos"),
    # Pobreza: NBI e IPM censales (Dim 14)
    _ind("nbi_pct", 14, "Necesidades Básicas Insatisfechas", "índice de Necesidades Básicas Insatisfechas - NBI",
         "Necesidades básicas insatisfechas (censo), %", "censo"),
    _ind("miseria_pct", 14, "Necesidades Básicas Insatisfechas", "Población en condición de miseria",
         "Población en condición de miseria (censo), %", "censo"),
    _ind("ipm_censo", 14, "Indicadores agregados de IPM", "índice de pobreza multidimensional - IPM (Censo)",
         "Índice de pobreza multidimensional (censo), %", "censo"),
    # Mercado laboral (Dim 16)
    _ind("cotizantes", 16, "Trabajadores cotizantes al sistema general de seguridad social", "Trabajadores cotizantes al sistema general",
         "Trabajadores cotizantes a seguridad social", prefijo=True),
    _ind("empresas_empleo_formal", 16, "Número de empresas generadoras de empleo formal", "Número de empresas generadoras de empleo formal",
         "Empresas generadoras de empleo formal por cada 10.000 habitantes", prefijo=True),
    _ind("ocupacion_formal_pct", 16, "Ocupación formal", "Porcentaje de personas ocupadas formalmente",
         "Porcentaje de personas ocupadas formalmente", prefijo=True),
    _ind("tgp", 16, "Tasa global de participación", "Tasa Global de Participación", "Tasa global de participación laboral, %"),
]

DIMENSIONES = sorted({i["dimension"] for i in INDICADORES})


def clave_catalogo() -> list[tuple]:
    """Catálogo con textos normalizados para comparar: (columna, dimension, subcat, indicador, prefijo)."""
    return [(i["columna"], i["dimension"], normalizar_texto(i["subcategoria"]), normalizar_texto(i["indicador"]), i["prefijo"])
            for i in INDICADORES]


def descripcion_columna(columna: str) -> str | None:
    """Descripción de un indicador por el nombre de columna (sin `td_` ni rezago)."""
    return next((i["descripcion"] for i in INDICADORES if i["columna"] == columna), None)


def _completar(ancho: pd.DataFrame, columnas: list[str], hacia_atras: bool) -> pd.DataFrame:
    """Arrastra hacia adelante (y, si se pide, hacia atrás) los valores de cada municipio."""
    if not columnas:
        return ancho
    ancho = ancho.sort_values(["cod_divipola_mpio", "anio"]).copy()
    ancho[columnas] = ancho.groupby("cod_divipola_mpio")[columnas].ffill()
    if hacia_atras:
        ancho[columnas] = ancho.groupby("cod_divipola_mpio")[columnas].bfill()
    return ancho


def transformar_terridata(largo: pd.DataFrame, anios: list[int]) -> pd.DataFrame:
    """Formato largo (cod_divipola_mpio, anio, columna, valor) -> ancho `td_*` por municipio x año.

    Devuelve los años `min(anios)-1 .. max(anios)-1`, que son los de origen del rezago t-1.
    """
    origen = range(min(anios) - 1, max(anios))
    if largo.empty:
        return pd.DataFrame(columns=["cod_divipola_mpio", "anio"])
    ancho = largo.pivot_table(index=["cod_divipola_mpio", "anio"], columns="columna", values="valor", aggfunc="mean")
    ancho.columns.name = None
    modos = {i["columna"]: i["modo"] for i in INDICADORES}
    if {"poblacion_hombres", "poblacion_mujeres"} & set(ancho.columns):
        for c in ("poblacion_hombres", "poblacion_mujeres"):
            if c not in ancho:
                ancho[c] = float("nan")
        ancho["poblacion_total"] = ancho["poblacion_hombres"] + ancho["poblacion_mujeres"]
        ancho = ancho.drop(columns=["poblacion_hombres", "poblacion_mujeres"])
        modos["poblacion_total"] = "poblacion"
    # malla completa municipio x año para poder arrastrar valores entre años
    municipios = ancho.index.get_level_values(0).unique()
    anios_malla = sorted(set(ancho.index.get_level_values(1)) | set(origen))
    malla = pd.MultiIndex.from_product([municipios, anios_malla], names=["cod_divipola_mpio", "anio"])
    ancho = ancho.reindex(malla).reset_index()
    cols = [c for c in ancho.columns if c not in ("cod_divipola_mpio", "anio")]
    ancho = _completar(ancho, [c for c in cols if modos.get(c) == "censo"], hacia_atras=False)
    ancho = _completar(ancho, [c for c in cols if modos.get(c) == "poblacion"], hacia_atras=True)
    ancho = ancho[ancho["anio"].isin(origen)].rename(columns={c: f"td_{c}" for c in cols})
    return ancho.sort_values(["cod_divipola_mpio", "anio"]).reset_index(drop=True)
