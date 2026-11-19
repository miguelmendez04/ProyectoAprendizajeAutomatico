"""Banderas de calidad por IES-año: régimen especial, SENA e inconsistencia entre SPADIES y SNIES.

- `regimen_especial`: escuelas militares y policiales, cuya población (cadetes, alumnos con otro régimen de
  permanencia) no se comporta como la de una IES civil. Lista explícita de códigos SNIES (no prefijos: dentro de
  91xx hay instituciones civiles, como el Instituto Caro y Cuervo, 9108, y el ITEC, 9109).
- `es_sena`: el SENA (9110) es una institución de formación para el trabajo con una escala de otro orden; se marca
  aparte y no se excluye salvo que se pida (`excluir_sena`).
- `inconsistencia_fuentes`: el denominador de SPADIES (matrícula t-2) y la matrícula SNIES de t-2 deben ser del mismo
  orden de magnitud. Se marca cuando su cociente (`ratio_spadies_snies`) es mayor que `umbral_ratio` o menor que su
  inverso, o cuando SNIES no tiene dato. Si el año t-2 queda antes del inicio de la cobertura de SNIES (2015) la
  ausencia es estructural de la fuente, no una inconsistencia, y no se marca (si no, se perderían todos los años
  cuyo t-2 es anterior a 2015).
"""
import pandas as pd

# 2903 es la Escuela de Comunicaciones; 2906 la Escuela de Postgrados de la Fuerza Aérea; 2106 la Dirección Nacional de Escuelas
REGIMEN_ESPECIAL = frozenset({
    "2106", "2901", "2902", "2903", "2904", "2905", "2906", "3114", "3901", "3902",
    "9101", "9102", "9103", "9104", "9105", "9107",
})
CODIGO_SENA = "9110"


def es_regimen_especial(cod_ies: pd.Series) -> pd.Series:
    """True para las escuelas militares y policiales de la lista explícita `REGIMEN_ESPECIAL`."""
    return cod_ies.astype(str).isin(REGIMEN_ESPECIAL).astype(bool)


def es_sena(cod_ies: pd.Series) -> pd.Series:
    """True para el SENA (código 9110)."""
    return (cod_ies.astype(str) == CODIGO_SENA).astype(bool)


def calcular_banderas(
    objetivo: pd.DataFrame,
    snies_programa: pd.DataFrame,
    rezago: int,
    umbral_ratio: float = 3.0,
) -> pd.DataFrame:
    """Agrega a `objetivo` (cod_ies, anio, matriculados_t2_spadies) la matrícula SNIES t-2, el ratio y las banderas.

    Columnas agregadas: matriculados_t2_snies, ratio_spadies_snies, inconsistencia_fuentes, regimen_especial, es_sena.
    """
    mat = snies_programa.groupby(["cod_ies", "anio"], as_index=False)["matriculados"].sum()
    mat["anio"] = mat["anio"] + rezago  # la matrícula de t-rezago queda en la fila del año t
    mat = mat.rename(columns={"matriculados": "matriculados_t2_snies"})
    out = objetivo.merge(mat, on=["cod_ies", "anio"], how="left")

    snies = out["matriculados_t2_snies"].where(out["matriculados_t2_snies"] > 0)  # 0 cuenta como sin dato
    out["ratio_spadies_snies"] = out["matriculados_t2_spadies"] / snies
    primer_anio_snies = snies_programa["anio"].min()
    fuera_de_cobertura = (out["anio"] - rezago) < primer_anio_snies
    sin_dato = snies.isna() & ~fuera_de_cobertura
    ratio = out["ratio_spadies_snies"]
    out["inconsistencia_fuentes"] = ((ratio > umbral_ratio) | (ratio < 1 / umbral_ratio) | sin_dato).fillna(False).astype(bool)
    out["regimen_especial"] = es_regimen_especial(out["cod_ies"])
    out["es_sena"] = es_sena(out["cod_ies"])
    return out
