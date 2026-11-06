"""Banderas de calidad por IES-año: régimen especial e inconsistencia entre SPADIES y SNIES.

- `regimen_especial`: escuelas militares y policiales, cuya población (cadetes, alumnos con otro régimen de
  permanencia) no se comporta como la de una IES civil. Se identifican por código (2106, 29xx, 39xx, 910x) y por
  el nombre del catálogo oficial (ESCUELA MILITAR, POLICIA, NAVAL y la sigla FAC).
- `inconsistencia_fuentes`: el denominador de SPADIES (matrícula t-2) y la matrícula SNIES de t-2 deben ser del mismo
  orden de magnitud. Se marca cuando su cociente (`ratio_spadies_snies`) es mayor que `umbral_ratio` o menor que su
  inverso, o cuando SNIES no tiene dato. Si el año t-2 queda antes del inicio de la cobertura de SNIES (2015) la
  ausencia es estructural y no se marca (si no, se perderían todos los años cuyo t-2 es anterior a 2015).
"""
import numpy as np
import pandas as pd

from src.homologation.divipola import normalizar_texto

PATRON_CODIGOS = r"^(?:2106|29\d\d|39\d\d|910\d)$"
PATRON_NOMBRES = r"ESCUELA MILITAR|POLICIA|NAVAL|\bFAC\b"


def es_regimen_especial(cod_ies: pd.Series, nombres: pd.Series | None = None) -> pd.Series:
    """True para las escuelas militares y policiales (por código y, si se pasan, por nombre alineado con `cod_ies`)."""
    marca = cod_ies.astype(str).str.match(PATRON_CODIGOS)
    if nombres is not None:
        normal = nombres.map(lambda n: normalizar_texto(n) if isinstance(n, str) else "")
        marca = marca | normal.str.contains(PATRON_NOMBRES, regex=True)
    return marca.astype(bool)


def calcular_banderas(
    objetivo: pd.DataFrame,
    snies_programa: pd.DataFrame,
    rezago: int,
    umbral_ratio: float = 3.0,
    nombres: pd.Series | None = None,
) -> pd.DataFrame:
    """Agrega a `objetivo` (cod_ies, anio, matriculados_t2_spadies) la matrícula SNIES t-2, el ratio y las dos banderas.

    `nombres` es un Series cod_ies -> nombre oficial (catálogo de IES). Columnas agregadas:
    matriculados_t2_snies, ratio_spadies_snies, inconsistencia_fuentes, regimen_especial.
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
    nombre_ies = out["cod_ies"].map(nombres) if nombres is not None else None
    out["regimen_especial"] = es_regimen_especial(out["cod_ies"], nombre_ies)
    return out
