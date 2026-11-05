"""Agregación de Saber 11 (estudiante) a municipio x año."""
import pandas as pd


def agregar_saber11_municipio(df: pd.DataFrame) -> pd.DataFrame:
    """Resume Saber 11 por municipio del colegio y año.

    `pct_estrato_bajo` (estratos 1-2) se calcula solo sobre estudiantes con estrato conocido.
    """
    d = df.dropna(subset=["cod_divipola_mpio"]).copy()
    d["estrato_bajo"] = (d["estrato"] <= 2).astype("float").where(d["estrato"].notna())
    d["colegio_oficial"] = d["colegio_oficial"].astype(float)
    return (
        d.groupby(["cod_divipola_mpio", "anio"], as_index=False)
        .agg(
            punt_global_media=("punt_global", "mean"),
            punt_global_mediana=("punt_global", "median"),
            pct_estrato_bajo=("estrato_bajo", "mean"),  # mean ignora nulos
            pct_colegio_oficial=("colegio_oficial", "mean"),
            n_estudiantes=("punt_global", "size"),
        )
    )
