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
            n_estudiantes=("punt_global", "count"),  # solo estudiantes con puntaje
        )
    )


def anios_saber_incompletos(saber: pd.DataFrame, umbral: float = 0.5) -> list[int]:
    """Años cuyo total de estudiantes es menor que `umbral` x la mediana anual.

    El dataset unificado no trae todos los periodos (p. ej. faltan 2018-2 o 2020-2): en esos años
    solo hay calendario B, una submuestra sesgada que no es comparable con los demás años.
    """
    totales = saber.groupby("anio")["n_estudiantes"].sum()
    return sorted(int(a) for a in totales[totales < umbral * totales.median()].index)


def excluir_anios_incompletos(saber: pd.DataFrame, notas: list[str] | None = None) -> pd.DataFrame:
    """Quita de la tabla Saber 11 los años incompletos (quedan nulos aguas abajo, no sesgados)."""
    malos = anios_saber_incompletos(saber)
    if malos and notas is not None:
        notas.append(
            f"Saber 11: cobertura incompleta en los años {malos} (falta el calendario A); se excluyen, "
            f"por lo que las variables Saber 11 de los años {[a + 1 for a in malos]} quedan nulas"
        )
    return saber[~saber["anio"].isin(malos)].reset_index(drop=True)
