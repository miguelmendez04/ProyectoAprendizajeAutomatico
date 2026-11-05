"""Transformación de DANE (población, NBI) y TerriData (indicadores territoriales).

Ambas fuentes se descargan a mano; este módulo define el formato de entrada esperado.
"""
import pandas as pd

from src.homologation.divipola import emparejar_por_nombre, normalizar_codigo


def transformar_terridata(df: pd.DataFrame) -> pd.DataFrame:
    """Export en formato largo (cod_municipio, anio, indicador, valor) -> ancho con prefijo `td_`."""
    d = pd.DataFrame(
        {
            "cod_divipola_mpio": df["cod_municipio"].map(normalizar_codigo),
            "anio": df["anio"].astype(int),
            "indicador": "td_" + df["indicador"].astype(str),
            "valor": pd.to_numeric(df["valor"], errors="coerce"),
        }
    ).dropna(subset=["cod_divipola_mpio"])
    ancho = d.pivot_table(index=["cod_divipola_mpio", "anio"], columns="indicador", values="valor", aggfunc="mean")
    ancho.columns.name = None
    return ancho.reset_index()


def transformar_dane(
    df: pd.DataFrame,
    divipola: pd.DataFrame | None = None,
    reportes: list[pd.DataFrame] | None = None,
    notas: list[str] | None = None,
) -> pd.DataFrame:
    """Población y NBI por municipio (sin año). Entrada: cod_mpio, municipio, departamento, poblacion, nbi.

    Si falta el código pero hay nombre + departamento y se pasa `divipola`, se resuelve por nombre;
    los emparejamientos no exactos (y los fallidos) se agregan a `reportes`. Las filas que no se
    pueden identificar se cuentan en `notas`. Si un municipio aparece más de una vez (p. ej. un
    archivo con población por año) se lanza ValueError: la tabla es estática por municipio.
    """
    codigo = df["cod_mpio"].map(normalizar_codigo)
    sin_codigo = codigo.isna()
    if divipola is not None and sin_codigo.any():
        resuelto, reporte = emparejar_por_nombre(df[sin_codigo], divipola, "municipio", "departamento")
        codigo[sin_codigo] = resuelto["cod_divipola_mpio"]
        if reportes is not None:
            reportes.append(reporte)
    out = pd.DataFrame(
        {
            "cod_divipola_mpio": codigo,
            "poblacion": pd.to_numeric(df["poblacion"], errors="coerce").astype("Int64"),
            "nbi_pct": pd.to_numeric(df["nbi"], errors="coerce"),
        }
    )
    descartadas = int(out["cod_divipola_mpio"].isna().sum())
    if notas is not None and descartadas:
        notas.append(f"DANE: {descartadas} fila(s) sin municipio identificable descartada(s)")
    out = out.dropna(subset=["cod_divipola_mpio"])
    if out["cod_divipola_mpio"].duplicated().any():
        raise ValueError("DANE: municipios duplicados; la tabla debe traer una fila por municipio (estática)")
    return out.reset_index(drop=True)
