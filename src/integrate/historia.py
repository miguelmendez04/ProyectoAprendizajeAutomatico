"""Variables de historia de la tasa de cada entidad (IES): solo años anteriores al de la fila.

Para cada fila (entidad, año t) y usando únicamente filas del dataset con año < t:
- `tasa_t1`: la tasa del año t-1 de la misma entidad (nula si no hay fila de ese año exacto).
- `tasa_t2`: la tasa del año t-2.
- `tasa_media_previa`: la media de todas las tasas anteriores disponibles (nula si no hay ninguna).
- `n_anios_previos`: cuántas tasas anteriores hay (0 en el primer año de la entidad en el dataset).
- `falta_historia`: verdadero si no hay `tasa_t1`.

«Anterior» se define por el año de la fila: la tasa de t, y cualquiera posterior, nunca entra. Las filas son las del dataset
ya depurado (después de los filtros de calidad), de modo que `tasa_t1` coincide con la línea base ingenua de los experimentos.
A diferencia de `n_anios_ies` (que cuenta también los años posteriores), `n_anios_previos` no usa el futuro.
"""
import pandas as pd

COLUMNAS_HISTORIA = ["tasa_t1", "tasa_t2", "tasa_media_previa", "n_anios_previos", "falta_historia"]


def agregar_historia(df: pd.DataFrame, llave_entidad: list[str], objetivo: str = "tasa_desercion") -> pd.DataFrame:
    """Copia de `df` con las cinco columnas de historia al final. `llave_entidad`: columnas que identifican a la entidad
    (p. ej. `["cod_ies"]`); `anio` más la entidad debe ser único."""
    if df.duplicated(llave_entidad + ["anio"]).any():
        raise ValueError(f"{llave_entidad + ['anio']} no es único: la historia no está definida")
    out = df.copy()
    tasas = out[llave_entidad + ["anio", objetivo]]
    for k in (1, 2):  # la fila del año t-k de la misma entidad: se desplaza su año k posiciones hacia adelante
        previa = tasas.assign(anio=tasas["anio"] + k).rename(columns={objetivo: f"tasa_t{k}"})
        out = out.merge(previa, on=llave_entidad + ["anio"], how="left", validate="one_to_one")

    orden = out.sort_values(llave_entidad + ["anio"])
    tasa = orden[objetivo]
    valida = tasa.notna()
    por_entidad = [orden[c] for c in llave_entidad]
    # Se desplaza una posición dentro de cada entidad ANTES de acumular: la fila actual nunca entra en la suma ni en el conteo
    # (restarla después dejaría diferencias de redondeo que dependen del valor de la propia fila)
    n_previos = valida.astype(int).groupby(por_entidad).shift(1, fill_value=0).groupby(por_entidad).cumsum()
    suma_previa = tasa.fillna(0.0).groupby(por_entidad).shift(1, fill_value=0.0).groupby(por_entidad).cumsum()
    media = (suma_previa / n_previos.where(n_previos > 0)).reindex(out.index)
    out["tasa_media_previa"] = media
    out["n_anios_previos"] = n_previos.reindex(out.index).astype(int)
    out["falta_historia"] = out["tasa_t1"].isna()
    return out
