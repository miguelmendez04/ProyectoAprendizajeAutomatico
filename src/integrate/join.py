"""Utilidades de unión: rezago temporal (sin fuga) y joins que registran filas perdidas."""
import pandas as pd


def rezagar_features(df: pd.DataFrame, llave: list[str], cols: list[str], rezago: int = 1) -> pd.DataFrame:
    """Mueve los valores del año t-rezago a la fila del año t y renombra con sufijo `_t<rezago>`.

    `llave` debe incluir `anio`. Así, al unir con el objetivo del año t, solo hay información previa.
    """
    out = df[llave + cols].copy()
    out["anio"] = out["anio"] + rezago
    return out.rename(columns={c: f"{c}_t{rezago}" for c in cols})


def unir_con_perdida(
    izq: pd.DataFrame,
    der: pd.DataFrame,
    llave: list[str],
    nombre: str,
    log: list[dict],
    how: str = "left",
) -> pd.DataFrame:
    """Une dos tablas y agrega al `log` cuántas filas de `izq` no encontraron pareja."""
    if der.duplicated(llave).any():
        raise ValueError(f"Join '{nombre}': llave duplicada en la tabla derecha {llave}")
    resultado = izq.merge(der, on=llave, how=how)
    # Filas de la izquierda cuya llave no existe en la derecha
    con_pareja = izq.merge(der[llave].drop_duplicates(), on=llave, how="inner")
    log.append(
        {
            "join": nombre,
            "filas_izq": len(izq),
            "filas_der": len(der),
            "filas_resultado": len(resultado),
            "filas_sin_match": len(izq) - len(con_pareja),
        }
    )
    return resultado
