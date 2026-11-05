"""Reglas de calidad del dataset integrado (pandera)."""
import pandera.pandas as pa

# Llave que identifica una fila según la unidad de análisis
LLAVES = {
    "municipio": ["cod_divipola_mpio", "anio"],
    "ies": ["cod_ies", "anio"],
    "programa": ["cod_ies", "cod_snies_programa", "anio"],
}

PUNTAJE_MAX_SABER11 = 500  # el puntaje global de Saber 11 va de 0 a 500


def construir_esquema(unidad: str, max_nulos_pct: float, objetivo_obligatorio: bool = True) -> pa.DataFrameSchema:
    """Esquema con rangos, unicidad de la llave, tope de nulos por columna y consistencia.

    Las columnas se declaran opcionales (`required=False`): solo se validan las que existan.
    Con `objetivo_obligatorio=False` la tasa puede ser nula (dataset preliminar sin SPADIES).
    """
    if unidad not in LLAVES:
        raise ValueError(f"unidad desconocida: {unidad!r}")

    def nulos_dentro_del_tope(df) -> bool:
        proporcion = df.isna().mean()
        if not objetivo_obligatorio:
            proporcion = proporcion.drop("tasa_desercion", errors="ignore")
        return bool((proporcion <= max_nulos_pct).all())

    def matriculados_mayor_que_desertores(df) -> bool:
        if "matriculados" not in df.columns or "desertores" not in df.columns:
            return True
        return bool((df["matriculados"] >= df["desertores"]).all())

    def col(**kwargs):
        return pa.Column(regex=True, required=False, nullable=True, **kwargs)

    return pa.DataFrameSchema(
        columns={
            "tasa_desercion": pa.Column(
                float, pa.Check.in_range(0, 1), required=False, nullable=not objetivo_obligatorio
            ),
            "cod_divipola_mpio": pa.Column(
                checks=pa.Check.str_matches(r"^\d{5}$"), required=False, nullable=False
            ),  # código DANE de 5 dígitos con ceros a la izquierda
            r"^punt_global_(media|mediana)_t\d+$": col(checks=pa.Check.in_range(0, PUNTAJE_MAX_SABER11)),
            r"^pct_.*": col(checks=pa.Check.in_range(0, 1)),
            r"^nbi_pct$": col(checks=pa.Check.in_range(0, 100)),
            r"^(n_estudiantes|n_ies|matriculados|mpio_|poblacion).*": col(checks=pa.Check.ge(0)),
        },
        checks=[
            pa.Check(nulos_dentro_del_tope, error=f"alguna columna supera {max_nulos_pct:.0%} de nulos"),
            pa.Check(matriculados_mayor_que_desertores, error="matriculados < desertores"),
        ],
        unique=LLAVES[unidad],
        report_duplicates="all",
        strict=False,
        coerce=False,
    )


def validar(df, unidad: str, max_nulos_pct: float, objetivo_obligatorio: bool = True):
    """Valida `df`; lanza `pandera.errors.SchemaErrors` con todas las violaciones (lazy)."""
    return construir_esquema(unidad, max_nulos_pct, objetivo_obligatorio).validate(df, lazy=True)
