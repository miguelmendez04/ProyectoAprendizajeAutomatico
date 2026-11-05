"""Generación automática del diccionario de datos (reports/data_dictionary.md)."""
import re
from pathlib import Path

import pandas as pd

# nombre base de la variable -> (fuente, descripción)
_CATALOGO = {
    "tasa_desercion": ("SPADIES (MEN)", "Tasa de deserción anual como proporción 0-1. Variable objetivo"),
    "cod_divipola_mpio": ("DIVIPOLA (DANE)", "Código DANE del municipio, 5 dígitos con ceros a la izquierda"),
    "cod_ies": ("SNIES (MEN)", "Código de la institución de educación superior"),
    "cod_snies_programa": ("SNIES (MEN)", "Código SNIES del programa académico"),
    "anio": ("—", "Año del objetivo (t). Las variables explicativas son del año t-1"),
    "sector": ("SNIES (MEN)", "Sector de la IES (id_sector: oficial/privada)"),
    "caracter": ("SNIES (MEN)", "Carácter académico de la IES (id_caracter)"),
    "n_ies": ("SNIES (MEN)", "Número de IES con sede en el municipio incluidas en la tasa agregada"),
    "punt_global_media": ("Saber 11 (ICFES)", "Media del puntaje global de los estudiantes del municipio"),
    "punt_global_mediana": ("Saber 11 (ICFES)", "Mediana del puntaje global de los estudiantes del municipio"),
    "pct_estrato_bajo": ("Saber 11 (ICFES)", "Proporción de estudiantes de estratos 1-2 (entre los de estrato conocido)"),
    "pct_colegio_oficial": ("Saber 11 (ICFES)", "Proporción de estudiantes de colegio oficial"),
    "n_estudiantes": ("Saber 11 (ICFES)", "Número de estudiantes que presentaron Saber 11"),
    "mpio_matriculados": ("SNIES (MEN)", "Matrícula de educación superior en el municipio"),
    "mpio_ies_con_oferta": ("SNIES (MEN)", "IES con oferta en el municipio"),
    "matriculados_ies": ("SNIES (MEN)", "Matrícula anual de la IES (suma de sus programas)"),
    "matriculados_programa": ("SNIES (MEN)", "Matrícula anual del programa (máximo semestral)"),
    "poblacion": ("DANE", "Población del municipio"),
    "nbi_pct": ("DANE", "Necesidades básicas insatisfechas (%)"),
}
_REZAGO = re.compile(r"^(?P<base>.+)_t(?P<k>\d+)$")


def describir_columnas(columnas: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    """Devuelve (fuentes, descripciones) para las columnas conocidas; el resto queda como '—'."""
    fuentes, descripciones = {}, {}
    for c in columnas:
        m = _REZAGO.match(c)
        base, sufijo = (m["base"], f" (año t-{m['k']})") if m else (c, "")
        if base.startswith("td_"):
            fuentes[c], descripciones[c] = "TerriData (DNP)", f"Indicador territorial {base[3:]}{sufijo}"
        elif base in _CATALOGO:
            fuente, desc = _CATALOGO[base]
            fuentes[c], descripciones[c] = fuente, desc + sufijo
    return fuentes, descripciones


def generar_diccionario(
    df: pd.DataFrame,
    fuentes: dict[str, str],
    descripciones: dict[str, str],
    destino: Path,
) -> Path:
    """Escribe un markdown con variable, tipo, fuente, descripción, % de nulos y rango."""
    lineas = [
        "# Diccionario de datos",
        "",
        f"Filas: {len(df)} · Columnas: {df.shape[1]}",
        "",
        "| Variable | Tipo | Fuente | Descripción | % nulos | Rango |",
        "|---|---|---|---|---|---|",
    ]
    for c in df.columns:
        serie = df[c]
        if pd.api.types.is_numeric_dtype(serie) and not pd.api.types.is_bool_dtype(serie) and serie.notna().any():
            rango = f"{serie.min():.4g} – {serie.max():.4g}"
        else:
            rango = f"{serie.nunique()} valores distintos"
        lineas.append(
            f"| `{c}` | {serie.dtype} | {fuentes.get(c, '—')} | {descripciones.get(c, '—')} "
            f"| {serie.isna().mean() * 100:.1f}% | {rango} |"
        )
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return destino
