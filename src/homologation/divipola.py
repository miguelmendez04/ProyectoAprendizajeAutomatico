"""Homologación de códigos y nombres de municipios contra DIVIPOLA (DANE).

Llave canónica: código de 5 dígitos como str con ceros a la izquierda.
Módulo central: lo usan todas las fuentes y se reutiliza en la tesis.
"""
import difflib
import re
import unicodedata
from pathlib import Path

import pandas as pd

# Valores que las fuentes usan para "sin información" en códigos de municipio.
CODIGOS_ESPECIALES = {0, 99999}
_PATRON_NUMERICO = re.compile(r"^\d+(\.0+)?$")


def normalizar_texto(s: str) -> str:
    """Mayúsculas, sin tildes y con espacios colapsados (para comparar nombres)."""
    sin_tildes = "".join(
        c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c)
    )
    return re.sub(r"\s+", " ", sin_tildes).strip().upper()


def normalizar_codigo(codigo) -> str | None:
    """Devuelve el código DANE de 5 dígitos, o None si es vacío o especial.

    Acepta str, int o float (las fuentes pierden los ceros a la izquierda).
    """
    if codigo is None or (isinstance(codigo, float) and pd.isna(codigo)):
        return None
    texto = str(codigo).strip()
    if not _PATRON_NUMERICO.match(texto):
        return None  # "SIN INFORMACION", vacío, etc.
    numero = int(float(texto))
    if numero in CODIGOS_ESPECIALES or len(str(numero)) > 5:
        return None
    return str(numero).zfill(5)


def cargar_divipola(path: Path) -> pd.DataFrame:
    """Lee el catálogo DIVIPOLA descargado (CSV de datos.gov.co `gdxc-w37w`).

    Devuelve cod_divipola_mpio, cod_depto, depto_norm y municipio_norm.
    """
    df = pd.read_csv(path, dtype=str)
    out = pd.DataFrame(
        {
            "cod_divipola_mpio": df["cod_mpio"].map(normalizar_codigo),
            "cod_depto": df["cod_dpto"].str.zfill(2),
            "depto_norm": df["dpto"].map(normalizar_texto),
            "municipio_norm": df["nom_mpio"].map(normalizar_texto),
        }
    )
    return out.dropna(subset=["cod_divipola_mpio"]).reset_index(drop=True)


def aplicar_cambios_codigo(df: pd.DataFrame, col: str, cambios: dict[str, str]) -> pd.DataFrame:
    """Reemplaza códigos viejos por vigentes (municipios nuevos o con código cambiado)."""
    out = df.copy()
    out[col] = out[col].replace(cambios)
    return out


def emparejar_por_nombre(
    df: pd.DataFrame,
    divipola: pd.DataFrame,
    col_municipio: str,
    col_depto: str,
    umbral: float = 0.9,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Asigna `cod_divipola_mpio` por nombre normalizado + departamento.

    Primero exacto; difflib solo como último recurso y dentro del mismo
    departamento. Devuelve (df con el código, reporte de emparejamientos no exactos).
    """
    # Índice departamento -> {municipio_norm: código}
    por_depto = {
        depto: dict(zip(g["municipio_norm"], g["cod_divipola_mpio"]))
        for depto, g in divipola.groupby("depto_norm")
    }
    codigos, reporte = [], []
    for nombre, depto in zip(df[col_municipio], df[col_depto]):
        candidatos = por_depto.get(normalizar_texto(depto), {})
        norm = normalizar_texto(nombre)
        if norm in candidatos:
            codigos.append(candidatos[norm])
            continue
        cercano = difflib.get_close_matches(norm, list(candidatos), n=1, cutoff=umbral)
        if cercano:
            score = difflib.SequenceMatcher(None, norm, cercano[0]).ratio()
            codigos.append(candidatos[cercano[0]])
            reporte.append((nombre, cercano[0], candidatos[cercano[0]], score))
        else:
            codigos.append(None)
    out = df.copy()
    out["cod_divipola_mpio"] = codigos
    columnas = ["nombre_original", "nombre_emparejado", "cod_divipola_mpio", "score"]
    return out, pd.DataFrame(reporte, columns=columnas)
