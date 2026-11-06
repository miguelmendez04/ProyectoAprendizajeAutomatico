"""Lector de los zips de TerriData (DNP) descargados a mano en `data/raw/terridata/`.

Cada `TerriData_Dim<N>.txt.zip` trae un TXT en formato largo separado por `|` (la dimensión 2, de
población, pesa ~1 GB descomprimida), así que se lee por bloques y se filtran en el acto las
entidades municipales y los indicadores del catálogo.
"""
import zipfile
from pathlib import Path

import pandas as pd

from src.homologation.divipola import normalizar_codigo, normalizar_texto
from src.transform.terridata import DIMENSIONES, clave_catalogo

COLUMNAS_LECTURA = ["Código Departamento", "Código Entidad", "Subcategoría", "Indicador", "Dato Numérico", "Año"]
BLOQUE_FILAS = 1_000_000


def _numero(texto: pd.Series) -> pd.Series:
    """'1.234.567,00' -> 1234567.0 (punto de miles, coma decimal)."""
    limpio = texto.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(limpio, errors="coerce")


def _filtrar_bloque(bloque: pd.DataFrame, catalogo: list[tuple], dimension: int) -> pd.DataFrame:
    """Deja filas municipales de indicadores del catálogo; añade `columna`."""
    entidad = bloque["Código Entidad"].str.strip()
    # municipio: 5 dígitos que no terminan en 000 (departamento) y no pertenecen a 'Colombia' (dpto 01)
    municipal = (entidad.str.len() == 5) & ~entidad.str.endswith("000") & (bloque["Código Departamento"].str.strip() != "01")
    bloque = bloque[municipal & (bloque["Dato Numérico"].str.strip() != "")]
    if bloque.empty:
        return bloque.assign(columna=pd.Series(dtype=str))
    # se normalizan solo los pares únicos (subcategoría, indicador), no todas las filas
    pares = bloque[["Subcategoría", "Indicador"]].drop_duplicates()
    asignado = {}
    for sub, ind in pares.itertuples(index=False):
        s, i = normalizar_texto(sub), normalizar_texto(ind)
        for columna, dim, cs, ci, prefijo in catalogo:
            if dim == dimension and s.startswith(cs) and (i.startswith(ci) if prefijo else i == ci):
                asignado[(sub, ind)] = columna
                break
    columna = pd.Series([asignado.get(k) for k in zip(bloque["Subcategoría"], bloque["Indicador"])], index=bloque.index)
    return bloque.assign(columna=columna).dropna(subset=["columna"])


def leer_terridata(directorio: Path, dimensiones: list[int] | None = None, notas: list[str] | None = None) -> pd.DataFrame:
    """Devuelve cod_divipola_mpio, anio, columna, valor para los indicadores del catálogo.

    Los zips que falten se omiten y se anotan; las llaves repetidas se promedian y se cuentan.
    """
    directorio = Path(directorio)
    catalogo = clave_catalogo()
    partes = []
    for dim in dimensiones or DIMENSIONES:
        ruta = directorio / f"TerriData_Dim{dim}.txt.zip"
        if not ruta.exists():
            if notas is not None:
                notas.append(f"TerriData: no existe {ruta.name} (Dim{dim}); sus indicadores no se integraron")
            continue
        with zipfile.ZipFile(ruta) as z, z.open(z.infolist()[0]) as f:
            lector = pd.read_csv(f, sep="|", dtype=str, usecols=COLUMNAS_LECTURA, keep_default_na=False,
                                 encoding="utf-8", chunksize=BLOQUE_FILAS)
            for bloque in lector:
                bloque = _filtrar_bloque(bloque, catalogo, dim)
                if not bloque.empty:
                    partes.append(pd.DataFrame({
                        "cod_divipola_mpio": bloque["Código Entidad"].str.strip().map(normalizar_codigo),
                        "anio": pd.to_numeric(bloque["Año"], errors="coerce"),
                        "columna": bloque["columna"],
                        "valor": _numero(bloque["Dato Numérico"].str.strip()),
                    }))
    if not partes:
        return pd.DataFrame(columns=["cod_divipola_mpio", "anio", "columna", "valor"])
    largo = pd.concat(partes, ignore_index=True).dropna(subset=["cod_divipola_mpio", "anio", "valor"])
    largo["anio"] = largo["anio"].astype(int)
    llave = ["cod_divipola_mpio", "anio", "columna"]
    repetidas = int(largo.duplicated(llave).sum())
    if repetidas and notas is not None:
        notas.append(f"TerriData: {repetidas} fila(s) duplicada(s) por municipio-año-indicador promediada(s)")
    return largo.groupby(llave, as_index=False)["valor"].mean()
