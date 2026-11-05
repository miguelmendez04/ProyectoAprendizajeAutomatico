"""Transformación de la matrícula de educación superior (SNIES / MEN).

Notas sobre los datos reales (verificadas en la Fase 0 y en la revisión final):
- En `5wck-szir` el conteo viene en una columna `matriculados_<año>` para todos los años.
- Hay una fila por género y semestre; los códigos de municipio pierden el cero inicial.
- `y9ga-zwzy` repite cada municipio-año (solo cambia la tildación del nombre) y `5wck-szir`
  repite filas completas en 2015-2018: ambos duplicados se eliminan antes de sumar.
"""
import pandas as pd

from src.homologation.divipola import normalizar_codigo

NIVELES = ["tecnica_profesional", "tecnologica", "universitaria", "especializacion", "maestria", "doctorado"]


def _col_matriculados(df: pd.DataFrame) -> str:
    """Nombre de la columna de conteo (`matriculados_2015`, etc.)."""
    cols = [c for c in df.columns if c.startswith("matriculados")]
    if not cols:
        raise KeyError("SNIES: no hay columna 'matriculados*'")
    return cols[0]


def transformar_snies_matricula(df: pd.DataFrame, notas: list[str] | None = None) -> pd.DataFrame:
    """Matrícula por municipio x año (`y9ga-zwzy`): suma de todos los niveles de formación.

    Se eliminan las filas que solo difieren en el nombre del municipio antes de sumar.
    Los descartes se cuentan en `notas`.
    """
    sin_nombre = [c for c in df.columns if c != "nombre_del_municipio"]
    unico = df.drop_duplicates(subset=sin_nombre)
    if notas is not None and len(unico) < len(df):
        notas.append(
            f"SNIES matrícula municipal: {len(df) - len(unico)} fila(s) duplicada(s) "
            "(solo difieren en el nombre) eliminada(s)"
        )
    out = pd.DataFrame(
        {
            "cod_divipola_mpio": unico["c_digo_delmunicipio"].map(normalizar_codigo),
            "anio": unico["a_o"].astype(int),
            "matriculados": unico[NIVELES].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1).astype(int),
            "ies_con_oferta": pd.to_numeric(unico["ies_con_oferta"], errors="coerce"),
        }
    )
    nulos = int(out["cod_divipola_mpio"].isna().sum())
    if notas is not None and nulos:
        notas.append(f"SNIES matrícula municipal: {nulos} fila(s) con código de municipio nulo o especial descartada(s)")
    out = out.dropna(subset=["cod_divipola_mpio"])
    return (
        out.groupby(["cod_divipola_mpio", "anio"], as_index=False)
        .agg(matriculados=("matriculados", "sum"), ies_con_oferta=("ies_con_oferta", "max"))
        .astype({"matriculados": int})
    )


def transformar_snies_catalogo(df: pd.DataFrame) -> pd.DataFrame:
    """Catálogo IES x programa: una fila por (cod_ies, cod_snies_programa).

    `cod_divipola_mpio` es el domicilio de la IES; `cod_divipola_mpio_programa` el municipio de oferta.
    Si el programa se ofrece en varios municipios se conserva aquel con mayor matrícula.
    """
    df = df.drop_duplicates()
    llave = ["c_digo_de_la_instituci_n", "c_digo_snies_delprograma"]
    peso = pd.to_numeric(df[_col_matriculados(df)], errors="coerce").fillna(0)
    # matrícula total del programa en cada municipio de oferta, para ordenar y quedarse con el mayor
    peso_mpio = peso.groupby([df[c] for c in llave + ["c_digo_del_municipio_programa"]]).transform("sum")
    df = df.assign(_peso=peso_mpio).sort_values("_peso", ascending=False, kind="stable")
    out = pd.DataFrame(
        {
            "cod_ies": df["c_digo_de_la_instituci_n"].str.strip(),
            "cod_snies_programa": df["c_digo_snies_delprograma"].str.strip(),
            "cod_divipola_mpio": df["c_digo_del_municipio_ies"].map(normalizar_codigo),
            "cod_divipola_mpio_programa": df["c_digo_del_municipio_programa"].map(normalizar_codigo),
            "sector": df["id_sector"],
            "caracter": df["id_caracter"],
        }
    )
    return out.drop_duplicates(["cod_ies", "cod_snies_programa"]).reset_index(drop=True)


def transformar_snies_matricula_programa(df: pd.DataFrame, notas: list[str] | None = None) -> pd.DataFrame:
    """Matrícula anual por IES x programa.

    Se suman los géneros dentro de cada semestre y se toma el máximo semestral como
    matrícula anual (sumar semestres contaría dos veces a los mismos estudiantes).
    Las filas completamente repetidas se eliminan antes de sumar.
    """
    unico = df.drop_duplicates()
    if notas is not None and len(unico) < len(df):
        notas.append(f"SNIES matrícula por programa: {len(df) - len(unico)} fila(s) duplicada(s) exactas eliminada(s)")
    d = pd.DataFrame(
        {
            "cod_ies": unico["c_digo_de_la_instituci_n"].str.strip(),
            "cod_snies_programa": unico["c_digo_snies_delprograma"].str.strip(),
            "anio": unico["a_o"].astype(int),
            "semestre": unico["semestre"],
            "matriculados": pd.to_numeric(unico[_col_matriculados(unico)], errors="coerce").fillna(0),
        }
    )
    por_semestre = d.groupby(["cod_ies", "cod_snies_programa", "anio", "semestre"], as_index=False)["matriculados"].sum()
    anual = por_semestre.groupby(["cod_ies", "cod_snies_programa", "anio"], as_index=False)["matriculados"].max()
    return anual.astype({"matriculados": int})
