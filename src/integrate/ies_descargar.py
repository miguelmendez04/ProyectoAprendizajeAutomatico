"""Lista de las IES cuyo export de SPADIES hay que descargar (reports/ies_a_descargar.csv).

SPADIES se exporta por IES. Criterio: todas las IES con sede en el Valle del Cauca, las de mayor
matrícula de las ciudades principales (cobertura territorial) y las de mayor matrícula del país
hasta completar `n_objetivo`. Si las obligatorias ya superan `n_objetivo` se conservan todas.
"""
from pathlib import Path

import pandas as pd

COLUMNAS = ["codigo_snies", "nombre_oficial", "municipio_sede", "departamento", "sector", "matricula"]
COD_DEPTO_VALLE = "76"
# código DIVIPOLA -> cuántas IES (las de mayor matrícula) se aseguran en cada ciudad
CAPITALES_PRINCIPALES = {"11001": 3, "05001": 3, "76001": 3, "08001": 3, "68001": 3}
OTRAS_CAPITALES = [
    "13001", "54001", "66001", "17001", "73001", "47001", "50001", "52001", "15001", "19001",
    "41001", "63001", "23001", "20001",
]  # Cartagena, Cúcuta, Pereira, Manizales, Ibagué, Santa Marta, Villavicencio, Pasto, Tunja, Popayán, Neiva, Armenia, Montería, Valledupar


def seleccionar_ies(
    ies: pd.DataFrame,
    catalogo: pd.DataFrame,
    matricula_programa: pd.DataFrame,
    divipola: pd.DataFrame,
    anio: int,
    n_objetivo: int = 60,
) -> pd.DataFrame:
    """Devuelve las IES a exportar con las columnas de `COLUMNAS`, ordenadas por matrícula.

    `ies`: catálogo oficial (c_digo_instituci_n, nombre_instituci_n, sector); `catalogo`: IES x programa
    con su municipio sede; `matricula_programa`: matrícula anual por programa. Solo se consideran las IES
    con matrícula en `anio`. La sede es el municipio con más programas (ver `sede_por_ies`).
    """
    from src.integrate.build import sede_por_ies

    sede, _ = sede_por_ies(catalogo)
    mat = matricula_programa[matricula_programa["anio"] == anio].groupby("cod_ies", as_index=False)["matriculados"].sum()
    mat = mat[mat["matriculados"] > 0].rename(columns={"matriculados": "matricula"})
    nombres = ies.rename(columns={"c_digo_instituci_n": "cod_ies", "nombre_instituci_n": "nombre_oficial"})
    nombres = nombres[["cod_ies", "nombre_oficial", "sector"]].drop_duplicates("cod_ies")
    nombres["sector"] = nombres["sector"].str.strip().str.lower().map({"oficial": "oficial", "privado": "privada"})
    geo = divipola.rename(columns={"municipio_norm": "municipio_sede", "depto_norm": "departamento"})
    geo = geo[["cod_divipola_mpio", "cod_depto", "municipio_sede", "departamento"]]

    todo = (
        mat.merge(sede[["cod_ies", "cod_divipola_mpio"]], on="cod_ies", how="inner")
        .merge(nombres, on="cod_ies", how="left")
        .merge(geo, on="cod_divipola_mpio", how="left")
        .sort_values(["matricula", "cod_ies"], ascending=[False, True])
        .reset_index(drop=True)
    )
    obligatoria = todo["cod_depto"] == COD_DEPTO_VALLE
    ranking = todo.groupby("cod_divipola_mpio").cumcount() + 1  # posición por matrícula dentro del municipio sede
    for cod, k in {**CAPITALES_PRINCIPALES, **{c: 1 for c in OTRAS_CAPITALES}}.items():
        obligatoria |= (todo["cod_divipola_mpio"] == cod) & (ranking <= k)

    faltan = max(n_objetivo - int(obligatoria.sum()), 0)
    resto = todo[~obligatoria].head(faltan)
    out = pd.concat([todo[obligatoria], resto]).sort_values(["matricula", "cod_ies"], ascending=[False, True])
    return out.rename(columns={"cod_ies": "codigo_snies"})[COLUMNAS].reset_index(drop=True)


def generar_ies_a_descargar(cfg: dict, anio: int | None = None, n_objetivo: int = 60) -> Path:
    """Lee `data/raw` + `data/interim` y escribe `<reports>/ies_a_descargar.csv`."""
    from src.homologation.divipola import cargar_divipola

    raw, interim = Path(cfg["rutas"]["raw"]), Path(cfg["rutas"]["interim"])
    anio = anio or max(cfg["anios"])
    ies = pd.read_csv(raw / "snies" / "ies_catalogo.csv", dtype=str)
    catalogo = pd.read_csv(interim / "snies_catalogo.csv", dtype=str)
    prog = pd.read_csv(interim / "snies_matricula_programa.csv", dtype={"cod_ies": str, "cod_snies_programa": str})
    divipola = cargar_divipola(raw / "dane" / "divipola.csv")
    out = seleccionar_ies(ies, catalogo, prog, divipola, anio, n_objetivo)
    destino = Path(cfg["rutas"].get("reports", "reports")) / "ies_a_descargar.csv"
    destino.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(destino, index=False, encoding="utf-8-sig")
    return destino


if __name__ == "__main__":
    from src.config import load_config

    print(generar_ies_a_descargar(load_config()))
