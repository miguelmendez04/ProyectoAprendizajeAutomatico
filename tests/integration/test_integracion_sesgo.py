"""Pruebas de integración de `test_sesgo`: necesitan data/processed, data/interim o los resultados publicados (se omiten si faltan)."""

from pathlib import Path

import pandas as pd
import pytest

from desercion_ies.analysis import sesgo as S
from desercion_ies.features.particion import cargar_dataset, columnas_explicativas

RAIZ = Path(__file__).resolve().parents[2]


@pytest.mark.integration
def test_las_explicativas_por_defecto_declaran_su_rezago():
    """Cada explicativa por defecto del dataset tiene sufijo _t1/_t2 o es un indicador de bloque de años."""
    ruta = RAIZ / "data" / "processed" / "dataset_ies.csv"
    if not ruta.exists():
        pytest.skip("no hay dataset procesado")
    cfg = {"unidad_analisis": "ies", "rutas": {"processed": str(ruta.parent)}}
    df = cargar_dataset(cfg)
    assert S.explicativas_sin_rezago(columnas_explicativas(df)) == []


@pytest.mark.integration
def test_ninguna_explicativa_usa_un_anio_posterior_al_permitido_en_el_dataset_real():
    """Recalcula cada explicativa desde las tablas intermedias: Saber 11 y TerriData en t-1, matrícula SNIES en t-2."""
    interim = RAIZ / "data" / "interim"
    ruta = RAIZ / "data" / "processed" / "dataset_ies.csv"
    necesarios = [
        "saber11_municipio.csv",
        "snies_matricula_mpio.csv",
        "snies_matricula_programa.csv",
        "terridata_largo.csv",
        "spadies.csv",
    ]
    if not ruta.exists() or not all((interim / n).exists() for n in necesarios):
        pytest.skip("faltan data/processed o data/interim (se generan con `desercion etl`)")
    from desercion_ies.config import load_config
    from desercion_ies.etl.transform.terridata import transformar_terridata

    cfg = load_config(RAIZ / "configs" / "config.yaml")
    cod = {"cod_divipola_mpio": str, "cod_ies": str}
    df = pd.read_csv(ruta, dtype={**cod, "sector": str, "caracter": str})
    td = transformar_terridata(
        pd.read_csv(interim / "terridata_largo.csv", dtype=cod), cfg["anios"]
    )
    r = S.verificar_rezagos(
        df,
        pd.read_csv(interim / "saber11_municipio.csv", dtype=cod),
        pd.read_csv(interim / "snies_matricula_mpio.csv", dtype=cod),
        pd.read_csv(interim / "snies_matricula_programa.csv", dtype=cod),
        td,
        pd.read_csv(interim / "spadies.csv", dtype=cod),
        cfg.get("rezago_matricula", 2),
    )
    assert len(r) >= 30 and r["comparadas"].min() > 0
    assert r[r["discrepancias"] > 0].empty, r[r["discrepancias"] > 0].to_string()
