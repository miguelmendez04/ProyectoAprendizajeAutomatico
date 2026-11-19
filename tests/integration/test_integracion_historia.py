"""Pruebas de integración de `test_historia`: necesitan data/processed, data/interim o los resultados publicados (se omiten si faltan)."""
from pathlib import Path

import pandas as pd
import pytest

from desercion_ies.etl.integrate.historia import (
    COLUMNAS_HISTORIA_TODAS, agregar_historia,
)

DATASET = Path("data/processed/dataset_ies.csv")
INTERIM = Path("data/interim/spadies.csv")


@pytest.mark.integration
@pytest.mark.skipif(not DATASET.exists() or not INTERIM.exists(), reason="faltan data/processed o data/interim")
def test_la_historia_del_dataset_publicado_se_recalcula_solo_con_el_pasado():
    d = pd.read_csv(DATASET, dtype={"cod_ies": str})
    assert set(COLUMNAS_HISTORIA_TODAS) <= set(d.columns)
    previa = pd.read_csv(INTERIM, dtype={"cod_ies": str})[["cod_ies", "anio", "tasa_desercion"]]
    previa = previa[previa["anio"] < d["anio"].min()]
    recalculado = agregar_historia(d.drop(columns=COLUMNAS_HISTORIA_TODAS), ["cod_ies"], previa=previa)
    for c in COLUMNAS_HISTORIA_TODAS:
        pd.testing.assert_series_equal(d[c].astype(float), recalculado[c].astype(float), check_names=False)
    # desde el segundo año de la ventana, tasa_t1 publicada es la tasa de la fila del año anterior en el dataset
    previa_fila = d.set_index(["cod_ies", "anio"])["tasa_desercion"]
    esperado = pd.Series(list(zip(d["cod_ies"], d["anio"] - 1))).map(previa_fila.to_dict())
    dentro = (d["anio"] > d["anio"].min()).to_numpy()
    assert d["tasa_t1"].to_numpy()[dentro] == pytest.approx(esperado.to_numpy(float)[dentro], nan_ok=True)
    # las filas del dataset no cambian por la historia previa: la tasa publicada es la de SPADIES
    assert d["tasa_desercion"].notna().all()
