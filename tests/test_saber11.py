"""Pruebas del transform de Saber 11 (esquema unificado kgxf-xxbe)."""
import pandas as pd
import pytest

from src.transform.saber11 import COLUMN_MAP, estandarizar_saber11

COLUMNAS_CANONICAS = ["anio", "cod_divipola_mpio", "punt_global", "estrato", "colegio_oficial"]


@pytest.fixture
def crudo():
    return pd.read_csv("tests/fixtures/saber11_muestra.csv", dtype=str)


def test_estandariza_fixture_real(crudo):
    out = estandarizar_saber11(crudo)
    assert list(out.columns) == COLUMNAS_CANONICAS
    assert len(out) == len(crudo)
    assert set(out["anio"]) == {2015, 2016}  # periodo 20152 -> 2015, 20161 -> 2016
    assert out["cod_divipola_mpio"].str.len().eq(5).all()
    assert out["estrato"].dropna().between(1, 6).all()
    assert out["punt_global"].dtype == "float64"
    assert out["colegio_oficial"].dtype == "bool"


def test_columna_faltante_error_claro(crudo):
    with pytest.raises(KeyError) as e:
        estandarizar_saber11(crudo.drop(columns=["punt_global"]), periodo="20152")
    assert "20152" in str(e.value) and "punt_global" in str(e.value)


def test_codigos_especiales_a_nulo_y_ceros_a_la_izquierda(crudo):
    df = crudo.head(3).copy()
    df["cole_cod_mcpio_ubicacion"] = ["0", "99999", "5001"]
    out = estandarizar_saber11(df)
    assert out["cod_divipola_mpio"].isna().tolist() == [True, True, False]
    assert out["cod_divipola_mpio"].iloc[2] == "05001"


def test_sin_estrato_y_nulos_quedan_nulos(crudo):
    df = crudo.head(3).copy()
    df["fami_estratovivienda"] = ["Sin Estrato", None, "Estrato 3"]
    out = estandarizar_saber11(df)
    assert out["estrato"].isna().tolist() == [True, True, False]
    assert out["estrato"].iloc[2] == 3


def test_column_map_cubre_columnas_usadas():
    assert set(COLUMN_MAP) >= {"periodo", "cole_cod_mcpio_ubicacion", "punt_global",
                               "fami_estratovivienda", "cole_naturaleza"}


def test_leer_saber11_por_chunks_equivale_a_una_pasada(crudo):
    from src.transform.saber11 import leer_saber11

    por_chunks = leer_saber11("tests/fixtures/saber11_muestra.csv", chunksize=7)
    pd.testing.assert_frame_equal(por_chunks, estandarizar_saber11(crudo))
