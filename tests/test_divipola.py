"""Pruebas de la homologación DIVIPOLA."""
import pandas as pd
import pytest

from src.homologation.divipola import (
    aplicar_cambios_codigo,
    normalizar_codigo,
    normalizar_texto,
)


@pytest.mark.parametrize(
    "entrada, esperado",
    [(5001, "05001"), (5001.0, "05001"), ("5001", "05001"), ("11001", "11001"), (" 5001 ", "05001")],
)
def test_normalizar_codigo_agrega_ceros(entrada, esperado):
    assert normalizar_codigo(entrada) == esperado


@pytest.mark.parametrize("entrada", [0, "0", 99999, "99999", "SIN INFORMACION", "", None, float("nan")])
def test_normalizar_codigo_especiales_son_nulos(entrada):
    assert normalizar_codigo(entrada) is None


def test_normalizar_texto_quita_tildes_y_espacios():
    assert normalizar_texto("  Bogotá,  D.C. ") == "BOGOTA, D.C."


def test_aplicar_cambios_codigo_reemplaza_codigo_viejo():
    df = pd.DataFrame({"cod": ["05001", "99001"]})
    out = aplicar_cambios_codigo(df, "cod", {"99001": "99002"})
    assert out["cod"].tolist() == ["05001", "99002"]
