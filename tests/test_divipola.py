"""Pruebas de la homologación DIVIPOLA."""
import pandas as pd
import pytest

from src.homologation.divipola import (
    aplicar_cambios_codigo,
    emparejar_por_nombre,
    normalizar_codigo,
    normalizar_texto,
)


@pytest.fixture
def divipola():
    # Catálogo mínimo: "SAN ANTONIO" existe en dos departamentos distintos.
    return pd.DataFrame(
        {
            "cod_divipola_mpio": ["05001", "11001", "73671", "41668"],
            "cod_depto": ["05", "11", "73", "41"],
            "depto_norm": ["ANTIOQUIA", "BOGOTA, D.C.", "TOLIMA", "HUILA"],
            "municipio_norm": ["MEDELLIN", "BOGOTA, D.C.", "SAN ANTONIO", "SAN ANTONIO"],
        }
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


def test_emparejar_exacto_no_aparece_en_reporte(divipola):
    df = pd.DataFrame({"mpio": ["Medellín"], "dpto": ["Antioquia"]})
    out, reporte = emparejar_por_nombre(df, divipola, "mpio", "dpto")
    assert out["cod_divipola_mpio"].tolist() == ["05001"]
    assert reporte.empty


def test_emparejar_fuzzy_se_registra_en_reporte(divipola):
    df = pd.DataFrame({"mpio": ["Medelin"], "dpto": ["Antioquia"]})
    out, reporte = emparejar_por_nombre(df, divipola, "mpio", "dpto")
    assert out["cod_divipola_mpio"].tolist() == ["05001"]
    assert len(reporte) == 1
    assert reporte.iloc[0]["score"] < 1
    assert list(reporte.columns) == ["nombre_original", "nombre_emparejado", "cod_divipola_mpio", "score"]


def test_emparejar_nombre_repetido_respeta_departamento(divipola):
    df = pd.DataFrame({"mpio": ["San Antonio", "San Antonio"], "dpto": ["Tolima", "Huila"]})
    out, _ = emparejar_por_nombre(df, divipola, "mpio", "dpto")
    assert out["cod_divipola_mpio"].tolist() == ["73671", "41668"]


def test_emparejar_sin_departamento_coincidente_queda_nulo(divipola):
    df = pd.DataFrame({"mpio": ["San Antonio"], "dpto": ["Atlantico"]})
    out, _ = emparejar_por_nombre(df, divipola, "mpio", "dpto")
    assert out["cod_divipola_mpio"].isna().all()


def test_emparejar_departamento_con_otra_puntuacion_o_nombre_largo(divipola):
    divipola = pd.concat([divipola, pd.DataFrame({
        "cod_divipola_mpio": ["88001"], "cod_depto": ["88"], "municipio_norm": ["SAN ANDRES"],
        "depto_norm": ["ARCHIPIELAGO DE SAN ANDRES, PROVIDENCIA Y SANTA CATALINA"]})], ignore_index=True)
    df = pd.DataFrame({"mpio": ["Bogotá D.C.", "San Andrés"], "dpto": ["Bogotá D.C.", "San Andrés"]})
    out, _ = emparejar_por_nombre(df, divipola, "mpio", "dpto")
    assert out["cod_divipola_mpio"].tolist() == ["11001", "88001"]


def test_emparejar_no_emparejados_quedan_en_el_reporte(divipola):
    df = pd.DataFrame({"mpio": ["Atlantida"], "dpto": ["Antioquia"]})
    out, reporte = emparejar_por_nombre(df, divipola, "mpio", "dpto")
    assert out["cod_divipola_mpio"].isna().all()
    assert len(reporte) == 1 and reporte.iloc[0]["nombre_original"] == "Atlantida"
    assert pd.isna(reporte.iloc[0]["cod_divipola_mpio"])
