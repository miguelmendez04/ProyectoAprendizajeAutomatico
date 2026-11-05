"""Pruebas de los transformadores de SNIES, DANE y TerriData."""
import pandas as pd
import pytest

from src.transform.dane import transformar_dane, transformar_terridata
from src.transform.snies import (
    transformar_snies_catalogo,
    transformar_snies_matricula,
    transformar_snies_matricula_programa,
)

NIVELES = ["tecnica_profesional", "tecnologica", "universitaria", "especializacion", "maestria", "doctorado"]


def _matricula_mpio():
    filas = [
        # municipio con código sin cero a la izquierda y dos filas para el mismo año
        ("2018", "5001", "MEDELLÍN", [2, 4, 100, 1, 1, 1], "48"),
        ("2018", "5001", "MEDELLÍN", [0, 0, 10, 0, 0, 0], "5"),
        ("2018", "11001", "BOGOTÁ D.C.", [1, 1, 1, 1, 1, 1], "100"),
        ("2018", "0", "SIN INFORMACION", [9, 9, 9, 9, 9, 9], "1"),
    ]
    return pd.DataFrame(
        [{"a_o": a, "c_digo_delmunicipio": c, "nombre_del_municipio": n, **dict(zip(NIVELES, map(str, v))), "ies_con_oferta": i}
         for a, c, n, v, i in filas]
    )


def test_matricula_municipio_llave_unica_y_suma_de_niveles():
    out = transformar_snies_matricula(_matricula_mpio())
    assert out["cod_divipola_mpio"].str.len().eq(5).all()
    assert not out.duplicated(["cod_divipola_mpio", "anio"]).any()
    assert (out["matriculados"] >= 0).all()
    medellin = out[out["cod_divipola_mpio"] == "05001"].iloc[0]
    assert medellin["anio"] == 2018 and medellin["matriculados"] == 119  # (2+4+100+1+1+1) + 10
    assert "00000" not in set(out["cod_divipola_mpio"]) and len(out) == 2  # código especial descartado


def _programas():
    base = {
        "c_digo_de_la_instituci_n": "1208", "c_digo_snies_delprograma": "3612",
        "c_digo_del_municipio_ies": "5001", "c_digo_del_municipio_programa": "76109",
        "id_sector": "1", "id_caracter": "4", "a_o": "2019",
    }
    # dos géneros en el semestre 1 y una fila en el semestre 2
    return pd.DataFrame(
        [
            {**base, "semestre": "1", "id_g_nero": "1", "matriculados_2015": "2"},
            {**base, "semestre": "1", "id_g_nero": "2", "matriculados_2015": "3"},
            {**base, "semestre": "2", "id_g_nero": "1", "matriculados_2015": "1"},
        ]
    )


def test_catalogo_una_fila_por_ies_programa():
    out = transformar_snies_catalogo(_programas())
    assert len(out) == 1
    fila = out.iloc[0]
    assert fila["cod_ies"] == "1208" and fila["cod_snies_programa"] == "3612"
    assert fila["cod_divipola_mpio"] == "05001"  # domicilio de la IES, con cero a la izquierda
    assert fila["cod_divipola_mpio_programa"] == "76109"
    assert {"sector", "caracter"} <= set(out.columns)


def test_matricula_programa_anual_es_el_maximo_semestral():
    out = transformar_snies_matricula_programa(_programas())
    assert len(out) == 1
    assert out.iloc[0]["matriculados"] == 5  # max(2+3 del semestre 1, 1 del semestre 2)
    assert out.iloc[0]["anio"] == 2019


def test_terridata_pivotea_indicadores_con_prefijo_td():
    largo = pd.DataFrame(
        {
            "cod_municipio": ["5001", "5001", "11001"],
            "anio": [2018, 2018, 2018],
            "indicador": ["cobertura_neta", "pobreza_mpi", "cobertura_neta"],
            "valor": [90.5, 12.1, 85.0],
        }
    )
    out = transformar_terridata(largo)
    assert {"td_cobertura_neta", "td_pobreza_mpi"} <= set(out.columns)
    assert not out.duplicated(["cod_divipola_mpio", "anio"]).any()
    assert out.loc[out["cod_divipola_mpio"] == "05001", "td_cobertura_neta"].iloc[0] == 90.5


def test_dane_resuelve_municipio_solo_con_nombre():
    divipola = pd.DataFrame(
        {"cod_divipola_mpio": ["05001"], "cod_depto": ["05"], "depto_norm": ["ANTIOQUIA"], "municipio_norm": ["MEDELLIN"]}
    )
    crudo = pd.DataFrame(
        {"cod_mpio": [None], "municipio": ["Medelin"], "departamento": ["Antioquia"], "poblacion": ["2500000"], "nbi": ["10.5"]}
    )
    out = transformar_dane(crudo, divipola=divipola)
    assert out["cod_divipola_mpio"].tolist() == ["05001"]
    assert out["poblacion"].iloc[0] == 2500000 and out["nbi_pct"].iloc[0] == pytest.approx(10.5)
