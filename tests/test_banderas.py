"""Banderas de calidad por IES-año: régimen especial e inconsistencia entre SPADIES y SNIES."""
import numpy as np
import pandas as pd
import pytest

from src.integrate.banderas import calcular_banderas, es_regimen_especial
from src.integrate.build import construir_dataset_desde_tablas


def test_regimen_especial_por_codigo():
    cods = pd.Series(["2106", "2901", "2999", "3901", "3999", "9100", "9109", "9110", "1101", "2833", "4106", "21060"])
    esperado = [True, True, True, True, True, True, True, False, False, False, False, False]
    assert es_regimen_especial(cods).tolist() == esperado  # 9110 (SENA) no es 910x


def test_regimen_especial_por_nombre_sin_tildes_ni_dobles_espacios():
    cods = pd.Series(["1", "2", "3", "4", "5", "6", "7"])
    nombres = pd.Series(["ESCUELA  MILITAR DE CADETES", "Escuela de Policía Nacional", "ESCUELA NAVAL DE SUBOFICIALES",
                         "ESCUELA FAC", "FACULTAD DE ARTES", "UNIVERSIDAD DEL VALLE", None], index=cods.index)
    assert es_regimen_especial(cods, nombres).tolist() == [True, True, True, True, False, False, False]  # FAC solo como palabra


def _objetivo(**cols):
    base = {"cod_ies": ["1", "2", "3", "4", "5"], "anio": [2018] * 5, "tasa_desercion": [0.1] * 5,
            "matriculados_t2_spadies": [100.0, 100.0, 100.0, 100.0, 100.0]}
    base.update(cols)
    return pd.DataFrame(base)


def _snies(valores: dict[str, float], anio=2016):
    return pd.DataFrame({"cod_ies": list(valores), "cod_snies_programa": "p", "anio": anio, "matriculados": list(valores.values())})


def test_ratio_e_inconsistencia_con_umbral_de_tres_en_ambos_sentidos():
    snies = _snies({"1": 100, "2": 20, "3": 400, "4": 1000})  # IES 5 sin dato de SNIES en 2016
    out = calcular_banderas(_objetivo(), snies, rezago=2, umbral_ratio=3.0).set_index("cod_ies")
    assert out.loc["1", "ratio_spadies_snies"] == pytest.approx(1.0) and not out.loc["1", "inconsistencia_fuentes"]
    assert out.loc["2", "ratio_spadies_snies"] == pytest.approx(5.0) and out.loc["2", "inconsistencia_fuentes"]  # > 3
    assert out.loc["3", "ratio_spadies_snies"] == pytest.approx(0.25) and out.loc["3", "inconsistencia_fuentes"]  # < 1/3
    assert out.loc["4", "ratio_spadies_snies"] == pytest.approx(0.1) and out.loc["4", "inconsistencia_fuentes"]
    assert np.isnan(out.loc["5", "ratio_spadies_snies"]) and out.loc["5", "inconsistencia_fuentes"]  # SNIES sin dato
    assert out.loc["1", "matriculados_t2_snies"] == 100


def test_justo_en_el_umbral_no_es_inconsistencia():
    out = calcular_banderas(_objetivo(), _snies({"1": 100 / 3, "2": 300, "3": 100, "4": 100, "5": 100}), 2, 3.0).set_index("cod_ies")
    assert not out.loc["1", "inconsistencia_fuentes"] and not out.loc["2", "inconsistencia_fuentes"]  # ratio 3 y 1/3 exactos


def test_sin_dato_de_snies_fuera_de_su_cobertura_no_es_inconsistencia():
    # SNIES por programa empieza en 2016 en este ejemplo: para 2017 el t-2 (2015) no existe en la fuente
    obj = _objetivo(anio=[2017] * 5)
    snies = pd.concat([_snies({"1": 100}, anio=2016), _snies({"1": 100}, anio=2017)])
    out = calcular_banderas(obj, snies, rezago=2, umbral_ratio=3.0).set_index("cod_ies")
    assert not out["inconsistencia_fuentes"].any() and out["ratio_spadies_snies"].isna().all()


def test_snies_en_cero_cuenta_como_sin_dato():
    out = calcular_banderas(_objetivo(), _snies({"1": 0, "2": 100, "3": 100, "4": 100, "5": 100}), 2, 3.0).set_index("cod_ies")
    assert np.isnan(out.loc["1", "ratio_spadies_snies"]) and out.loc["1", "inconsistencia_fuentes"]


def test_regimen_especial_se_marca_en_la_misma_tabla():
    obj = _objetivo(cod_ies=["2106", "2", "3", "4", "5"])
    out = calcular_banderas(obj, _snies({"2106": 100, "2": 100, "3": 100, "4": 100, "5": 100}), 2, 3.0)
    assert out["regimen_especial"].tolist() == [True, False, False, False, False]


# ---- integración con construir_dataset_desde_tablas ----------------------------------------------

def _tablas():
    catalogo = pd.DataFrame({"cod_ies": ["1", "2106", "3"], "cod_snies_programa": ["p", "p", "p"], "cod_divipola_mpio": "05001",
                             "cod_divipola_mpio_programa": "05001", "sector": "1", "caracter": "4"})
    prog = pd.DataFrame({"cod_ies": ["1", "2106", "3"] * 3, "cod_snies_programa": "p",
                         "anio": [2016] * 3 + [2017] * 3 + [2018] * 3,
                         "matriculados": [100, 100, 100, 100, 100, 100, 100, 100, 100]})
    mpio = pd.DataFrame({"cod_divipola_mpio": ["05001"] * 3, "anio": [2016, 2017, 2018], "matriculados": 1, "ies_con_oferta": 1})
    saber = pd.DataFrame({"cod_divipola_mpio": ["05001"], "anio": [2017], "punt_global_media": [250.0], "punt_global_mediana": [250.0],
                          "pct_estrato_bajo": [0.5], "pct_colegio_oficial": [0.5], "n_estudiantes": [10]})
    # IES 1: ratio 1 (bien); 2106: régimen especial; 3: SPADIES 1000 vs SNIES 100 -> inconsistente
    objetivo = pd.DataFrame({"cod_ies": ["1", "2106", "3"], "anio": [2018] * 3, "tasa_desercion": [0.1, 0.9, 0.8],
                             "matriculados_t2_spadies": [100.0, 100.0, 1000.0]})
    return dict(saber_mpio=saber, snies_mpio=mpio, snies_programa=prog, catalogo=catalogo, objetivo=objetivo)


def test_exclusiones_por_defecto_y_columnas_de_diagnostico_en_el_dataset():
    registro: dict = {}
    df, _ = construir_dataset_desde_tablas("ies", [2018], registro=registro, **_tablas())
    assert df["cod_ies"].tolist() == ["1"]
    assert {"ratio_spadies_snies", "inconsistencia_fuentes", "regimen_especial"} <= set(df.columns)
    assert not {"matriculados_t2_spadies", "matriculados_t2_snies"} & set(df.columns)  # el denominador no entra al dataset
    banderas = registro["banderas"].set_index("cod_ies")
    assert banderas.loc["2106", "regimen_especial"] and banderas.loc["3", "inconsistencia_fuentes"]


def test_las_opciones_de_exclusion_se_pueden_apagar_por_separado():
    t = _tablas()
    df, _ = construir_dataset_desde_tablas("ies", [2018], excluir_inconsistencias=False, **t)
    assert sorted(df["cod_ies"]) == ["1", "3"]  # solo sale el régimen especial
    df, _ = construir_dataset_desde_tablas("ies", [2018], excluir_regimen_especial=False, **t)
    assert sorted(df["cod_ies"]) == ["1", "2106"]
    df, _ = construir_dataset_desde_tablas("ies", [2018], excluir_inconsistencias=False, excluir_regimen_especial=False, **t)
    assert len(df) == 3


def test_en_unidad_municipio_las_exclusiones_se_aplican_antes_de_agregar():
    df, _ = construir_dataset_desde_tablas("municipio", [2018], **_tablas())
    assert df["tasa_desercion"].tolist() == [0.1]  # solo la IES 1 entra al promedio municipal
