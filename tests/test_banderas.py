"""Banderas de calidad por IES-año: régimen especial e inconsistencia entre SPADIES y SNIES."""
import numpy as np
import pandas as pd
import pytest

from src.integrate.banderas import REGIMEN_ESPECIAL, calcular_banderas, es_regimen_especial, es_sena
from src.integrate.build import construir_dataset_desde_tablas


LISTA = ["2106", "2901", "2902", "2903", "2904", "2905", "2906", "3114", "3901", "3902",
         "9101", "9102", "9103", "9104", "9105", "9107"]


def test_regimen_especial_es_una_lista_explicita_de_16_codigos():
    assert sorted(REGIMEN_ESPECIAL) == sorted(LISTA) and len(REGIMEN_ESPECIAL) == 16
    assert es_regimen_especial(pd.Series(LISTA)).all()


def test_9108_y_9109_no_son_militares_y_no_pueden_quedar_marcadas():
    """Instituto Caro y Cuervo (9108) e ITEC (9109) están dentro del rango 910x pero no son escuelas militares."""
    marca = es_regimen_especial(pd.Series(["9108", "9109"]))
    assert not marca.any(), "9108 o 9109 quedaron marcadas como régimen especial"
    assert "9108" not in REGIMEN_ESPECIAL and "9109" not in REGIMEN_ESPECIAL


def test_la_lista_explicita_no_marca_codigos_vecinos_ni_al_sena():
    otros = pd.Series(["9100", "9106", "9110", "2907", "2999", "3903", "3999", "1101", "2833", "4106", "21060", "3114 "])
    assert not es_regimen_especial(otros).any()  # 9110 (SENA), prefijos 29xx/39xx/910x ya no cuentan


def test_es_sena_solo_marca_el_9110():
    assert es_sena(pd.Series(["9110", "9111", "9100", "1101", "91100"])).tolist() == [True, False, False, False, False]


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


def test_regimen_especial_y_sena_se_marcan_en_la_misma_tabla():
    obj = _objetivo(cod_ies=["2106", "9110", "9108", "4", "5"])
    out = calcular_banderas(obj, _snies({"2106": 100, "9110": 100, "9108": 100, "4": 100, "5": 100}), 2, 3.0)
    assert out["regimen_especial"].tolist() == [True, False, False, False, False]
    assert out["es_sena"].tolist() == [False, True, False, False, False]


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


def test_el_sena_se_marca_pero_no_se_excluye_por_defecto_y_excluir_sena_lo_saca():
    t = _tablas()
    t["catalogo"] = pd.concat([t["catalogo"], pd.DataFrame({"cod_ies": ["9110"], "cod_snies_programa": "p", "cod_divipola_mpio": "05001",
                                                           "cod_divipola_mpio_programa": "05001", "sector": "1", "caracter": "4"})])
    t["snies_programa"] = pd.concat([t["snies_programa"], pd.DataFrame({"cod_ies": ["9110"] * 3, "cod_snies_programa": "p",
                                                                       "anio": [2016, 2017, 2018], "matriculados": 100})])
    t["objetivo"] = pd.concat([t["objetivo"], pd.DataFrame({"cod_ies": ["9110"], "anio": [2018], "tasa_desercion": [0.12],
                                                           "matriculados_t2_spadies": [100.0]})])
    df, _ = construir_dataset_desde_tablas("ies", [2018], **t)
    assert sorted(df["cod_ies"]) == ["1", "9110"] and df.set_index("cod_ies").loc["9110", "es_sena"]
    df, _ = construir_dataset_desde_tablas("ies", [2018], excluir_sena=True, **t)
    assert df["cod_ies"].tolist() == ["1"]


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
