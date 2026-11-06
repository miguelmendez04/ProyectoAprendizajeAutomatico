"""Utilidades de partición para la fase de modelado."""
import numpy as np
import pandas as pd
import pytest

from src.modelado.particion import (
    COLUMNAS_CATEGORICAS,
    cargar_dataset,
    EXCLUIDAS_POR_DEFECTO,
    columnas_explicativas,
    columnas_explicativas_desde_config,
    particion_group_kfold,
    particion_temporal,
)


def _df():
    """10 IES x 2015-2021; 'cod_ies' como texto."""
    filas = [{"cod_ies": str(i), "anio": a, "tasa_desercion": 0.1 + 0.01 * i, "x_t1": float(i + a)}
             for i in range(10) for a in range(2015, 2022)]
    return pd.DataFrame(filas)


def test_group_kfold_nunca_reparte_una_ies_entre_entrenamiento_y_prueba():
    df = _df()
    for entrena, prueba in particion_group_kfold(df, n_splits=5):
        assert set(df.iloc[entrena]["cod_ies"]).isdisjoint(set(df.iloc[prueba]["cod_ies"]))


def test_group_kfold_cada_fila_es_de_prueba_exactamente_una_vez():
    df = _df()
    pruebas = np.concatenate([p for _, p in particion_group_kfold(df, n_splits=5)])
    assert sorted(pruebas) == list(range(len(df)))


def test_group_kfold_es_determinista_y_devuelve_n_particiones():
    df = _df()
    a = [p.tolist() for _, p in particion_group_kfold(df, n_splits=5)]
    b = [p.tolist() for _, p in particion_group_kfold(df, n_splits=5)]
    assert a == b and len(a) == 5


def test_group_kfold_con_mas_particiones_que_ies_falla_con_mensaje_claro():
    with pytest.raises(ValueError, match="IES"):
        list(particion_group_kfold(_df(), n_splits=11))


def test_temporal_entrena_hasta_2019_y_prueba_2020_2021():
    df = _df()
    entrena, prueba = particion_temporal(df, ultimo_anio_entrenamiento=2019)
    assert set(df.iloc[entrena]["anio"]) == {2015, 2016, 2017, 2018, 2019}
    assert set(df.iloc[prueba]["anio"]) == {2020, 2021}
    assert len(entrena) + len(prueba) == len(df) and set(entrena).isdisjoint(prueba)


def test_temporal_sin_filas_de_prueba_falla():
    with pytest.raises(ValueError, match="prueba"):
        particion_temporal(_df(), ultimo_anio_entrenamiento=2021)


def test_columnas_explicativas_excluye_llaves_objetivo_y_texto():
    df = _df().assign(sector="1", cod_divipola_mpio="05001")
    assert columnas_explicativas(df) == ["x_t1"]
    assert COLUMNAS_CATEGORICAS == ["sector", "caracter"]


def test_cargar_dataset_conserva_codigos_como_texto(tmp_path):
    ruta = tmp_path / "dataset_ies.csv"
    pd.DataFrame({"cod_ies": ["1830"], "anio": [2018], "tasa_desercion": [0.1], "cod_divipola_mpio": ["05001"],
                  "sector": [2], "caracter": [4]}).to_csv(ruta, index=False)
    cfg = {"unidad_analisis": "ies", "rutas": {"processed": str(tmp_path)}}
    df = cargar_dataset(cfg)
    assert df["cod_divipola_mpio"].iloc[0] == "05001" and df["sector"].iloc[0] == "2" and df["cod_ies"].iloc[0] == "1830"


def test_la_bandera_posible_cierre_no_es_explicativa():
    df = _df().assign(posible_cierre=False)
    assert "posible_cierre" not in columnas_explicativas(df)


def test_poblacion_total_y_ratio_quedan_fuera_por_defecto_pero_las_relativas_entran():
    df = _df().assign(td_poblacion_total_t1=1.0, td_pct_poblacion_rural_t1=2.0, td_pct_pob_15_19_t1=3.0, ratio_spadies_snies=1.0)
    cols = columnas_explicativas(df)
    assert "td_poblacion_total_t1" not in cols and "ratio_spadies_snies" not in cols
    assert {"td_pct_poblacion_rural_t1", "td_pct_pob_15_19_t1", "x_t1"} <= set(cols)
    assert set(EXCLUIDAS_POR_DEFECTO) == {"td_poblacion_total_t1", "ratio_spadies_snies", "matriculados_ies_t2"}


def test_se_puede_cambiar_la_lista_de_exclusion():
    df = _df().assign(td_poblacion_total_t1=1.0)
    assert "td_poblacion_total_t1" in columnas_explicativas(df, excluir=[])
    cfg = {"modelado": {"excluir_variables": ["x_t1"]}}
    assert "x_t1" not in columnas_explicativas_desde_config(df, cfg) and "td_poblacion_total_t1" in columnas_explicativas_desde_config(df, cfg)
    assert "td_poblacion_total_t1" not in columnas_explicativas_desde_config(df, {})  # sin sección: lista por defecto


def test_matricula_bruta_queda_fuera_y_su_logaritmo_entra_por_defecto():
    df = _df().assign(matriculados_ies_t2=100.0, log_matriculados_ies_t2=4.6, td_poblacion_total_t1=1.0)
    cols = columnas_explicativas(df)
    assert "matriculados_ies_t2" not in cols and "log_matriculados_ies_t2" in cols
    assert "matriculados_ies_t2" in EXCLUIDAS_POR_DEFECTO
    assert "matriculados_ies_t2" in columnas_explicativas(df, excluir=[])  # sigue en el dataset


def test_cargar_dataset_con_excluir_sena_para_el_analisis_de_sensibilidad(tmp_path):
    ruta = tmp_path / "dataset_ies.csv"
    pd.DataFrame({"cod_ies": ["1830", "9110", "9110"], "anio": [2018, 2018, 2019], "tasa_desercion": [0.1, 0.12, 0.11],
                  "es_sena": [False, True, True]}).to_csv(ruta, index=False)
    cfg = {"unidad_analisis": "ies", "rutas": {"processed": str(tmp_path)}}
    assert len(cargar_dataset(cfg)) == 3  # por defecto el SENA se queda
    assert cargar_dataset(cfg, excluir_sena=True)["cod_ies"].tolist() == ["1830"]
    cfg["modelado"] = {"excluir_sena": True}
    assert cargar_dataset(cfg)["cod_ies"].tolist() == ["1830"]  # opción de la configuración
    assert len(cargar_dataset(cfg, excluir_sena=False)) == 3  # el argumento manda sobre la configuración
