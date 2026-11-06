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
    assert set(EXCLUIDAS_POR_DEFECTO) == {"td_poblacion_total_t1", "ratio_spadies_snies", "matriculados_ies_t2", "log_matriculados_ies_t2"}


def test_se_puede_cambiar_la_lista_de_exclusion():
    df = _df().assign(td_poblacion_total_t1=1.0)
    assert "td_poblacion_total_t1" in columnas_explicativas(df, excluir=[])
    cfg = {"modelado": {"excluir_variables": ["x_t1"]}}
    assert "x_t1" not in columnas_explicativas_desde_config(df, cfg) and "td_poblacion_total_t1" in columnas_explicativas_desde_config(df, cfg)
    assert "td_poblacion_total_t1" not in columnas_explicativas_desde_config(df, {})  # sin sección: lista por defecto


def test_tamano_por_defecto_es_el_log_del_denominador_spadies_y_el_de_snies_es_la_alternativa():
    df = _df().assign(matriculados_ies_t2=100.0, log_matriculados_ies_t2=4.6, log_matriculados_spadies_t2=5.0, td_poblacion_total_t1=1.0)
    cols = columnas_explicativas(df)
    assert "log_matriculados_spadies_t2" in cols
    assert "matriculados_ies_t2" not in cols and "log_matriculados_ies_t2" not in cols
    assert {"matriculados_ies_t2", "log_matriculados_ies_t2"} <= set(columnas_explicativas(df, excluir=[]))  # siguen en el dataset


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


# ---- variables categóricas, preprocesador y ventana ----------------------------------------------------

def _df_cat():
    df = _df().assign(sector=["1", "2"] * 35, caracter=["4", "3", "2", "1", "4"] * 14, log_matriculados_spadies_t2=5.0,
                      log_matriculados_ies_t2=5.1, falta_saber11=False, falta_mdm=False, falta_snies=True)
    df.loc[0, "x_t1"] = np.nan
    return df


def test_columnas_categoricas_son_sector_y_caracter_si_existen():
    from src.modelado.particion import columnas_categoricas

    assert columnas_categoricas(_df_cat()) == ["sector", "caracter"]
    assert columnas_categoricas(_df()) == []


def test_etiquetas_de_sector_y_caracter_segun_snies():
    from src.modelado.particion import ETIQUETAS_CARACTER, ETIQUETAS_SECTOR, etiquetar_categoricas

    assert ETIQUETAS_SECTOR == {"1": "oficial", "2": "privada"}
    assert ETIQUETAS_CARACTER == {"1": "técnica profesional", "2": "tecnológica", "3": "institución universitaria", "4": "universidad"}
    out = etiquetar_categoricas(pd.DataFrame({"sector": ["1", "2", "9"], "caracter": ["4", "1", None]}))
    assert out["sector"].tolist() == ["oficial", "privada", "desconocido"]
    assert out["caracter"].tolist()[:2] == ["universidad", "técnica profesional"] and out["caracter"].iloc[2] == "desconocido"


def test_variables_modelado_separa_numericas_y_categoricas_con_el_tamano_spadies_por_defecto():
    from src.modelado.particion import variables_modelado

    numericas, categoricas = variables_modelado(_df_cat())
    assert categoricas == ["sector", "caracter"]
    assert "log_matriculados_spadies_t2" in numericas and "log_matriculados_ies_t2" not in numericas  # la de SNIES es la alternativa
    assert {"falta_saber11", "falta_mdm", "falta_snies"} <= set(numericas) and "x_t1" in numericas
    assert not set(numericas) & set(categoricas)
    sin_falta, _ = variables_modelado(_df_cat(), incluir_indicadores_falta=False)
    assert not any(c.startswith("falta_") for c in sin_falta)
    alternativa, _ = variables_modelado(_df_cat(), excluir=["log_matriculados_spadies_t2"])
    assert "log_matriculados_ies_t2" in alternativa and "log_matriculados_spadies_t2" not in alternativa


def test_preprocesador_one_hot_imputa_y_tolera_categorias_nuevas():
    from src.modelado.particion import construir_preprocesador, preparar_x, variables_modelado

    df = _df_cat()
    numericas, categoricas = variables_modelado(df)
    X = preparar_x(df, numericas, categoricas)
    assert X["sector"].isin(["oficial", "privada"]).all() and X["falta_snies"].dtype.kind in "iu"
    pre = construir_preprocesador(numericas, categoricas)
    M = pre.fit_transform(X)
    assert not np.isnan(M).any()  # el nulo de x_t1 se imputó
    assert M.shape[1] == len(numericas) + 2 + 4  # 2 sectores + 4 caracteres en one-hot
    nuevo = X.head(1).assign(caracter="categoría nunca vista")
    assert pre.transform(nuevo).shape[1] == M.shape[1]  # handle_unknown="ignore"


def test_ventana_modelado_filtra_los_anios(tmp_path):
    ruta = tmp_path / "dataset_ies.csv"
    pd.DataFrame({"cod_ies": ["1"] * 7, "anio": list(range(2015, 2022)), "tasa_desercion": 0.1}).to_csv(ruta, index=False)
    cfg = {"unidad_analisis": "ies", "rutas": {"processed": str(tmp_path)}}
    assert cargar_dataset(cfg)["anio"].tolist() == list(range(2015, 2022))  # por defecto 2015–2021
    assert cargar_dataset(cfg, ventana_modelado=(2017, 2021))["anio"].tolist() == [2017, 2018, 2019, 2020, 2021]
    cfg["modelado"] = {"ventana_modelado": [2017, 2021]}
    assert cargar_dataset(cfg)["anio"].min() == 2017
    entrena, prueba = particion_temporal(cargar_dataset(cfg), 2019)
    assert len(entrena) == 3 and len(prueba) == 2  # con la ventana 2017–2021 se entrena con 2017–2019


def test_ventana_invalida_error_claro(tmp_path):
    ruta = tmp_path / "dataset_ies.csv"
    pd.DataFrame({"cod_ies": ["1"], "anio": [2018], "tasa_desercion": 0.1}).to_csv(ruta, index=False)
    cfg = {"unidad_analisis": "ies", "rutas": {"processed": str(tmp_path)}}
    with pytest.raises(ValueError, match="ventana_modelado"):
        cargar_dataset(cfg, ventana_modelado=(2021, 2017))


def test_variables_modelado_desde_config_usa_las_opciones_de_la_configuracion():
    from src.modelado.particion import variables_modelado_desde_config

    df = _df_cat()
    num, cat = variables_modelado_desde_config(df, {})
    assert "log_matriculados_spadies_t2" in num and "falta_mdm" in num and cat == ["sector", "caracter"]
    cfg = {"modelado": {"excluir_variables": ["log_matriculados_spadies_t2"], "incluir_indicadores_falta": False}}
    num, _ = variables_modelado_desde_config(df, cfg)
    assert "log_matriculados_spadies_t2" not in num and "log_matriculados_ies_t2" in num and not any(c.startswith("falta_") for c in num)
