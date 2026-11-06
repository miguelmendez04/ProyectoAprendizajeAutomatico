"""Transformación opcional del objetivo (logit) para el modelado."""
import numpy as np
import pandas as pd
import pytest

from src.modelado.objetivo import invertir_objetivo, objetivo_para_modelado, transformar_objetivo


def test_ninguna_deja_el_objetivo_igual():
    y = pd.Series([0.1, 0.5])
    assert transformar_objetivo(y, "ninguna").tolist() == [0.1, 0.5]


def test_logit_de_cero_coma_cinco_es_cero_y_es_simetrico():
    z = transformar_objetivo(pd.Series([0.5, 0.2, 0.8]), "logit")
    assert z.iloc[0] == pytest.approx(0.0) and z.iloc[1] == pytest.approx(-z.iloc[2])
    assert z.iloc[2] == pytest.approx(np.log(0.8 / 0.2))


def test_logit_recorta_los_extremos_con_epsilon():
    z = transformar_objetivo(pd.Series([0.0, 1.0]), "logit", epsilon=0.001)
    assert np.isfinite(z).all() and z.iloc[0] == pytest.approx(np.log(0.001 / 0.999))
    assert z.iloc[1] == pytest.approx(-z.iloc[0])


def test_inversion_recupera_la_tasa():
    y = pd.Series([0.05, 0.1, 0.4, 0.9])
    assert invertir_objetivo(transformar_objetivo(y, "logit"), "logit").tolist() == pytest.approx(y.tolist())
    assert invertir_objetivo(y, "ninguna").tolist() == y.tolist()


def test_metodo_desconocido_error_claro():
    with pytest.raises(ValueError, match="transformacion_objetivo"):
        transformar_objetivo(pd.Series([0.1]), "probit")


def test_objetivo_para_modelado_lee_la_configuracion():
    df = pd.DataFrame({"tasa_desercion": [0.2, 0.5]})
    cfg = {"modelado": {"transformacion_objetivo": "logit", "epsilon_logit": 0.001}}
    assert objetivo_para_modelado(df, cfg).iloc[1] == pytest.approx(0.0)
    assert objetivo_para_modelado(df, {}).tolist() == [0.2, 0.5]  # sin configuración: sin transformar
