"""Motor de experimentos: sin fuga, mismos pliegues, línea base ingenua, logit y NUNCA_EXPLICATIVAS."""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler

import desercion_ies.evaluation.experimentos as exp
from desercion_ies.evaluation.experimentos import (
    BASE_INGENUA,
    BASE_MEDIANA,
    _inversa_logit,
    _logit,
    construir_pipeline,
    ejecutar_experimentos,
    grilla,
    nombre_configuracion,
    parametros_experimentos,
    prediccion_ingenua,
    preparar_datos,
    resumen,
)
from desercion_ies.features.objetivo import invertir_objetivo, transformar_objetivo
from desercion_ies.features.particion import NUNCA_EXPLICATIVAS, construir_preprocesador


def _datos(n_ies: int = 40, anios=(2017, 2018, 2019), semilla: int = 0) -> pd.DataFrame:
    """Panel sintético con nulos en una numérica, y las categóricas y el id como texto."""
    rng = np.random.default_rng(semilla)
    filas = []
    for i in range(n_ies):
        base = rng.uniform(0.05, 0.25)
        for a in anios:
            filas.append(
                {
                    "cod_ies": str(1000 + i),
                    "anio": a,
                    "sector": str(1 + i % 2),
                    "caracter": str(1 + i % 4),
                    "tasa_desercion": float(np.clip(base + rng.normal(0, 0.02), 0.01, 0.9)),
                    "x1": rng.normal(),
                    "x2": rng.normal() * 10 + (np.nan if rng.random() < 0.15 else 0),
                    "x3": rng.uniform(0, 1),
                    "falta_saber11": a == 2019,
                }
            )
    return pd.DataFrame(filas)


def _cfg(**experimentos) -> dict:
    base = {
        "semilla": 7,
        "objetivo": ["ninguna", "logit"],
        "imputacion": ["mediana", "knn"],
        "seleccion": ["ninguna", "pca"],
        "modelo": ["ridge"],
        "k_info_mutua": 3,
    }
    base.update(experimentos)
    return {
        "particion": {"n_splits": 3},
        "modelado": {"excluir_variables": [], "experimentos": base},
    }


# ---------- grilla ----------
def test_la_grilla_por_defecto_tiene_36_configuraciones():
    g = grilla(parametros_experimentos({}))
    assert len(g) == 36 and len({nombre_configuracion(c) for c in g}) == 36


def test_valores_invalidos_de_la_grilla_fallan():
    with pytest.raises(ValueError, match="modelo"):
        parametros_experimentos({"modelado": {"experimentos": {"modelo": ["svm"]}}})


def test_imputacion_invalida_falla():
    with pytest.raises(ValueError, match="imputacion"):
        construir_preprocesador(["x1"], [], imputacion="media")


# ---------- sin fuga: todo se ajusta solo con el entrenamiento ----------
@pytest.mark.parametrize("imputacion", ["mediana", "knn"])
def test_el_preprocesamiento_se_ajusta_solo_con_el_entrenamiento(imputacion):
    df = _datos()
    p = parametros_experimentos(_cfg())
    X = df[["x1", "x2", "x3", "sector", "caracter"]].copy()
    entrena, prueba = np.arange(0, 90), np.arange(90, len(df))
    X.iloc[prueba, X.columns.get_loc("x1")] += (
        1000.0  # prueba con valores extremos: no debe influir en nada ajustado
    )
    config = {
        "objetivo": "ninguna",
        "imputacion": imputacion,
        "seleccion": "ninguna",
        "modelo": "ridge",
    }
    pipe = construir_pipeline(config, ["x1", "x2", "x3"], ["sector", "caracter"], p)
    pipe.fit(X.iloc[entrena], df["tasa_desercion"].iloc[entrena].to_numpy())
    num = pipe.named_steps["pre"].named_transformers_["num"]
    imputador, escalar = num.named_steps["imputar"], num.named_steps["escalar"]
    assert isinstance(escalar, StandardScaler)
    # la media y la desviación del escalador son las del entrenamiento (con x1 sin el desplazamiento de la prueba)
    x1_entrena = X["x1"].iloc[entrena]
    assert escalar.mean_[0] == pytest.approx(x1_entrena.mean())
    assert escalar.mean_[0] < 10  # con la prueba incluida valdría ~ +1000 * 30 / 120 = +250
    if imputacion == "mediana":
        assert imputador.statistics_[1] == pytest.approx(X["x2"].iloc[entrena].median())
    else:
        assert imputador._fit_X.shape[0] == len(entrena)  # los vecinos solo salen del entrenamiento


def test_la_seleccion_y_el_modelo_tambien_se_ajustan_solo_con_el_entrenamiento():
    df = _datos()
    p = parametros_experimentos(_cfg(k_info_mutua=2))
    X = df[["x1", "x2", "x3", "sector", "caracter"]]
    entrena = np.arange(0, 90)
    config = {
        "objetivo": "ninguna",
        "imputacion": "mediana",
        "seleccion": "info_mutua",
        "modelo": "ridge",
    }
    pipe = construir_pipeline(config, ["x1", "x2", "x3"], ["sector", "caracter"], p)
    pipe.fit(X.iloc[entrena], df["tasa_desercion"].iloc[entrena].to_numpy())
    assert (
        pipe.named_steps["sel"].n_features_in_
        == pipe.named_steps["pre"].transform(X.iloc[entrena]).shape[1]
    )
    assert pipe.named_steps["modelo"].n_features_in_ == 2
    assert pipe.named_steps["modelo"].coef_.shape == (2,)


# ---------- mismos pliegues ----------
def test_todas_las_configuraciones_usan_los_mismos_pliegues(monkeypatch):
    llamadas = []
    original = exp.kfold_desde_config

    def espia(df, cfg):
        llamadas.append(1)
        return original(df, cfg)

    monkeypatch.setattr(exp, "kfold_desde_config", espia)
    res = ejecutar_experimentos(_cfg(), _datos())
    assert len(llamadas) == 1  # los pliegues se generan una sola vez
    n_cfg = res["configuracion"].nunique()
    assert n_cfg == 2 + 8  # 2 líneas base + 2 objetivos x 2 imputaciones x 2 selecciones x 1 modelo
    huellas = res.groupby("pliegue")["hash_prueba"].nunique()
    assert (huellas == 1).all()
    n_prueba = res.groupby("pliegue")["n_prueba"].nunique()
    assert (n_prueba == 1).all()
    assert res.groupby("configuracion")["pliegue"].nunique().eq(3).all()


def test_los_pliegues_no_comparten_ies_entre_entrenamiento_y_prueba():
    df = _datos()
    cfg = _cfg()
    pliegues = list(exp.kfold_desde_config(df, cfg))
    assert len(pliegues) == 3
    for entrena, prueba in pliegues:
        assert set(df.iloc[entrena]["cod_ies"]).isdisjoint(df.iloc[prueba]["cod_ies"])
    todas = np.concatenate([p for _, p in pliegues])
    assert sorted(todas) == list(range(len(df)))  # cada fila se evalúa exactamente una vez


# ---------- línea base ingenua ----------
def test_la_ingenua_toma_el_anio_t_menos_1_de_la_misma_ies():
    df = pd.DataFrame(
        {
            "cod_ies": ["A", "A", "A", "B", "B", "C"],
            "anio": [2015, 2016, 2018, 2016, 2017, 2017],
            "tasa_desercion": [0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
        }
    )
    y = df["tasa_desercion"]
    entrena = np.array([5])  # solo C entrena: la mediana de entrenamiento es 0,60
    prueba = np.array([0, 1, 2, 3, 4])
    pred, n_t1, n_mediana = prediccion_ingenua(df, y, entrena, prueba)
    # A2015: sin fila 2014 -> mediana; A2016 -> A2015 = 0,10; A2018: no hay 2017 (no toma 2016) -> mediana;
    # B2016: sin 2015 -> mediana; B2017 -> B2016 = 0,40
    assert pred.tolist() == pytest.approx([0.60, 0.10, 0.60, 0.60, 0.40])
    assert (n_t1, n_mediana) == (2, 3)


def test_la_ingenua_no_confunde_ies_distintas_en_el_mismo_anio():
    df = pd.DataFrame(
        {
            "cod_ies": ["A", "B", "A", "B"],
            "anio": [2016, 2016, 2017, 2017],
            "tasa_desercion": [0.1, 0.9, 0.2, 0.8],
        }
    )
    pred, n_t1, _ = prediccion_ingenua(df, df["tasa_desercion"], np.array([0, 1]), np.array([2, 3]))
    assert pred.tolist() == pytest.approx([0.1, 0.9]) and n_t1 == 2


def test_las_lineas_base_salen_en_el_mismo_formato_y_cuentan_los_casos():
    res = ejecutar_experimentos(
        _cfg(objetivo=["ninguna"], imputacion=["mediana"], seleccion=["ninguna"]), _datos()
    )
    base = res[res["tipo"] == "linea_base"]
    assert set(base["configuracion"]) == {BASE_MEDIANA, BASE_INGENUA}
    ing = base[base["configuracion"] == BASE_INGENUA]
    assert (ing["n_ingenua_t1"] + ing["n_ingenua_mediana"] == ing["n_prueba"]).all()
    assert ing["n_ingenua_t1"].sum() == 40 * 2  # 3 años por IES: t-1 existe en 2 de los 3
    assert list(res.columns) == exp.COLUMNAS_RESULTADO


# ---------- logit ----------
def test_el_logit_se_invierte_bien():
    p = np.array([0.01, 0.1, 0.5, 0.9, 0.99])
    assert _inversa_logit(_logit(p)) == pytest.approx(p)
    # coincide con objetivo.py
    assert _logit(p) == pytest.approx(transformar_objetivo(pd.Series(p), "logit").to_numpy())
    assert _inversa_logit(np.array([-3.0, 0.0, 2.0])) == pytest.approx(
        invertir_objetivo(pd.Series([-3.0, 0.0, 2.0]), "logit").to_numpy()
    )
    # un objetivo en 0 o 1 se recorta (epsilon) y no produce infinitos
    assert np.isfinite(_logit(np.array([0.0, 1.0]))).all()


def test_con_logit_el_modelo_se_entrena_en_logit_y_predice_en_escala_de_tasa(monkeypatch):
    df = _datos()
    p = parametros_experimentos(_cfg())
    monkeypatch.setattr(exp, "construir_modelo", lambda nombre, p: DummyRegressor(strategy="mean"))
    X, y = df[["x1", "x2", "x3", "sector", "caracter"]], df["tasa_desercion"].to_numpy()
    config = {
        "objetivo": "logit",
        "imputacion": "mediana",
        "seleccion": "ninguna",
        "modelo": "ridge",
    }
    pipe = construir_pipeline(config, ["x1", "x2", "x3"], ["sector", "caracter"], p)
    pipe.fit(X, y)
    z_media = float(np.mean(_logit(y)))
    assert pipe.regressor_.named_steps["modelo"].constant_.ravel()[0] == pytest.approx(
        z_media
    )  # entrenó en logit
    pred = pipe.predict(X)
    assert pred == pytest.approx(np.full(len(X), 1 / (1 + np.exp(-z_media))))  # y devuelve tasa
    assert not np.isclose(
        pred[0], y.mean()
    )  # no es la media de la tasa: viene de la inversa del logit
    assert ((pred > 0) & (pred < 1)).all()


def test_sin_logit_el_objetivo_no_se_transforma(monkeypatch):
    df = _datos()
    p = parametros_experimentos(_cfg())
    monkeypatch.setattr(exp, "construir_modelo", lambda nombre, p: DummyRegressor(strategy="mean"))
    config = {
        "objetivo": "ninguna",
        "imputacion": "mediana",
        "seleccion": "ninguna",
        "modelo": "ridge",
    }
    pipe = construir_pipeline(config, ["x1", "x2", "x3"], ["sector", "caracter"], p)
    pipe.fit(df[["x1", "x2", "x3", "sector", "caracter"]], df["tasa_desercion"].to_numpy())
    assert pipe.predict(df[["x1", "x2", "x3", "sector", "caracter"]])[0] == pytest.approx(
        df["tasa_desercion"].mean()
    )


# ---------- NUNCA_EXPLICATIVAS ----------
def test_ninguna_columna_de_nunca_explicativas_entra_al_modelo():
    df = _datos()
    df["n_anios_ies"] = 3.0 + np.arange(len(df)) % 2  # usa años futuros: nunca debe ser explicativa
    df["panel_incompleto"] = np.arange(len(df)) % 5 == 0
    # incluso con excluir=[] y con la opción del panel incompleto encendida
    cfg = _cfg()
    cfg["modelado"].update({"excluir_variables": [], "indicador_panel_incompleto": True})
    df, X, y, numericas, categoricas = preparar_datos(cfg, df)
    for col in NUNCA_EXPLICATIVAS:
        assert col not in numericas and col not in categoricas and col not in X.columns
    assert "panel_incompleto" in numericas  # esta sí puede entrar (opción de sesgo)
    # el pipeline ajustado tampoco la ve
    p = parametros_experimentos(cfg)
    config = {
        "objetivo": "ninguna",
        "imputacion": "mediana",
        "seleccion": "ninguna",
        "modelo": "ridge",
    }
    pipe = construir_pipeline(config, numericas, categoricas, p).fit(X, y.to_numpy())
    assert not set(NUNCA_EXPLICATIVAS) & set(pipe.named_steps["pre"].feature_names_in_)


def test_el_objetivo_y_las_llaves_no_entran_como_explicativas():
    df, X, y, numericas, categoricas = preparar_datos(_cfg(), _datos())
    for prohibida in ("tasa_desercion", "anio", "cod_ies"):
        assert prohibida not in X.columns


# ---------- resumen ----------
def test_el_resumen_calcula_la_mejora_frente_a_la_ingenua():
    res = ejecutar_experimentos(
        _cfg(objetivo=["ninguna"], imputacion=["mediana"], seleccion=["ninguna"]), _datos()
    )
    r = resumen(res).set_index("configuracion")
    ing = r.loc[BASE_INGENUA, "mae_media"]
    assert r.loc[BASE_INGENUA, "mejora_vs_ingenua_pct"] == pytest.approx(0)
    for _c, fila in r.iterrows():
        assert fila["mejora_vs_ingenua_pct"] == pytest.approx(100 * (ing - fila["mae_media"]) / ing)
    assert list(resumen(res)["mae_media"]) == sorted(resumen(res)["mae_media"])


def test_las_metricas_se_miden_en_escala_de_tasa_aunque_el_objetivo_sea_logit():
    res = ejecutar_experimentos(
        _cfg(objetivo=["logit"], imputacion=["mediana"], seleccion=["ninguna"]), _datos()
    )
    mod = res[res["tipo"] == "modelo"]
    assert (mod["mae"] < 1).all() and (mod["rmse"] < 1).all()  # en logit serían ~ 1 o más
