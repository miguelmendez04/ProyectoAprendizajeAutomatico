"""Formulaciones con historia: sin fuga, mismos pliegues, corrección sobre la ingenua, recorte y desglose con/sin historia."""
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

import src.modelado.formulaciones as fm
from src.integrate.historia import agregar_historia
from src.modelado.experimentos import BASE_INGENUA, BASE_MEDIANA, parametros_experimentos, prediccion_ingenua
from src.modelado.formulaciones import (
    FORMULACIONES, CorreccionSobreIngenua, ejecutar_formulaciones, grilla_formulacion, parametros_formulaciones, resumen,
    variables_formulacion,
)
from src.modelado.particion import COLUMNAS_HISTORIA, NUNCA_EXPLICATIVAS, preparar_x


def _datos(n_ies: int = 40, anios=range(2015, 2020), semilla: int = 0) -> pd.DataFrame:
    """Panel sintético con algunos huecos (filas ausentes), sector/carácter como texto y la historia ya agregada."""
    rng = np.random.default_rng(semilla)
    filas = []
    for i in range(n_ies):
        base = rng.uniform(0.05, 0.25)
        for a in anios:
            if rng.random() < 0.08:
                continue
            filas.append({"cod_ies": str(1000 + i), "anio": a, "sector": str(1 + i % 2), "caracter": str(1 + i % 4),
                          "tasa_desercion": float(np.clip(base + rng.normal(0, 0.02), 0.01, 0.9)),
                          "x1": rng.normal(), "x2": rng.normal(), "falta_saber11": a == 2019})
    return agregar_historia(pd.DataFrame(filas), ["cod_ies"])


def _cfg(**formulaciones) -> dict:
    f = {"incluir": list(FORMULACIONES), "objetivo": ["ninguna", "logit"], "imputacion": ["mediana"], "seleccion": ["ninguna", "pca"],
         "modelo": ["ridge"]}
    f.update(formulaciones)
    return {"particion": {"n_splits": 3},
            "modelado": {"excluir_variables": [], "experimentos": {"semilla": 3, "k_info_mutua": 15}, "formulaciones": f}}


# ---------- qué variables lleva cada formulación ----------
def test_el_contexto_no_lleva_historia_y_las_demas_formulaciones_si():
    df, cfg = _datos(), _cfg()
    num_a, cat_a = variables_formulacion(df, cfg, "contexto")
    assert not set(COLUMNAS_HISTORIA) & set(num_a + cat_a)
    num_b, cat_b = variables_formulacion(df, cfg, "historia_contexto")
    assert set(COLUMNAS_HISTORIA) <= set(num_b) and set(num_a) <= set(num_b) and cat_b == cat_a
    num_h, cat_h = variables_formulacion(df, cfg, "solo_historia")
    assert sorted(num_h) == sorted(COLUMNAS_HISTORIA) and cat_h == []


def test_ninguna_formulacion_lleva_nunca_explicativas():
    df = _datos().assign(n_anios_ies=5.0)
    for conjunto in ("contexto", "historia_contexto", "solo_historia"):
        num, cat = variables_formulacion(df, _cfg(), conjunto)
        assert not set(NUNCA_EXPLICATIVAS) & set(num + cat)


def test_sin_columnas_de_historia_el_error_es_claro():
    df = _datos().drop(columns=list(COLUMNAS_HISTORIA))
    with pytest.raises(ValueError, match="historia"):
        variables_formulacion(df, _cfg(), "solo_historia")


def test_las_variables_de_una_fila_no_dependen_de_su_propia_tasa_ni_de_las_posteriores():
    """Alterar la tasa del año t (y posteriores) de una IES no cambia la fila de entrada X del año t en ninguna formulación."""
    df, cfg = _datos(), _cfg()
    i = df.index[(df["anio"] == 2017) & df["tasa_t1"].notna()][0]
    ies = df.loc[i, "cod_ies"]
    alterado = df.drop(columns=list(COLUMNAS_HISTORIA)).copy()
    alterado.loc[(alterado["cod_ies"] == ies) & (alterado["anio"] >= 2017), "tasa_desercion"] = 0.999
    alterado = agregar_historia(alterado, ["cod_ies"])
    for conjunto in ("contexto", "historia_contexto", "solo_historia"):
        num, cat = variables_formulacion(df, cfg, conjunto)
        x0, x1 = preparar_x(df, num, cat).loc[i], preparar_x(alterado, num, cat).loc[i]
        pd.testing.assert_series_equal(x0.astype(str), x1.astype(str))


# ---------- corrección sobre la ingenua ----------
def test_la_correccion_suma_el_cambio_predicho_a_la_ingenua():
    X = pd.DataFrame({"tasa_t1": [0.10, 0.20, np.nan, 0.40, np.nan], "otra": 1.0})
    y = np.array([0.12, 0.25, 0.30, 0.38, 0.10])
    m = CorreccionSobreIngenua(DummyRegressor(strategy="mean")).fit(X, y)
    mediana = np.median(y)
    assert m.mediana_ == pytest.approx(mediana)
    base = np.array([0.10, 0.20, mediana, 0.40, mediana])  # sin t1: mediana del entrenamiento
    cambio = y - base
    assert m.regressor_.constant_.ravel()[0] == pytest.approx(cambio.mean())  # el regresor se entrenó con el cambio
    assert m.predict(X) == pytest.approx(base + cambio.mean())


def test_la_mediana_de_la_correccion_sale_solo_del_entrenamiento():
    entrena = pd.DataFrame({"tasa_t1": [np.nan, 0.1], "x": 0.0})
    m = CorreccionSobreIngenua(DummyRegressor(strategy="constant", constant=0.0)).fit(entrena, np.array([0.2, 0.4]))
    prueba = pd.DataFrame({"tasa_t1": [np.nan], "x": 0.0})
    assert m.predict(prueba)[0] == pytest.approx(0.3)  # la mediana de [0.2, 0.4], no de datos de prueba


def test_en_la_correccion_no_hay_logit():
    p = parametros_formulaciones(_cfg())
    assert {c["objetivo"] for c in grilla_formulacion("C_correccion", p)} == {"ninguna"}
    assert {c["objetivo"] for c in grilla_formulacion("B_historia_contexto", p)} == {"ninguna", "logit"}


def test_la_grilla_por_defecto_tiene_72_configuraciones():
    p = parametros_formulaciones({})
    assert sum(len(grilla_formulacion(f, p)) for f in p["incluir"]) == 3 * 18 + 2 * 9


# ---------- mismos pliegues y líneas base ----------
def test_todas_las_formulaciones_y_lineas_base_usan_los_mismos_pliegues(monkeypatch):
    llamadas = []
    original = fm.kfold_desde_config

    def espia(df, cfg):
        llamadas.append(1)
        return original(df, cfg)

    monkeypatch.setattr(fm, "kfold_desde_config", espia)
    res = ejecutar_formulaciones(_cfg(), _datos())
    assert len(llamadas) == 1
    assert (res.groupby("pliegue")["hash_prueba"].nunique() == 1).all()
    assert (res.groupby("pliegue")["n_prueba"].nunique() == 1).all()
    assert set(res["formulacion"]) == set(FORMULACIONES) | {"linea_base"}
    assert res.groupby("configuracion")["pliegue"].nunique().eq(3).all()


def test_los_pliegues_coinciden_con_los_de_la_v1():
    """Mismo generador de pliegues que el motor de la v1: las huellas de las IES de prueba coinciden."""
    import src.modelado.experimentos as exp

    df = _datos()
    v1 = [exp.hash_pliegue(df, prueba) for _, prueba in exp.kfold_desde_config(df, _cfg())]
    v2 = ejecutar_formulaciones(_cfg(incluir=["A_contexto"], objetivo=["ninguna"], seleccion=["ninguna"]), df)
    assert v2[v2["configuracion"] == BASE_INGENUA].sort_values("pliegue")["hash_prueba"].tolist() == v1


def test_la_ingenua_coincide_con_tasa_t1_y_usa_la_mediana_si_no_hay():
    df = _datos()
    y = df["tasa_desercion"]
    entrena, prueba = np.arange(0, 100), np.arange(100, len(df))
    pred, n_t1, n_mediana = prediccion_ingenua(df, y, entrena, prueba)
    t1 = df["tasa_t1"].to_numpy()[prueba]
    assert pred == pytest.approx(np.where(np.isnan(t1), np.median(y.iloc[entrena]), t1))
    assert n_t1 == int((~np.isnan(t1)).sum()) and n_mediana == int(np.isnan(t1).sum())


def test_con_y_sin_historia_suman_las_filas_de_prueba_y_la_ingenua_sin_historia_es_la_mediana():
    res = ejecutar_formulaciones(_cfg(incluir=["A_contexto"], objetivo=["ninguna"], seleccion=["ninguna"]), _datos())
    assert (res["n_con_historia"] + res["n_sin_historia"] == res["n_prueba"]).all()
    ing = res[res["configuracion"] == BASE_INGENUA].set_index("pliegue")
    med = res[res["configuracion"] == BASE_MEDIANA].set_index("pliegue")
    assert ing["mae_sin_historia"].to_numpy() == pytest.approx(med["mae_sin_historia"].to_numpy())  # misma predicción
    assert (ing["mae_con_historia"] < med["mae_con_historia"]).all()  # con historia la ingenua mejora a la mediana (tasas persistentes)


# ---------- recorte a [0, 1] ----------
class _FueraDeRango:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.full(len(X), 5.0)


def test_las_predicciones_finales_se_recortan_a_0_1(monkeypatch):
    df, cfg = _datos(), _cfg()
    monkeypatch.setattr(fm, "construir_pipeline_formulacion", lambda *a, **k: _FueraDeRango())
    y = df["tasa_desercion"]
    num, cat = variables_formulacion(df, cfg, "contexto")
    X = preparar_x(df, num, cat)
    pl = list(fm.kfold_desde_config(df, cfg))
    c = {"formulacion": "A_contexto", "objetivo": "ninguna", "imputacion": "mediana", "seleccion": "ninguna", "modelo": "ridge"}
    p_exp = parametros_experimentos(cfg)
    con = fm.evaluar_configuracion(c, df, y, X, pl, num, cat, p_exp, recortar=True)
    sin = fm.evaluar_configuracion(c, df, y, X, pl, num, cat, p_exp, recortar=False)
    prueba0 = pl[0][1]
    assert con[0]["mae"] == pytest.approx(np.mean(np.abs(y.iloc[prueba0] - 1.0)))
    assert sin[0]["mae"] == pytest.approx(np.mean(np.abs(y.iloc[prueba0] - 5.0)))


# ---------- solo historia con selección por información mutua ----------
def test_info_mutua_con_solo_historia_recorta_k_al_numero_de_variables():
    df, cfg = _datos(), _cfg()
    num, cat = variables_formulacion(df, cfg, "solo_historia")
    c = {"formulacion": "B_solo_historia", "objetivo": "ninguna", "imputacion": "mediana", "seleccion": "info_mutua", "modelo": "ridge"}
    pipe = fm.construir_pipeline_formulacion(c, num, cat, parametros_experimentos(cfg))
    pipe.fit(preparar_x(df, num, cat), df["tasa_desercion"].to_numpy())  # k = 15 > 5 variables no debe fallar
    assert pipe.named_steps["sel"].k == 5


# ---------- resumen e informe ----------
def test_el_resumen_calcula_la_mejora_frente_a_la_ingenua_total_y_por_subconjunto():
    res = ejecutar_formulaciones(_cfg(incluir=["A_contexto", "B_solo_historia"], objetivo=["ninguna"], seleccion=["ninguna"]), _datos())
    r = resumen(res).set_index("configuracion")
    ing = r.loc[BASE_INGENUA]
    for _, fila in r.iterrows():
        assert fila["mejora_vs_ingenua_pct"] == pytest.approx(100 * (ing["mae_media"] - fila["mae_media"]) / ing["mae_media"])
        assert fila["mejora_vs_ingenua_con_historia_pct"] == pytest.approx(
            100 * (ing["mae_con_historia_media"] - fila["mae_con_historia_media"]) / ing["mae_con_historia_media"])
    assert r.loc[BASE_INGENUA, "mejora_vs_ingenua_pct"] == pytest.approx(0)


def test_el_informe_y_la_figura_se_generan(tmp_path):
    from src.modelado.informe_formulaciones import escribir_informe, figura_formulaciones

    res = ejecutar_formulaciones(_cfg(objetivo=["ninguna"], seleccion=["ninguna"]), _datos())
    r = resumen(res)
    meta = {"filas": 200, "ies": 40, "ventana": [2015, 2019], "n_pliegues": 3, "n_contexto": 5, "k_info_mutua": 15, "varianza_pca": 0.9,
            "semilla": 3, "tiempo_total_s": 1.0}
    texto = escribir_informe(res, r, meta, tmp_path / "r.md")
    assert "Conclusión principal" in texto and "B_solo_historia" in texto and "Aporte del contexto" in texto
    figura_formulaciones(r, tmp_path / "f.png")
    assert (tmp_path / "f.png").stat().st_size > 1000
