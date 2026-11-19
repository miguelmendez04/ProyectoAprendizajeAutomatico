"""Ajuste anidado, particiones temporales y arranque en frío por municipio: sin fuga entre entrenamiento y prueba."""
import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import loguniform
from sklearn.metrics import r2_score
from sklearn.model_selection import ParameterSampler, RandomizedSearchCV

import src.modelado.ajuste as aj
import src.modelado.arranque_frio as af
import src.modelado.temporal as tp
from src.integrate.historia import agregar_historia
from src.modelado import tarea2 as t2
from src.modelado.experimentos import parametros_experimentos
from src.modelado.protocolo import particiones_repetidas

MUNICIPIOS = ["11001", "05001", "76001", "68001", "13001", "23001", "08001", "66001"]


def _datos(n_ies: int = 40, semilla: int = 0) -> pd.DataFrame:
    """Panel 2015–2021 con historia, municipio sede, sector y carácter (texto)."""
    rng = np.random.default_rng(semilla)
    filas = []
    for i in range(n_ies):
        m = MUNICIPIOS[i % len(MUNICIPIOS)] if i % 4 else "11001"  # Bogotá concentra más IES
        base = rng.uniform(0.05, 0.25)
        for a in range(2015, 2022):
            filas.append({"cod_ies": str(1000 + i), "anio": a, "cod_divipola_mpio": m, "sector": str(1 + i % 2), "caracter": str(1 + i % 4),
                          "tasa_desercion": float(np.clip(base + rng.normal(0, 0.02), 0.01, 0.9)), "x1": rng.normal() + base * 5, "x2": rng.normal(),
                          "x3": rng.uniform(0, 1), "falta_saber11": a == 2020})
    return agregar_historia(pd.DataFrame(filas), ["cod_ies"])


def _cfg(tmp_path, **ajuste) -> dict:
    csv = tmp_path / "conf.csv"
    pd.DataFrame([{"tipo": "modelo", "modelo": "ridge", "objetivo": "logit", "imputacion": "mediana", "reduccion": "info_mutua", "clusters": False, "mae_media": 0.04},
                  {"tipo": "modelo", "modelo": "xgboost", "objetivo": "ninguna", "imputacion": "mediana", "reduccion": "ninguna", "clusters": False, "mae_media": 0.05}]).to_csv(csv, index=False)
    a = {"semillas_externas": [301, 302], "n_splits_externo": 3, "n_splits_interno": 2, "n_iter": 3, "n_jobs": 1, "k_info_mutua": [3, 5], "confirmacion_tarea2": str(csv),
         "configuraciones_tarea1": [
             {"nombre": "T1 C solo historia | ridge", "formulacion": "C_correccion_solo_historia", "modelo": "ridge", "objetivo": "ninguna", "seleccion": "ninguna", "imputacion": "mediana"},
             {"nombre": "T1 B historia+contexto | xgboost logit", "formulacion": "B_historia_contexto", "modelo": "xgboost", "objetivo": "logit", "seleccion": "ninguna", "imputacion": "mediana"}],
         "particiones_temporales": {"prepandemia": {"entrena": [2015, 2017], "prueba": [2018, 2019]}, "estres": {"entrena": [2015, 2019], "prueba": [2020, 2021]}}}
    a.update(ajuste)
    return {"particion": {"n_splits": 3}, "modelado": {"excluir_variables": [], "experimentos": {"semilla": 3, "k_info_mutua": 4},
            "tarea2": {"semilla_cribado": 11, "semillas_confirmacion": [201, 202], "n_splits": 3, "n_jobs": 1, "autoencoder": {"max_iter": 30}, "clusters": {"k_min": 3, "k_max": 4, "n_init": 2}},
            "ajuste_temporal": a}}


# ---------- espacios y parámetros ----------
def test_los_espacios_de_busqueda_son_los_del_enunciado():
    e = aj.ESPACIOS
    assert set(e["ridge"]) == {"alpha"}
    a = e["ridge"]["alpha"].rvs(2000, random_state=0)
    assert a.min() >= 1e-3 and a.max() <= 1e3 and a.min() < 1e-2 and a.max() > 1e2  # log-uniforme sobre 6 órdenes de magnitud
    assert e["random_forest"] == {"n_estimators": [200, 500], "max_depth": [3, 5, 8, None], "min_samples_leaf": [1, 5, 10, 20], "max_features": [0.3, 0.5, "sqrt"]}
    x = e["xgboost"]
    assert x["n_estimators"] == [200, 500] and x["max_depth"] == [2, 3, 4, 6] and x["subsample"] == [0.7, 1.0]
    assert x["colsample_bytree"] == [0.5, 0.8, 1.0] and x["min_child_weight"] == [1, 5, 10]
    lr, lam = x["learning_rate"].rvs(2000, random_state=1), x["reg_lambda"].rvs(2000, random_state=2)
    assert lr.min() >= 0.01 and lr.max() <= 0.3 and lam.min() >= 0.1 and lam.max() <= 10


def test_las_semillas_externas_no_pueden_repetir_las_de_etapas_anteriores():
    with pytest.raises(ValueError, match="distintas"):
        aj.parametros_ajuste({"modelado": {"ajuste_temporal": {"semillas_externas": [201, 301]}}})
    assert aj.parametros_ajuste({})["semillas_externas"] == [301, 302, 303, 304, 305]


def test_la_configuracion_del_repo_fija_las_configuraciones_de_la_tarea_1_y_el_ajuste():
    import yaml
    from pathlib import Path

    cfg = yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8"))
    p = aj.parametros_ajuste(cfg)
    assert [(c["formulacion"], c["modelo"]) for c in p["configuraciones_tarea1"]] == [
        ("C_correccion_solo_historia", "ridge"), ("C_correccion_solo_historia", "random_forest"), ("C_correccion_solo_historia", "xgboost"), ("B_historia_contexto", "xgboost")]
    assert (p["n_splits_externo"], p["n_splits_interno"], p["n_iter"]) == (5, 3, 30)


def test_la_tarea_2_toma_la_mejor_configuracion_de_cada_modelo_de_la_confirmacion(tmp_path):
    cfg = _cfg(tmp_path)
    specs = aj.configuraciones_ajuste(cfg)
    t2s = [s for s in specs if s["tarea"] == "T2"]
    assert [s["modelo"] for s in t2s] == ["ridge", "xgboost"] and t2s[0]["reduccion"] == "info_mutua" and t2s[0]["clusters"] is False
    assert len([s for s in specs if s["tarea"] == "T1"]) == 2


# ---------- estimadores ajustables ----------
def test_el_estimador_ajustable_usa_ridge_con_alpha_y_ajusta_k_con_info_mutua(tmp_path):
    cfg = _cfg(tmp_path)
    df = _datos()
    p_exp, p_t2 = parametros_experimentos(cfg), t2.parametros_tarea2(cfg)
    spec = {"tarea": "T2", "modelo": "ridge", "objetivo": "logit", "imputacion": "mediana", "reduccion": "info_mutua", "clusters": False, "nombre": "x"}
    X, num, cat = aj.preparar(spec, cfg, df)
    est, dist = aj.estimador_ajustable(aj.construir_estimador(spec, num, cat, p_exp, p_t2), spec, [3, 5])
    assert set(dist) == {"regressor__modelo__alpha", "regressor__red__k"}
    muestra = next(iter(ParameterSampler(dist, 1, random_state=0)))
    est.set_params(**muestra).fit(X, df["tasa_desercion"].to_numpy())  # los nombres de parámetros son válidos y el ajuste corre
    assert type(est.regressor.named_steps["modelo"]).__name__ == "Ridge"


@pytest.mark.parametrize("modelo,esperadas", [("random_forest", {"n_estimators", "max_depth", "min_samples_leaf", "max_features"}),
                                              ("xgboost", {"n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "min_child_weight", "reg_lambda"})])
def test_los_espacios_de_arboles_se_aplican_al_modelo_del_pipeline(tmp_path, modelo, esperadas):
    cfg = _cfg(tmp_path)
    df = _datos()
    spec = {"tarea": "T2", "modelo": modelo, "objetivo": "ninguna", "imputacion": "mediana", "reduccion": "ninguna", "clusters": False, "nombre": "x"}
    X, num, cat = aj.preparar(spec, cfg, df)
    est, dist = aj.estimador_ajustable(aj.construir_estimador(spec, num, cat, parametros_experimentos(cfg), t2.parametros_tarea2(cfg)), spec, [3, 5])
    assert {k.split("__")[-1] for k in dist} == esperadas and all(k.startswith("modelo__") for k in dist)  # sin TransformedTargetRegressor
    est.set_params(**next(iter(ParameterSampler(dist, 1, random_state=0))))


# ---------- la búsqueda interna no ve la prueba externa ----------
def test_la_busqueda_interna_solo_recibe_el_entrenamiento_externo(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    df = _datos()
    p, p_exp, p_t2 = aj.parametros_ajuste(cfg), parametros_experimentos(cfg), t2.parametros_tarea2(cfg)
    spec = {"tarea": "T2", "modelo": "ridge", "objetivo": "ninguna", "imputacion": "mediana", "reduccion": "ninguna", "clusters": False, "nombre": "x"}
    X, num, cat = aj.preparar(spec, cfg, df)
    y, grupos = df["tasa_desercion"], df["cod_ies"].astype(str).to_numpy()
    entrena, prueba = particiones_repetidas(df, 3, [301])[0][0]
    vistos = []
    original = RandomizedSearchCV.fit

    def espia(self, X_, y_=None, **kw):
        vistos.append({"indice": set(X_.index), "n_y": len(y_), "grupos": set(kw["groups"])})
        return original(self, X_, y_, **kw)

    monkeypatch.setattr(RandomizedSearchCV, "fit", espia)
    aj.ajustar_y_evaluar(spec, X, y, grupos, entrena, prueba, num, cat, p_exp, p_t2, p)
    assert len(vistos) == 1
    v = vistos[0]
    assert v["indice"] == set(X.index[entrena]) and not v["indice"] & set(X.index[prueba]) and v["n_y"] == len(entrena)
    assert not v["grupos"] & set(grupos[prueba])  # ninguna IES de la prueba externa participa en la búsqueda


def test_el_resultado_de_la_busqueda_no_depende_de_los_objetivos_de_la_prueba_externa(tmp_path):
    cfg = _cfg(tmp_path)
    df = _datos()
    p, p_exp, p_t2 = aj.parametros_ajuste(cfg), parametros_experimentos(cfg), t2.parametros_tarea2(cfg)
    spec = {"tarea": "T2", "modelo": "ridge", "objetivo": "ninguna", "imputacion": "mediana", "reduccion": "ninguna", "clusters": False, "nombre": "x"}
    X, num, cat = aj.preparar(spec, cfg, df)
    grupos = df["cod_ies"].astype(str).to_numpy()
    entrena, prueba = particiones_repetidas(df, 3, [301])[0][0]
    y1 = df["tasa_desercion"].copy()
    y2 = y1.copy()
    y2.iloc[prueba] = np.clip(y2.iloc[prueba] + 0.4, 0, 1)  # otros objetivos en la prueba externa
    _, a1 = aj.ajustar_y_evaluar(spec, X, y1, grupos, entrena, prueba, num, cat, p_exp, p_t2, p)
    _, a2 = aj.ajustar_y_evaluar(spec, X, y2, grupos, entrena, prueba, num, cat, p_exp, p_t2, p)
    assert a1["hiperparametros"] == a2["hiperparametros"] and a1["mae_interno"] == pytest.approx(a2["mae_interno"])
    assert a1["pred"] == pytest.approx(a2["pred"])  # el modelo elegido y sus predicciones son los mismos


def test_el_ajuste_anidado_usa_los_mismos_pliegues_externos_y_guarda_tiempos_e_hiperparametros(tmp_path):
    cfg = _cfg(tmp_path)
    res = aj.ejecutar_ajuste(cfg, _datos())
    assert (res.groupby(["repeticion", "pliegue"])["hash_prueba"].nunique() == 1).all()  # defecto, ajustado y líneas base: mismos pliegues
    n_cfg = 4  # 2 de la Tarea 1 + 2 de la Tarea 2
    ajustadas = res[res["variante"] == "ajustado"]
    assert len(ajustadas) == n_cfg * 2 * 3 and (res[res["variante"] == "defecto"].groupby("configuracion").size() == 6).all()
    assert (ajustadas["t_busqueda_s"] > 0).all() and (ajustadas["t_ajuste_s"] > 0).all() and (ajustadas["t_prediccion_s"] > 0).all()
    assert (res[res["variante"] == "defecto"]["t_ajuste_s"] > 0).all()
    assert all(set(json.loads(h)) for h in ajustadas["hiperparametros"])
    assert set(res.loc[res["variante"] == "linea_base", "configuracion"]) == {"T1 base_ingenua", "T1 base_mediana", "T2 base_mediana_grupo", "T2 base_mediana"}


def test_el_resumen_compara_con_la_linea_base_de_la_tarea_y_con_el_defecto(tmp_path):
    cfg = _cfg(tmp_path)
    res = aj.ejecutar_ajuste(cfg, _datos())
    r = aj.resumen_ajuste(res).set_index(["configuracion", "variante"])
    mae = aj.matriz(res)
    nombre = "T1 C solo historia | ridge"
    dif = mae["T1 base_ingenua"] - mae[f"{nombre} [ajustado]"]
    assert r.loc[(nombre, "ajustado"), "dif_base"] == pytest.approx(dif.mean())
    assert r.loc[(nombre, "ajustado"), "mejora_base_pct"] == pytest.approx(100 * dif.mean() / mae["T1 base_ingenua"].mean())
    assert r.loc[(nombre, "ajustado"), "dif_defecto"] == pytest.approx((mae[f"{nombre} [defecto]"] - mae[f"{nombre} [ajustado]"]).mean())
    t2n = [c for c in res["configuracion"].unique() if c.startswith("T2 | ridge")][0]
    assert r.loc[(t2n, "defecto"), "linea_base"] == "T2 base_mediana_grupo"
    hp = aj.resumen_hiperparametros(res)
    assert {"alpha", "k"} <= set(hp.loc[hp["configuracion"] == t2n, "hiperparametro"])


# ---------- particiones temporales ----------
def test_los_indices_temporales_separan_los_anios():
    df = _datos()
    e, t = tp.indices_temporales(df, (2015, 2017), (2018, 2019))
    assert set(df.iloc[e]["anio"]) == {2015, 2016, 2017} and set(df.iloc[t]["anio"]) == {2018, 2019} and not set(e) & set(t)
    with pytest.raises(ValueError, match="posteriores"):
        tp.indices_temporales(df, (2015, 2019), (2019, 2021))


def test_en_las_particiones_temporales_ningun_anio_de_prueba_entra_al_entrenamiento_ni_a_la_busqueda(tmp_path, monkeypatch):
    cfg, df = _cfg(tmp_path), _datos()
    anios = []
    original = aj.buscar

    def espia(est, dist, X, y, grupos, p):
        anios.append((set(df.loc[X.index, "anio"]), len(X)))
        return original(est, dist, X, y, grupos, p)

    monkeypatch.setattr(aj, "buscar", espia)
    pred = tp.ejecutar_temporal(cfg, df)
    assert len(anios) == 2 * 4  # 2 particiones x 4 configuraciones
    for conjunto, _ in anios:
        assert conjunto <= {2015, 2016, 2017, 2018, 2019}  # nunca 2020–2021 (la partición de estrés prueba ahí)
    assert sum(1 for c, _ in anios if c <= {2015, 2016, 2017}) == 4  # la prepandemia busca solo con 2015–2017
    for particion, (lo, hi) in {"prepandemia": (2018, 2019), "estres": (2020, 2021)}.items():
        assert pred[pred["particion"] == particion]["anio"].between(lo, hi).all()  # solo se predice en los años de prueba


def test_el_pronostico_a_un_paso_usa_la_tasa_observada_de_t_menos_1_aunque_este_en_la_prueba(tmp_path):
    cfg, df = _cfg(tmp_path), _datos()
    pred = tp.ejecutar_temporal(cfg, df)
    ing = pred[(pred["particion"] == "prepandemia") & (pred["configuracion"] == "T1 base_ingenua")].copy()
    d = df.set_index(["cod_ies", "anio"])
    ing["t1"] = [d.loc[(i, a), "tasa_t1"] for i, a in zip(ing["cod_ies"], ing["anio"])]
    tiene = ing["t1"].notna()
    assert ing.loc[tiene, "pred"].to_numpy() == pytest.approx(ing.loc[tiene, "t1"].to_numpy())
    f2019 = ing[(ing["anio"] == 2019)]
    previo = [d.loc[(i, 2018), "tasa_desercion"] for i in f2019["cod_ies"]]  # el año anterior (2018) está en el período de prueba
    assert f2019["pred"].to_numpy() == pytest.approx(np.array(previo))


# ---------- bootstrap sobre IES ----------
def _pred_sintetica(n_ies: int = 30, filas: int = 4, semilla: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    ies = np.repeat([f"i{k}" for k in range(n_ies)], filas)
    y = rng.uniform(0.02, 0.3, len(ies))
    partes = []
    for nombre, ruido in (("T1 base_ingenua", 0.04), ("T1 modelo", 0.02)):
        partes.append(pd.DataFrame({"particion": "p", "tarea": "T1", "configuracion": nombre, "variante": "linea_base" if "base" in nombre else "ajustado", "cod_ies": ies,
                                    "anio": 2018, "sector": "oficial", "caracter": "universidad", "y": y, "pred": y + rng.normal(0, ruido, len(y)),
                                    "t_ajuste_s": 0.1, "t_prediccion_s": 0.01, "t_busqueda_s": np.nan, "hiperparametros": "{}", "n_entrenamiento": 100}))
    return pd.concat(partes, ignore_index=True)


def test_el_bootstrap_remuestrea_ies_completas_y_coincide_con_el_calculo_manual():
    pred = _pred_sintetica()
    r = tp.bootstrap_ies(pred, replicas=200, semilla=7, referencia="T1 base_ingenua").set_index("configuracion")
    m = pred[pred["configuracion"] == "T1 modelo"]
    assert r.loc["T1 modelo", "mae"] == pytest.approx((m["pred"] - m["y"]).abs().mean())
    assert r.loc["T1 modelo", "rmse"] == pytest.approx(np.sqrt(((m["pred"] - m["y"]) ** 2).mean()))
    assert r.loc["T1 modelo", "r2"] == pytest.approx(r2_score(m["y"], m["pred"]))
    # cálculo manual con el mismo remuestreo por IES
    ies = np.sort(m["cod_ies"].unique())
    rng = np.random.default_rng(7)
    pesos = rng.multinomial(len(ies), np.full(len(ies), 1 / len(ies)), size=200)
    err = (m["pred"] - m["y"]).abs().groupby(m["cod_ies"]).sum().reindex(ies).to_numpy()
    n = m.groupby("cod_ies").size().reindex(ies).to_numpy()
    maes = (pesos @ err) / (pesos @ n)
    assert (r.loc["T1 modelo", "mae_ic_inf"], r.loc["T1 modelo", "mae_ic_sup"]) == pytest.approx(tuple(np.percentile(maes, [2.5, 97.5])))
    assert (pesos.sum(axis=1) == len(ies)).all()  # cada réplica tiene tantas IES como la muestra: se remuestrea por IES, no por fila


def test_el_bootstrap_es_reproducible_y_la_diferencia_con_la_referencia_tiene_el_signo_correcto():
    pred = _pred_sintetica()
    a = tp.bootstrap_ies(pred, 200, 7, referencia="T1 base_ingenua")
    b = tp.bootstrap_ies(pred, 200, 7, referencia="T1 base_ingenua")
    c = tp.bootstrap_ies(pred, 200, 8, referencia="T1 base_ingenua")
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a["mae_ic_inf"], c["mae_ic_inf"])
    f = a.set_index("configuracion")
    assert f.loc["T1 base_ingenua", "dif_ref"] == pytest.approx(0)
    assert f.loc["T1 modelo", "dif_ref"] > 0 and f.loc["T1 modelo", "prop_gana_ref"] > 0.9  # el modelo con menos ruido gana
    assert f.loc["T1 modelo", "dif_ref_ic_inf"] > 0


def test_el_error_por_grupo_suma_las_filas_y_reproduce_el_mae_global():
    p = _pred_sintetica().assign(sector=lambda d: np.where(d["cod_ies"].str[1:].astype(int) % 2 == 0, "oficial", "privada"),
                                 anio=lambda d: 2018 + d["cod_ies"].str[1:].astype(int) % 2)
    g = tp.errores_por_grupo(p)
    for variable in ("sector", "caracter", "anio"):
        s = g[(g["variable"] == variable) & (g["configuracion"] == "T1 modelo")]
        assert s["n"].sum() == len(p[p["configuracion"] == "T1 modelo"])
        assert (s["mae"] * s["n"]).sum() / s["n"].sum() == pytest.approx((p[p["configuracion"] == "T1 modelo"]["pred"] - p[p["configuracion"] == "T1 modelo"]["y"]).abs().mean())


# ---------- arranque en frío por municipio ----------
def test_con_agrupacion_por_municipio_ningun_municipio_de_prueba_esta_en_el_entrenamiento():
    df = _datos()
    for pliegues in particiones_repetidas(df, 3, [201, 202], af.GRUPO):
        for entrena, prueba in pliegues:
            assert set(df.iloc[entrena][af.GRUPO]).isdisjoint(df.iloc[prueba][af.GRUPO])
        assert sorted(np.concatenate([p for _, p in pliegues])) == list(range(len(df)))  # cada fila se evalúa una vez
    por_ies = particiones_repetidas(df, 3, [201])[0]
    assert any(set(df.iloc[e][af.GRUPO]) & set(df.iloc[t][af.GRUPO]) for e, t in por_ies)  # contraste: por IES sí se repiten ciudades


def test_el_arranque_en_frio_reporta_el_tamano_de_los_pliegues_y_donde_cae_bogota(tmp_path):
    cfg, df = _cfg(tmp_path), _datos()
    p = t2.parametros_tarea2(cfg)
    reps = particiones_repetidas(df, p["n_splits"], p["semillas_confirmacion"], af.GRUPO)
    tam = af.tamanos_pliegues(df, reps, p["semillas_confirmacion"])
    assert len(tam) == 2 * 3 and tam["n_prueba"].sum() == 2 * len(df)
    assert (tam.groupby("repeticion")["contiene_bogota"].sum() == 1).all()  # Bogotá está en exactamente un pliegue de prueba por repetición
    assert (tam["filas_bogota"].groupby(tam["repeticion"]).sum() == (df[af.GRUPO] == af.BOGOTA).sum()).all()
    assert tam["pct_prueba"].max() > 100 / 3  # Bogotá desbalancea: el pliegue que la contiene es el más grande


def test_el_arranque_en_frio_evalua_las_mismas_configuraciones_con_pliegues_por_municipio(tmp_path):
    cfg, df = _cfg(tmp_path), _datos()
    res, tam = af.ejecutar_arranque_frio(cfg, df)
    assert (res.groupby(["repeticion", "pliegue"])["hash_prueba"].nunique() == 1).all()
    assert set(res["configuracion"]) >= {t2.BASE_MEDIANA, t2.BASE_GRUPO} and (res["etapa"] == "arranque_frio_municipio").all()
    por_ies = t2.evaluar(cfg, af.configuraciones_confirmacion(cfg), "confirmacion", [201, 202], df)
    assert set(res["hash_prueba"]).isdisjoint(set(por_ies["hash_prueba"]))  # los pliegues por municipio no son los de por IES
    assert len(af.configuraciones_confirmacion(cfg)) == 2


# ---------- Bogotá aparte, informe y clases ----------
def test_el_error_por_grupo_separa_bogota_del_resto_cuando_hay_municipio():
    p = _pred_sintetica().assign(cod_divipola_mpio=lambda d: np.where(d["cod_ies"].str[1:].astype(int) % 3 == 0, "11001", "05001"))
    g = tp.errores_por_grupo(p)
    b = g[(g["variable"] == "bogota") & (g["configuracion"] == "T1 modelo")].set_index("nivel")
    assert set(b.index) == {"Bogotá", "resto"} and b["n"].sum() == len(p[p["configuracion"] == "T1 modelo"])
    m = p[(p["configuracion"] == "T1 modelo") & (p["cod_divipola_mpio"] == "11001")]
    assert b.loc["Bogotá", "mae"] == pytest.approx((m["pred"] - m["y"]).abs().mean())


def test_las_predicciones_temporales_traen_el_municipio(tmp_path):
    pred = tp.ejecutar_temporal(_cfg(tmp_path), _datos())
    assert {"cod_divipola_mpio", "cod_ies", "anio", "y", "pred", "t_ajuste_s", "t_prediccion_s"} <= set(pred.columns)


def test_la_clase_frente_a_la_linea_base_exige_significancia_en_las_dos_pruebas():
    from src.modelado.informe_ajuste import _clase

    assert _clase(0.002, 0.01, 0.03) == "supera" and _clase(-0.002, 0.01, 0.03) == "empeora"
    assert _clase(0.002, 0.01, 0.30) == "sin diferencia firme" and _clase(0.002, float("nan"), 0.01) == "sin diferencia firme"


def test_el_informe_se_genera_desde_los_csv_publicados(tmp_path):
    from pathlib import Path

    import yaml

    from src.modelado.informe_ajuste import generar_informe

    base = Path("reports/resultados")
    if not (base / "ajuste_resumen.csv").exists() or not (base / "temporal_predicciones.csv").exists():
        pytest.skip("faltan los CSV de reports/resultados")
    cfg = yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8"))
    s = {k: base / v for k, v in {"ajuste_resumen": "ajuste_resumen.csv", "hiperparametros": "ajuste_hiperparametros.csv", "temporal": "temporal.csv",
                                  "temporal_grupos": "temporal_grupos.csv", "temporal_predicciones": "temporal_predicciones.csv",
                                  "arranque_frio_resumen": "arranque_frio_municipio_resumen.csv", "arranque_frio_pliegues": "arranque_frio_pliegues.csv"}.items()}
    s.update({"informe": tmp_path / "r.md", "figura_ajuste": tmp_path / "a.png", "figura_temporal": tmp_path / "t.png"})
    texto = generar_informe(cfg, s, {"t_ajuste_s": 1.0, "t_temporal_s": 1.0, "t_frio_s": 1.0, "t_total_s": 3.0})
    assert "## Conclusión" in texto and "Nota de diagnóstico" in texto and "Arranque en frío estricto" in texto
    assert (tmp_path / "a.png").stat().st_size > 1000 and (tmp_path / "t.png").stat().st_size > 1000
