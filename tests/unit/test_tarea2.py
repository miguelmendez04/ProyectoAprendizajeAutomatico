"""Tarea 2: sin historia, sin fuga, mismos pliegues, líneas base por grupo, clusters que se agregan y selección para la confirmación."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

import desercion_ies.evaluation.tarea2 as t2
from desercion_ies.etl.integrate.historia import agregar_historia
from desercion_ies.features.particion import COLUMNAS_HISTORIA, COLUMNAS_HISTORIA_REZAGADA, NUNCA_EXPLICATIVAS


def _datos(n_ies: int = 40, semilla: int = 0) -> pd.DataFrame:
    """Panel sintético con historia agregada, columnas de NUNCA_EXPLICATIVAS y categóricas como texto."""
    rng = np.random.default_rng(semilla)
    filas = []
    for i in range(n_ies):
        base = rng.uniform(0.05, 0.25)
        for a in range(2017, 2021):
            filas.append({"cod_ies": str(1000 + i), "anio": a, "sector": str(1 + i % 2), "caracter": str(1 + i % 4),
                          "tasa_desercion": float(np.clip(base + rng.normal(0, 0.02), 0.01, 0.9)), "x1": rng.normal() + base * 5,
                          "x2": rng.normal() + (np.nan if rng.random() < 0.1 else 0), "x3": rng.uniform(0, 1), "falta_saber11": a == 2020})
    df = agregar_historia(pd.DataFrame(filas), ["cod_ies"])
    df["n_anios_ies"] = 4.0
    return df


def _cfg(**tarea2) -> dict:
    t = {"semilla_cribado": 11, "semillas_confirmacion": [201, 202], "n_splits": 3, "n_jobs": 1,
         "grilla": {"objetivo": ["ninguna"], "imputacion": ["mediana"], "reduccion": ["ninguna", "pca"], "clusters": [False, True], "modelo": ["ridge"]},
         "autoencoder": {"max_iter": 60}, "clusters": {"k_min": 3, "k_max": 5, "n_init": 3}}
    t.update(tarea2)
    return {"particion": {"n_splits": 3}, "modelado": {"excluir_variables": [], "experimentos": {"semilla": 3, "k_info_mutua": 4}, "tarea2": t}}


def _config(reduccion="ninguna", clusters=False, modelo="ridge", objetivo="ninguna", imputacion="mediana") -> dict:
    return {"objetivo": objetivo, "imputacion": imputacion, "reduccion": reduccion, "clusters": clusters, "modelo": modelo}


# ---------- grilla y parámetros ----------
def test_la_grilla_por_defecto_tiene_120_configuraciones():
    p = t2.parametros_tarea2({})
    g = t2.grilla(p)
    assert len(g) == 2 * 2 * 5 * 2 * 3 and len({t2.nombre_configuracion(c) for c in g}) == 120


def test_la_confirmacion_no_puede_reutilizar_la_semilla_del_cribado():
    with pytest.raises(ValueError, match="distintas"):
        t2.parametros_tarea2({"modelado": {"tarea2": {"semilla_cribado": 201, "semillas_confirmacion": [201, 202]}}})


def test_valores_invalidos_de_la_grilla_fallan():
    with pytest.raises(ValueError, match="reduccion"):
        t2.parametros_tarea2({"modelado": {"tarea2": {"grilla": {"reduccion": ["lda"]}}}})


def test_la_configuracion_del_repo_usa_semillas_de_confirmacion_distintas_a_las_del_cribado():
    import yaml
    from pathlib import Path

    cfg = yaml.safe_load(Path("configs/config.yaml").read_text(encoding="utf-8"))
    p = t2.parametros_tarea2(cfg)
    assert p["semilla_cribado"] not in p["semillas_confirmacion"] and len(p["semillas_confirmacion"]) == 5 and p["n_splits"] == 5


# ---------- sin historia ----------
def test_ninguna_columna_de_historia_ni_de_nunca_explicativas_entra_a_la_tarea2():
    df = _datos()
    _, X, _, numericas, categoricas = t2.preparar_datos(_cfg(), df)
    prohibidas = set(COLUMNAS_HISTORIA) | set(COLUMNAS_HISTORIA_REZAGADA) | set(NUNCA_EXPLICATIVAS)
    assert not prohibidas & set(X.columns) and not prohibidas & set(numericas + categoricas)
    cfg, p_exp = _cfg(), t2.parametros_experimentos(_cfg())
    for c in (_config(), _config("pca", True), _config("kernel_pca", True), _config("autoencoder", False)):
        pipe = t2.construir_pipeline(c, numericas, categoricas, p_exp, t2.parametros_tarea2(cfg)).fit(X, df["tasa_desercion"].to_numpy())
        interno = getattr(pipe, "regressor_", pipe)
        pre = interno.named_steps["features"].transformer_list[0][1].named_steps["pre"] if c["clusters"] else interno.named_steps["pre"]
        assert not prohibidas & set(pre.feature_names_in_)
        if c["clusters"]:
            usadas = dict(interno.named_steps["features"].transformer_list)["clusters"].pre_.feature_names_in_
            assert not prohibidas & set(usadas)


# ---------- sin fuga ----------
@pytest.mark.parametrize("c", [_config("ninguna", True), _config("kernel_pca", False), _config("autoencoder", True, imputacion="knn")])
def test_todo_se_ajusta_solo_con_el_entrenamiento(c):
    df, cfg = _datos(), _cfg()
    _, X, y, numericas, categoricas = t2.preparar_datos(cfg, df)
    p_exp, p = t2.parametros_experimentos(cfg), t2.parametros_tarea2(cfg)
    entrena, prueba = np.arange(0, 100), np.arange(100, len(df))
    X = X.copy()
    X.iloc[prueba, X.columns.get_loc("x1")] += 1000.0  # prueba extrema: no debe influir en nada ajustado
    pipe = t2.construir_pipeline(c, numericas, categoricas, p_exp, p).fit(X.iloc[entrena], y.iloc[entrena].to_numpy())
    interno = getattr(pipe, "regressor_", pipe)
    base = dict(interno.named_steps["features"].transformer_list)["base"] if c["clusters"] else interno
    escalar = base.named_steps["pre"].named_transformers_["num"].named_steps["escalar"]
    assert escalar.mean_[numericas.index("x1")] == pytest.approx(X["x1"].iloc[entrena].mean()) and escalar.mean_[numericas.index("x1")] < 10
    if c["clusters"]:
        cl = dict(interno.named_steps["features"].transformer_list)["clusters"]
        assert cl.n_entrenamiento_ == len(entrena) and cl.kmeans_.labels_.shape[0] == len(entrena)
        assert cl.pre_.transformers_[0][1].named_steps["escalar"].mean_[0] < 10
    if c["reduccion"] == "kernel_pca":
        assert base.named_steps["red"].kpca_.X_fit_.shape[0] == len(entrena)


# ---------- clusters que se agregan ----------
def test_los_clusters_agregan_2k_columnas_a_las_que_deja_la_reduccion():
    df, cfg = _datos(), _cfg()
    _, X, y, numericas, categoricas = t2.preparar_datos(cfg, df)
    p_exp, p = t2.parametros_experimentos(cfg), t2.parametros_tarea2(cfg)
    sin = t2.construir_pipeline(_config("pca", False), numericas, categoricas, p_exp, p).fit(X, y.to_numpy())
    con = t2.construir_pipeline(_config("pca", True), numericas, categoricas, p_exp, p).fit(X, y.to_numpy())
    k = t2.k_elegido(con)
    assert 3 <= k <= 5 and t2.k_elegido(sin) is None
    columnas_sin = sin.named_steps["pre"].transform(X).shape[1]
    base_pca = sin.named_steps["red"].transform(sin.named_steps["pre"].transform(X)).shape[1]
    assert con.named_steps["features"].transform(X).shape[1] == base_pca + 2 * k  # se agregan; no reemplazan
    assert columnas_sin > base_pca


# ---------- líneas base ----------
def test_la_mediana_por_grupo_sale_del_entrenamiento_y_usa_la_global_si_falta_el_grupo():
    df = pd.DataFrame({"caracter": ["1", "1", "2", "2", "3", "1"], "sector": ["1", "1", "2", "2", "1", "1"],
                       "tasa_desercion": [0.1, 0.3, 0.5, 0.7, 0.9, 0.2]})
    y = df["tasa_desercion"]
    entrena, prueba = np.array([0, 1, 2, 3]), np.array([4, 5])  # el grupo (3, 1) de la fila 4 no está en el entrenamiento
    pred, n_respaldo = t2.prediccion_mediana_grupo(df, y, entrena, prueba)
    global_ = np.median(y.iloc[entrena])
    assert pred == pytest.approx([global_, 0.2]) and n_respaldo == 1  # (1,1): mediana de 0,1 y 0,3 = 0,2
    # la tasa de la prueba (0,9) no participa en ninguna mediana
    assert 0.9 not in pred


def test_las_lineas_base_salen_con_los_mismos_pliegues_y_el_conteo_de_respaldo():
    df = _datos()
    res = t2.evaluar(_cfg(grilla={"objetivo": ["ninguna"], "imputacion": ["mediana"], "reduccion": ["ninguna"], "clusters": [False], "modelo": ["ridge"]}),
                     [_config()], "cribado", [11], df)
    assert set(res["configuracion"]) >= {t2.BASE_MEDIANA, t2.BASE_GRUPO}
    assert (res.groupby("pliegue")["hash_prueba"].nunique() == 1).all()
    assert res.loc[res["configuracion"] == t2.BASE_GRUPO, "n_respaldo_grupo"].notna().all()


# ---------- mismos pliegues y etapas ----------
def test_todas_las_configuraciones_comparten_pliegues_y_las_etapas_usan_semillas_distintas():
    df, cfg = _datos(), _cfg()
    configs = [_config("ninguna", False), _config("pca", True)]
    cri = t2.evaluar(cfg, configs, "cribado", [11], df)
    con = t2.evaluar(cfg, configs, "confirmacion", [201, 202], df)
    assert (cri.groupby(["repeticion", "pliegue"])["hash_prueba"].nunique() == 1).all()
    assert (con.groupby(["repeticion", "pliegue"])["hash_prueba"].nunique() == 1).all()
    assert con.groupby("configuracion").size().eq(2 * 3).all()
    assert set(cri["hash_prueba"]).isdisjoint(set(con["hash_prueba"]))  # pliegues distintos entre cribado y confirmación


def test_el_resultado_no_depende_del_numero_de_procesos():
    df = _datos()
    configs = [_config("ninguna", False), _config("autoencoder", True)]
    a = t2.evaluar(_cfg(n_jobs=1), configs, "cribado", [11], df).sort_values(["configuracion", "pliegue"])
    b = t2.evaluar(_cfg(n_jobs=2), configs, "cribado", [11], df).sort_values(["configuracion", "pliegue"])
    assert a["mae"].to_numpy() == pytest.approx(b["mae"].to_numpy())


# ---------- selección para la confirmación y efectos ----------
def _resumen_falso() -> pd.DataFrame:
    filas = []
    for i, (m, r, c) in enumerate((m, r, c) for m in t2.MODELOS for r in t2.REDUCCIONES for c in (False, True)):
        nombre = t2.nombre_configuracion({"modelo": m, "objetivo": "ninguna", "imputacion": "mediana", "reduccion": r, "clusters": c})
        filas.append({"configuracion": nombre, "tipo": "modelo", "modelo": m, "objetivo": "ninguna", "imputacion": "mediana", "reduccion": r,
                      "clusters": c, "mae_media": 0.04 + 0.001 * i})
    return pd.DataFrame(filas).sort_values("mae_media").reset_index(drop=True)


def test_la_seleccion_toma_la_mejor_de_cada_modelo_y_de_cada_reduccion_sin_repetir():
    r = _resumen_falso()
    elegidas = t2.seleccion_confirmacion(r)
    nombres = [t2.nombre_configuracion(c) for c in elegidas]
    assert len(nombres) == len(set(nombres)) <= 8
    assert {c["modelo"] for c in elegidas} == set(t2.MODELOS) and {c["reduccion"] for c in elegidas} == set(t2.REDUCCIONES)
    for m in t2.MODELOS:  # la elegida de cada modelo es la de menor MAE de ese modelo
        mejor = r[r["modelo"] == m].iloc[0]["configuracion"]
        assert mejor in nombres


def test_el_contraste_compara_pares_que_solo_difieren_en_el_factor():
    r = _resumen_falso()
    ef = t2.efecto_contraste(r, "reduccion", "ninguna").set_index("nivel")
    pares = r[r["tipo"] == "modelo"]
    esperado = []
    for _, f in pares[pares["reduccion"] == "pca"].iterrows():
        ref = pares[(pares["reduccion"] == "ninguna") & (pares["modelo"] == f["modelo"]) & (pares["clusters"] == f["clusters"])].iloc[0]
        esperado.append(f["mae_media"] - ref["mae_media"])
    assert ef.loc["pca", "dif_media"] == pytest.approx(np.mean(esperado)) and ef.loc["pca", "n_pares"] == len(esperado)


def test_el_resumen_de_confirmacion_compara_con_la_mediana_por_grupo():
    df = _datos()
    res = t2.evaluar(_cfg(semillas_confirmacion=[201, 202]), [_config()], "confirmacion", [201, 202], df)
    r = t2.resumen_confirmacion(res).set_index("configuracion")
    mae = t2.matriz(res)
    nombre = t2.nombre_configuracion(_config())
    dif = (mae[t2.BASE_GRUPO] - mae[nombre])
    assert r.loc[nombre, "dif_grupo"] == pytest.approx(dif.mean())
    assert r.loc[nombre, "mejora_grupo_pct"] == pytest.approx(100 * dif.mean() / mae[t2.BASE_GRUPO].mean())
    assert r.loc[t2.BASE_GRUPO, "mejora_grupo_pct"] == pytest.approx(0)
    assert r.loc[nombre, "n_ajustes"] == 6
    rho = t2.factor_nb(res)
    t = dif.mean() / np.sqrt((1 / 6 + rho) * dif.var(ddof=1))
    assert r.loc[nombre, "p_t_grupo"] == pytest.approx(2 * stats.t.sf(abs(t), 5))


# ---------- diagnóstico del autoencoder e informe de punta a punta ----------
def _datos_informe(n_ies: int = 60, semilla: int = 1) -> pd.DataFrame:
    """Panel sintético con las columnas que usa la descripción de clusters (municipio sede, tamaño y variables clave)."""
    rng = np.random.default_rng(semilla)
    municipios = ["11001", "05001", "76001", "68001", "13001", "23001"]
    filas = []
    for i in range(n_ies):
        m = municipios[i % len(municipios)]
        nivel = (i % len(municipios)) * 0.8
        for a in range(2017, 2021):
            filas.append({"cod_ies": str(1000 + i), "anio": a, "cod_divipola_mpio": m, "sector": str(1 + i % 2), "caracter": str(1 + i % 4),
                          "tasa_desercion": float(np.clip(0.08 + 0.01 * nivel + rng.normal(0, 0.02), 0.01, 0.9)),
                          "log_matriculados_spadies_t2": float(np.log1p(rng.uniform(200, 5000))), "punt_global_media_t1": 250 + 5 * nivel + rng.normal(0, 2),
                          "pct_estrato_bajo_t1": float(np.clip(0.5 + 0.05 * nivel + rng.normal(0, 0.02), 0, 1)), "td_nbi_pct_t1": 8 + 4 * nivel + rng.normal(0, 1),
                          "td_ipm_censo_t1": 20 + 5 * nivel + rng.normal(0, 1), "td_cob_bruta_media_t1": 90 + rng.normal(0, 3),
                          "td_mdm_t1": 70 - 3 * nivel + rng.normal(0, 1), "mpio_matriculados_t2": 1e4 * (6 - nivel) + rng.normal(0, 100),
                          "falta_saber11": a == 2020})
    return agregar_historia(pd.DataFrame(filas), ["cod_ies"])


def test_el_diagnostico_del_autoencoder_da_una_fila_por_pliegue():
    d = t2.diagnostico_autoencoder(_cfg(), _datos())
    assert len(d) == 3 and set(d.columns) >= {"pliegue", "n_iter", "max_iter", "paro_temprano", "mse_entrenamiento", "mse_prueba"}
    assert (d["n_iter"] <= d["max_iter"]).all() and (d["mse_entrenamiento"] > 0).all()


def test_el_informe_y_las_figuras_de_la_tarea2_se_generan(tmp_path):
    from desercion_ies.reporting.informe_tarea2 import describir_clusters, escribir_informe, figura_clusters, figura_metodologias

    df, cfg = _datos_informe(), _cfg()
    configs = [_config(r, c) for r in ("ninguna", "pca", "info_mutua") for c in (False, True)]
    cri = t2.evaluar(cfg, configs, "cribado", [11], df)
    conf = t2.evaluar(cfg, configs[:3], "confirmacion", [201, 202], df)
    r, cr = t2.resumen_cribado(cri), t2.resumen_confirmacion(conf)
    ag, dfc, etiquetas = t2.ajustar_clusters(cfg, df)
    _, X, _, numericas, _ = t2.preparar_datos(cfg, df)
    desc, medias = describir_clusters(dfc, X, etiquetas, numericas)
    assert desc["n_filas"].sum() == len(df) and len(desc) == ag.k_ and {"nombre", "tasa_mediana", "municipios"} <= set(desc.columns)
    figura_clusters(ag, X, etiquetas, desc, tmp_path / "c.png")
    figura_metodologias(r, tmp_path / "m.png")
    meta = {"filas": len(df), "ies": 60, "n_numericas": len(numericas), "k_vecinos": 5, "k_info_mutua": 4, "varianza_pca": 0.9, "n_pca": 5, "gamma_texto": "1 / n_variables",
            "ae_oculta": 16, "ae_cuello": 8, "ae_act": "tanh", "ae_val": 0.15, "k_min": 3, "k_max": 5, "semilla_cribado": 11, "semillas_confirmacion": [201, 202],
            "n_configuraciones": 6, "n_ajustes": 6, "t_cribado_s": 1.0, "t_confirmacion_s": 1.0, "t_total_s": 2.0, "sklearn": "x", "xgboost": "y",
            "k_clusters": ag.k_, "siluetas": ag.siluetas_}
    texto = escribir_informe(cri, r, conf, cr, desc, ["a", "b"], meta, tmp_path / "r.md", t2.diagnostico_autoencoder(cfg, df))
    assert "## Conclusión" in texto and "Sobre LDA" in texto and "Parada temprana del autoencoder" in texto and "## Clusters" in texto
    assert (tmp_path / "c.png").stat().st_size > 1000 and (tmp_path / "m.png").stat().st_size > 1000
