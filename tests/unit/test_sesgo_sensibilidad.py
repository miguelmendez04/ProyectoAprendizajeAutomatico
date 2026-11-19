"""Sesgo medido en los modelos finales: filtros de sensibilidad, subgrupos con bootstrap, hiperparámetros finales, diagnóstico del colapso e interpretabilidad."""

import json

import numpy as np
import pandas as pd
import pytest

import desercion_ies.analysis.diagnostico_colapso as dc
import desercion_ies.analysis.interpretabilidad as it
import desercion_ies.analysis.sensibilidad as sn
import desercion_ies.analysis.subgrupos as sg
import desercion_ies.evaluation.evaluacion_final as ef
from desercion_ies.evaluation import tarea2 as t2
from tests.unit.test_ajuste_temporal import _cfg as _cfg_ajuste
from tests.unit.test_ajuste_temporal import _datos as _datos_base

HISTORIA = {"tasa_t1", "tasa_t2", "tasa_media_previa"}


def _datos() -> pd.DataFrame:
    """Panel sintético 2015–2021 con las columnas que usan los análisis (SENA, tamaño, matrícula del municipio, historia)."""
    df = _datos_base()
    rng = np.random.default_rng(5)
    df["es_sena"] = df["cod_ies"].isin(["1003"])  # una IES es el SENA
    df["log_matriculados_spadies_t2"] = np.log1p(
        np.where(df["cod_ies"].astype(int) % 5 == 0, 150.0, rng.uniform(250, 6000, len(df)))
    )
    df["mpio_matriculados_t2"] = np.where(
        df["cod_divipola_mpio"] == "11001", 8e5, 1e5
    ) + rng.normal(0, 1e3, len(df))
    return df


def _cfg(tmp_path) -> dict:
    cfg = _cfg_ajuste(tmp_path)
    cfg["min_matriculados_t2"] = 100
    ajuste = tmp_path / "ajuste.csv"
    nombres = {
        "T1 C solo historia | ridge": '{"alpha": 5.0}',
        "T1 B historia+contexto | xgboost logit": '{"max_depth": 2, "n_estimators": 200}',
        "T2 | ridge | logit | mediana | info_mutua | sin clusters": '{"alpha": 2.0, "k": 5}',
        "T2 | xgboost | ninguna | mediana | ninguna | sin clusters": '{"max_depth": 2, "n_estimators": 200}',
    }
    pd.DataFrame(
        [
            {"configuracion": n, "variante": "ajustado", "hiperparametros": h}
            for n, h in nombres.items()
            for _ in range(3)
        ]
    ).to_csv(ajuste, index=False)
    cfg["modelado"]["sesgo_sensibilidad"] = {
        "semillas": [301, 302],
        "n_splits": 3,
        "nivel_confianza": 0.95,
        "n_jobs": 1,
        "bootstrap": {"replicas": 200, "semilla": 7},
        "ajuste": str(ajuste),
        "ciudades": {"11001": "Bogotá", "05001": "Medellín", "76001": "Cali"},
        "sensibilidad": {"umbral_denominador": 200, "anio_minimo_snies": 2017},
        "colapso": {"z_umbral": 3.0, "n_peores": 10, "recorte": 3.0, "particion": "prepandemia"},
        "permutacion": {"n_repeticiones": 2, "semilla": 301},
        "shap": {"max_display": 10, "top": 5},
    }
    cfg["modelado"]["tarea2"]["n_splits"] = 3
    return cfg


# ---------- filtros de sensibilidad ----------
def test_los_escenarios_filtran_el_dataset_ya_construido_sin_cambiarlo(tmp_path):
    cfg, df = _cfg(tmp_path), _datos()
    antes = df.copy()
    esc = sn.escenarios(df, cfg)
    assert list(esc) == sn.ESCENARIOS
    pd.testing.assert_frame_equal(df, antes)  # el dataset base no se modifica
    pd.testing.assert_frame_equal(esc["base"], df.reset_index(drop=True))
    assert not (esc["sin_bogota"]["cod_divipola_mpio"].astype(str) == "11001").any() and 0 < len(
        esc["sin_bogota"]
    ) < len(df)
    assert not esc["sin_sena"]["es_sena"].any() and len(esc["sin_sena"]) == (~df["es_sena"]).sum()
    assert (
        esc["desde_2017"]["anio"].min() == 2017
        and len(esc["desde_2017"]) == (df["anio"] >= 2017).sum()
    )


def test_el_umbral_200_deja_solo_filas_con_denominador_de_al_menos_200(tmp_path):
    cfg, df = _cfg(tmp_path), _datos()
    f = sn.escenarios(df, cfg)["umbral_200"]
    assert (np.expm1(f["log_matriculados_spadies_t2"]) >= 200 - 1e-6).all()
    esperado = (np.expm1(df["log_matriculados_spadies_t2"]) >= 200 - 1e-6).sum()
    assert len(f) == esperado and 0 < len(f) < len(df)  # quita las IES de ~150 y deja el resto


def test_no_se_puede_bajar_el_umbral_por_debajo_del_del_pipeline(tmp_path):
    from desercion_ies.analysis.mitigacion import filtrar_denominador_minimo

    with pytest.raises(ValueError, match="menor"):
        filtrar_denominador_minimo(_datos(), 50, 100)


def test_cada_escenario_conserva_las_columnas_y_la_historia_sin_recalcularla(tmp_path):
    cfg, df = _cfg(tmp_path), _datos()
    for nombre, d in sn.escenarios(df, cfg).items():
        assert list(d.columns) == list(df.columns), nombre
        m = d.merge(
            df[["cod_ies", "anio", "tasa_t1", "tasa_media_previa"]],
            on=["cod_ies", "anio"],
            suffixes=("", "_orig"),
        )
        assert len(m) == len(d)
        assert np.allclose(m["tasa_t1"].fillna(-1), m["tasa_t1_orig"].fillna(-1)) and np.allclose(
            m["tasa_media_previa"].fillna(-1), m["tasa_media_previa_orig"].fillna(-1)
        )


# ---------- hiperparámetros finales ----------
def test_los_hiperparametros_finales_son_la_moda_y_la_mediana_de_los_pliegues():
    filas = [
        {"configuracion": "c", "variante": "ajustado", "hiperparametros": json.dumps(h)}
        for h in [
            {"alpha": 1.0, "max_depth": 3, "k": 10},
            {"alpha": 4.0, "max_depth": 3, "k": 15},
            {"alpha": 100.0, "max_depth": 6, "k": 15},
            {"alpha": 2.0, "max_depth": None, "k": 10},
        ]
    ]
    hp = ef.hiperparametros_finales(pd.DataFrame(filas))["c"]
    assert hp["alpha"] == pytest.approx(3.0)  # mediana de un hiperparámetro continuo
    assert hp["max_depth"] == 3 and hp["k"] == 10  # moda; en el empate de k (10 y 15), el menor


def test_los_modelos_finales_aplican_sus_hiperparametros(tmp_path):
    from desercion_ies.evaluation import ajuste as aj
    from desercion_ies.evaluation.experimentos import parametros_experimentos

    cfg, df = _cfg(tmp_path), _datos()
    modelos = {m["nombre"]: m for m in ef.modelos_finales(cfg)}
    assert len(modelos) == 5 and list(modelos) == [f[0] for f in ef.FINALES]
    p_exp, p_t2 = parametros_experimentos(cfg), t2.parametros_tarea2(cfg)
    m = modelos["T2 ridge info_mutua (ajustado)"]
    X, num, cat = aj.preparar(m["spec"], cfg, df)
    est = ef.construir_final(m, num, cat, p_exp, p_t2)
    assert (
        type(est.regressor.named_steps["modelo"]).__name__ == "Ridge"
        and est.regressor.named_steps["modelo"].alpha == 2.0
        and est.regressor.named_steps["red"].k == 5
    )
    d = modelos["T2 ridge info_mutua (defecto)"]
    X, num, cat = aj.preparar(d["spec"], cfg, df)
    assert (
        type(ef.construir_final(d, num, cat, p_exp, p_t2).regressor.named_steps["modelo"]).__name__
        == "RidgeCV"
    )
    x = modelos[
        "T2 xgboost con clusters (ajustado)"
    ]  # en la prueba la configuración de xgboost no lleva clusters, pero sí sus hiperparámetros
    X, num, cat = aj.preparar(x["spec"], cfg, df)
    modelo_x = ef.construir_final(x, num, cat, p_exp, p_t2)
    assert (
        modelo_x.get_params()["modelo__max_depth"] == 2
        and modelo_x.get_params()["modelo__n_estimators"] == 200
    )


# ---------- evaluación fuera de muestra ----------
def test_la_evaluacion_da_una_prediccion_por_fila_y_repeticion_con_los_mismos_pliegues(tmp_path):
    cfg, df = _cfg(tmp_path), _datos()
    modelos = ef.modelos_finales(cfg)
    res, oof = ef.evaluar_oof(cfg, df, modelos)
    assert all(
        v.shape == (2, len(df)) and not np.isnan(v).any() for v in oof.values()
    )  # cada fila se predice una vez por repetición
    assert (
        res.groupby(["repeticion", "pliegue"])["hash_prueba"].nunique() == 1
    ).all()  # modelos y líneas base: mismos pliegues
    assert {
        "T1 base_ingenua",
        "T2 base_mediana_grupo",
        "T1 base_mediana",
        "T2 base_mediana",
    } <= set(res["configuracion"])
    y = df["tasa_desercion"].to_numpy()  # noqa: F841  (se conserva: documenta el valor)
    ing = oof["T1 base_ingenua"]
    t1 = df["tasa_t1"].to_numpy()
    assert ing[0][~np.isnan(t1)] == pytest.approx(
        t1[~np.isnan(t1)]
    )  # la ingenua es la tasa de t-1 cuando existe


def test_la_clase_exige_significancia_en_las_dos_pruebas():
    assert ef.clase(0.01, 0.01, 0.02) == "supera" and ef.clase(-0.01, 0.01, 0.02) == "empeora"
    assert (
        ef.clase(0.01, 0.01, 0.2) == "sin diferencia firme"
        and ef.clase(0.01, np.nan, 0.01) == "sin diferencia firme"
    )


def test_la_sensibilidad_compara_cada_escenario_con_el_base(tmp_path):
    cfg, df = _cfg(tmp_path), _datos()
    modelos = ef.modelos_finales(cfg)
    r, pl = sn.ejecutar_sensibilidad(cfg, df, modelos, parcial=tmp_path / "parcial")
    assert set(r["escenario"]) == set(sn.ESCENARIOS) and set(pl["escenario"]) == set(sn.ESCENARIOS)
    n = r.groupby("escenario")[["n_filas", "n_ies"]].first()
    assert (
        n.loc["base", "n_filas"] == len(df)
        and n.loc["sin_bogota", "n_filas"] < len(df)
        and n.loc["desde_2017", "n_filas"] == (df["anio"] >= 2017).sum()
    )
    assert not r.loc[
        r["escenario"] == "base", "cambia_conclusion"
    ].any()  # el base no cambia respecto de sí mismo
    # reanudar: lo guardado se reutiliza y da lo mismo
    r2, _ = sn.ejecutar_sensibilidad(cfg, df, modelos, parcial=tmp_path / "parcial")
    pd.testing.assert_frame_equal(r.reset_index(drop=True), r2.reset_index(drop=True))
    assert sorted(p.name for p in (tmp_path / "parcial").iterdir()) == sorted(
        f"escenario_{e}.csv" for e in sn.ESCENARIOS
    )


# ---------- subgrupos ----------
def _oof_sintetico(n_ies: int = 20, filas: int = 6, semilla: int = 0):
    rng = np.random.default_rng(semilla)
    ies = np.repeat([str(1000 + i) for i in range(n_ies)], filas)
    df = pd.DataFrame(
        {
            "cod_ies": ies,
            "tasa_desercion": rng.uniform(0.02, 0.3, len(ies)),
            "sector": np.where(np.arange(len(ies)) % 2 == 0, "1", "2"),
            "caracter": "4",
            "cod_divipola_mpio": np.where(np.arange(len(ies)) < 2 * filas, "11001", "05001"),
            "es_sena": False,
            "log_matriculados_spadies_t2": np.log1p(rng.uniform(100, 5000, len(ies))),
            "falta_historia": False,
        }
    )
    df.loc[df["cod_ies"] == "1019", "es_sena"] = True  # una sola IES
    y = df["tasa_desercion"].to_numpy()
    oof = {
        "modelo": (y + rng.normal(0, 0.01, (3, len(y)))),
        "base": (y + rng.normal(0, 0.05, (3, len(y)))),
    }
    return df, oof


def test_los_subgrupos_se_asignan_con_terciles_y_etiquetas():
    df, _ = _oof_sintetico()
    sub, cortes = sg.asignar_subgrupos(df, {"11001": "Bogotá", "05001": "Medellín"})
    assert (
        set(sub["sector"]) == {"oficial", "privada"}
        and set(sub["region"]) == {"Bogotá", "Medellín"}
        and set(sub["tamano"]) == {"tamaño bajo", "tamaño medio", "tamaño alto"}
    )
    assert (
        sub["tamano"].value_counts().min() >= len(df) // 3 - 1 and len(cortes) == 4
    )  # terciles: tres grupos casi iguales
    assert (sub["es_sena"] == "SENA").sum() == 6


def test_el_bootstrap_de_subgrupos_es_por_ies_reproducible_y_marca_los_no_evaluables(monkeypatch):
    df, oof = _oof_sintetico()
    sub, _ = sg.asignar_subgrupos(df, {"11001": "Bogotá", "05001": "Medellín"})
    monkeypatch.setitem(sg.BASES, "T1", "base")
    modelos = [{"nombre": "modelo", "tarea": "T1"}]
    a = sg.analizar_subgrupos(df, sub, oof, modelos, replicas=300, semilla=3)
    b = sg.analizar_subgrupos(df, sub, oof, modelos, replicas=300, semilla=3)
    pd.testing.assert_frame_equal(a, b)
    # punto estimado = promedio de las repeticiones del error absoluto de las filas del subgrupo
    y = df["tasa_desercion"].to_numpy()
    f = a[(a["variable"] == "sector") & (a["nivel"] == "oficial")].iloc[0]
    m = (sub["sector"] == "oficial").to_numpy()
    assert f["mae_modelo"] == pytest.approx(np.abs(oof["modelo"] - y).mean(axis=0)[m].mean())
    assert f["dif"] == pytest.approx(
        np.abs(oof["base"] - y).mean(axis=0)[m].mean() - f["mae_modelo"]
    )
    assert (
        f["dif"] > 0 and f["dif_ic_inf"] > 0 and f["clasificacion"] == "mejor (IC excluye 0)"
    )  # el modelo tiene mucho menos ruido
    sena = a[(a["variable"] == "es_sena") & (a["nivel"] == "SENA")].iloc[0]
    assert (
        sena["n_ies"] == 1
        and np.isnan(sena["dif_ic_inf"])
        and sena["clasificacion"].startswith("no evaluable")
    )  # una IES: no hay IC válido
    assert (
        a.groupby("variable")["n_filas"].sum() == len(df)
    ).all()  # los niveles de cada variable suman todas las filas


def test_un_modelo_peor_que_su_linea_base_queda_marcado(monkeypatch):
    df, oof = _oof_sintetico()
    oof = {
        "modelo": oof["base"],
        "base": oof["modelo"],
    }  # se intercambian: ahora el modelo es el ruidoso
    sub, _ = sg.asignar_subgrupos(df, {"11001": "Bogotá", "05001": "Medellín"})
    monkeypatch.setitem(sg.BASES, "T1", "base")
    a = sg.analizar_subgrupos(
        df, sub, oof, [{"nombre": "modelo", "tarea": "T1"}], replicas=300, semilla=3
    )
    f = a[(a["variable"] == "sector") & (a["nivel"] == "privada")].iloc[0]
    assert f["dif"] < 0 and f["clasificacion"] == "peor que la línea base (IC excluye 0)"


# ---------- diagnóstico del colapso ----------
def test_el_diagnostico_calcula_z_solo_con_el_entrenamiento_y_prueba_las_variantes(tmp_path):
    cfg, df = _cfg(tmp_path), _datos()
    pred = tmp_path / "pred.csv"
    spec = next(
        s
        for s in __import__(
            "desercion_ies.evaluation.ajuste", fromlist=["x"]
        ).configuraciones_ajuste(cfg)
        if s["tarea"] == "T2" and s["modelo"] == "ridge"
    )
    pd.DataFrame(
        [
            {
                "particion": "prepandemia",
                "configuracion": spec["nombre"],
                "variante": "ajustado",
                "hiperparametros": '{"alpha": 0.5, "k": 5}',
            }
        ]
    ).to_csv(pred, index=False)
    cfg["modelado"]["ajuste_temporal"]["salidas"] = {"temporal_predicciones": str(pred)}
    df = df.copy()
    df.loc[df["anio"] == 2019, "x1"] += (
        1000.0  # una variable se desplaza mucho solo en el período de prueba
    )
    resumen, z, peores = dc.ejecutar_diagnostico(cfg, df)
    assert list(resumen["variante"]) == [
        "referencia",
        "i_sin_variable",
        "ii_alfa_defecto",
        "ii_defecto_completo",
        "iii_recorte",
        "iv_solo_2017",
    ]
    assert (
        resumen.loc[resumen["variante"] == "i_sin_variable", "selecciona_variable"].iloc[0] == False  # noqa: E712  (comparación elemento a elemento de pandas)
    )
    assert (
        resumen.loc[resumen["variante"] == "iv_solo_2017", "n_entrenamiento"].iloc[0]
        == (df["anio"] == 2017).sum()
    )
    fila_x1 = z[z["variable"] == "x1"].iloc[0]
    assert (
        fila_x1["z_max_abs"] > 50 and z.iloc[0]["variable"] == "x1"
    )  # el z se mide contra la media y la desviación de 2015–2017 (sin 2019)
    assert (
        len(peores) == 10
        and peores["error"].is_monotonic_decreasing
        and {"municipio", "anio", "aporte_variable_sospechosa"} <= set(peores.columns)
    )
    assert resumen["mae"].between(0, 1).all()


def test_el_recorte_de_las_variables_escaladas_limita_a_menos_y_mas_3():
    from sklearn.preprocessing import FunctionTransformer

    f = FunctionTransformer(np.clip, kw_args={"a_min": -3.0, "a_max": 3.0})
    assert f.fit_transform(np.array([[-10.0, 0.5, 8.0]])).tolist() == [[-3.0, 0.5, 3.0]]


# ---------- interpretabilidad ----------
def test_las_columnas_del_modelo_se_agrupan_por_variable_original():
    assert (
        it._grupo("cat__sector_oficial") == "sector"
        and it._grupo("cat__caracter_universidad") == "caracter"
    )
    assert (
        it._grupo("cluster_2") == it.CLUSTERS
        and it._grupo("distancia_cluster_0") == it.CLUSTERS
        and it._grupo("num__x1") == "x1"
    )


def test_shap_permutacion_y_coeficientes_sobre_el_modelo_final(tmp_path):
    from desercion_ies.evaluation import ajuste as aj
    from desercion_ies.evaluation.experimentos import parametros_experimentos

    cfg, df = _cfg(tmp_path), _datos()
    modelos = {m["nombre"]: m for m in ef.modelos_finales(cfg)}
    m = modelos["T1 B historia+contexto | xgboost logit (ajustado)"]
    X, num, cat = aj.preparar(m["spec"], cfg, df)
    y = df["tasa_desercion"].astype(float)
    est = ef.construir_final(
        m, num, cat, parametros_experimentos(cfg), t2.parametros_tarea2(cfg)
    ).fit(X, y.to_numpy())
    sv, M, nombres = it.calcular_shap(est, X)
    assert sv.shape == M.shape == (len(df), len(nombres))
    top = it.tabla_shap(sv, M, nombres, m["nombre"], 5)
    assert (
        len(top) == 5
        and top["mean_abs_shap"].is_monotonic_decreasing
        and top.iloc[0]["variable"] in HISTORIA
        and top.iloc[0]["spearman"] > 0.5
    )  # la historia domina y, a más tasa previa, más deserción
    g = it.shap_por_variable(sv, M, nombres, m["nombre"])
    assert {"sector", "caracter"} <= set(g["variable"])
    pm = it.permutacion(cfg, df, m, 1)
    assert (
        pm.iloc[0]["variable"] in HISTORIA
        and pm["n_pliegues"].iloc[0] == 3
        and set(pm["variable"]) == set(X.columns)
    )
    r = modelos["T2 ridge info_mutua (ajustado)"]
    c = it.coeficientes_ridge(cfg, df, r, 1)
    assert (
        (c["de_pliegues"] == 6).all()
        and (c["n_pliegues_seleccionada"] <= 6).all()
        and c["signo_constante"].dtype == bool
    )


def test_la_tabla_simple_exige_importancia_por_encima_de_la_variacion_y_no_contradecir_al_ridge():
    from desercion_ies.reporting.informe_sesgo import tabla_simple

    top = pd.DataFrame(
        {
            "modelo": "m",
            "rank": [1, 2, 3, 4],
            "variable": ["a", "b", "c", "d"],
            "mean_abs_shap": [0.3, 0.2, 0.1, 0.05],
            "spearman": [0.9, 0.8, 0.7, 0.05],
        }
    )
    perm = pd.DataFrame(
        {
            "modelo": "m",
            "variable": ["a", "b", "c", "d"],
            "aumento_mae_medio": [0.02, 0.01, 0.0001, 0.0],
            "aumento_mae_sd": [0.001, 0.001, 0.005, 0.001],
        }
    )
    coef = pd.DataFrame(
        {
            "modelo": "r",
            "variable": ["a", "b"],
            "n_pliegues_seleccionada": [25, 25],
            "de_pliegues": [25, 25],
            "media": [0.5, -0.4],
            "signo_constante": [True, True],
        }
    )
    t = tabla_simple(top, perm, coef, "m", "r").set_index("variable")
    assert t.loc["a", "consistente"] == "sí"  # SHAP y ridge suben, y la permutación importa
    assert t.loc["b", "consistente"] == "no"  # SHAP dice sube y el ridge dice baja
    assert (
        t.loc["c", "consistente"] == "parcial"
    )  # su permutación no supera su variación entre pliegues
    assert t.loc["d", "direccion"].startswith("sin dirección clara")


# ---------- informe y actualización de 05_sesgo_dataset.md ----------
def test_la_actualizacion_de_analisis_sesgo_conserva_los_saltos_de_linea_y_es_idempotente(tmp_path):
    from desercion_ies.reporting.informe_sesgo import actualizar_analisis_sesgo

    crlf = chr(13) + chr(10)
    base = (
        "## Conclusiones"
        + crlf
        + crlf
        + "**Poder predictivo.** texto viejo."
        + crlf
        + crlf
        + "### Bias"
        + crlf
        + crlf
        + "**Predictive power.** old text."
    )
    ruta = tmp_path / "a.md"
    ruta.write_bytes(base.encode("utf-8"))
    actualizar_analisis_sesgo(ruta, "NUEVO ES", "NEW EN")
    t1 = ruta.read_bytes().decode("utf-8")
    assert "texto viejo" not in t1 and "old text" not in t1 and "NUEVO ES" in t1 and "NEW EN" in t1
    assert t1.count(chr(10)) == t1.count(crlf)  # no se mezclaron saltos de línea
    actualizar_analisis_sesgo(ruta, "ES 2", "EN 2")
    t2_ = ruta.read_bytes().decode("utf-8")
    assert (
        "ES 2" in t2_
        and "NUEVO ES" not in t2_
        and t2_.count("sesgo_final:inicio") == 2
        and t2_.count(chr(10)) == t2_.count(crlf)
    )
