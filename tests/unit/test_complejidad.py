"""Complejidad medida: el script corre con un dataset de juguete, las submuestras son por IES, los exponentes y las estadísticas de las mediciones son correctos."""
import sys

import numpy as np
import pandas as pd
import pytest

import desercion_ies.analysis.complejidad as cx
from desercion_ies.evaluation import tarea2 as t2
from desercion_ies.evaluation.protocolo import prediccion_base
from tests.unit.test_sesgo_sensibilidad import _cfg as _cfg_sesgo
from tests.unit.test_sesgo_sensibilidad import _datos


def _cfg(tmp_path) -> dict:
    cfg = _cfg_sesgo(tmp_path)
    cfg["modelado"]["complejidad"] = {"repeticiones": 2, "calentamiento": 1, "fracciones": [0.5, 1.0], "semilla_submuestra": 3, "kmeans_k": 4, "afinidad_cpu": None}
    return cfg


# ---------- estadística de las mediciones ----------
def test_el_resumen_da_mediana_cuartiles_y_riq():
    r = cx.resumen_tiempos([5, 1, 3, 2, 4])
    assert (r["mediana"], r["q1"], r["q3"], r["riq"], r["n"]) == (3.0, 2.0, 4.0, 2.0, 5)


def test_el_exponente_log_log_recupera_una_potencia_conocida():
    n = np.array([500, 1000, 1500, 2000])
    for b in (1.0, 1.5, 2.0):
        e = cx.exponente_log_log(n, 3e-9 * n ** b)
        assert e["exponente"] == pytest.approx(b) and e["error_estandar"] < 1e-9 and e["r2"] == pytest.approx(1.0)
    ruidoso = cx.exponente_log_log(n, 3e-9 * n * np.array([1.0, 1.2, 0.9, 1.1]))
    assert ruidoso["error_estandar"] > 0 and 0.5 < ruidoso["exponente"] < 1.5
    with pytest.raises(ValueError, match="3 puntos"):
        cx.exponente_log_log([1, 2], [1, 2])


def test_el_exponente_teorico_de_n_log_n_esta_entre_1_y_1_5():
    e = cx.exponente_teorico_n_log_n(500, 2000)
    assert 1.0 < e < 1.3 and e == pytest.approx(np.log((2000 * np.log(2000)) / (500 * np.log(500))) / np.log(4))


# ---------- submuestras por IES ----------
def test_las_submuestras_son_de_ies_completas_anidadas_y_reproducibles():
    df = _datos()
    por_ies = df.groupby("cod_ies").size()
    s25, s50, s100 = (cx.submuestra_por_ies(df, f, 7) for f in (0.25, 0.5, 1.0))
    for s, f in ((s25, 0.25), (s50, 0.5)):
        ies = s["cod_ies"].unique()
        assert len(ies) == int(np.ceil(f * df["cod_ies"].nunique()))
        assert (s.groupby("cod_ies").size() == por_ies.loc[ies]).all()  # cada IES elegida trae TODAS sus filas: nunca se parte una IES
    assert set(s25["cod_ies"]) <= set(s50["cod_ies"]) <= set(s100["cod_ies"])  # anidadas
    assert len(s100) == len(df) and len(s25) < len(s50) < len(s100)
    pd.testing.assert_frame_equal(s50, cx.submuestra_por_ies(df, 0.5, 7))  # misma semilla, misma submuestra
    assert set(cx.submuestra_por_ies(df, 0.5, 8)["cod_ies"]) != set(s50["cod_ies"])  # otra semilla, otras IES


def test_las_filas_no_se_mezclan_entre_ies_en_la_submuestra():
    df = _datos()
    s = cx.submuestra_por_ies(df, 0.25, 1)
    original = df[df["cod_ies"].isin(s["cod_ies"].unique())].reset_index(drop=True)
    assert len(s) == len(original) and sorted(s["tasa_desercion"]) == pytest.approx(sorted(original["tasa_desercion"]))


# ---------- medición ----------
def test_medir_calienta_repite_y_resume_con_un_solo_hilo():
    llamadas, hilos = [], []

    def ajustar():
        llamadas.append("a")
        from threadpoolctl import threadpool_info

        hilos.append([b["num_threads"] for b in threadpool_info()])
        return np.ones((500, 500)) @ np.ones((500, 500))

    r = cx.medir(ajustar, lambda obj: obj.sum(), n_filas=2000, repeticiones=3, calentamiento=1, memoria=True)
    assert r["ajuste"]["n"] == 3 and len(r["ajuste_crudo"]) == 3 and r["ajuste"]["mediana"] > 0 and r["ajuste"]["q1"] <= r["ajuste"]["mediana"] <= r["ajuste"]["q3"]
    assert len(llamadas) == 1 + 3 + 2  # calentamiento + repeticiones + dos ejecuciones de memoria (tracemalloc y RSS)
    assert all(h == 1 for hs in hilos for h in hs)  # BLAS y OpenMP limitados a un hilo dentro de la medición
    assert r["memoria_pico_mib"] > 1.0  # la matriz de 500 × 500 de float64 pesa ~2 MiB
    assert r["prediccion_1000"]["mediana"] >= 0


def test_la_prediccion_se_expresa_por_mil_filas():
    import time

    r = cx.medir(lambda: None, lambda obj: time.sleep(0.02), n_filas=500, repeticiones=3, calentamiento=0)
    assert 20 * 2 * 0.8 < r["prediccion_1000"]["mediana"] < 20 * 2 * 3  # 20 ms para 500 filas = ~40 ms por 1.000 filas


def test_el_entorno_registra_cpu_ram_sistema_y_versiones():
    e = cx.entorno()
    for k in ("cpu", "nucleos_fisicos", "hilos_logicos", "ram_gib", "sistema_operativo", "python", "numpy", "scikit_learn", "xgboost", "hilos_en_la_medicion"):
        assert k in e and e[k] not in (None, "")
    assert e["python"] == sys.version.split()[0] and e["hilos_en_la_medicion"] == 1 and e["ram_gib"] > 1


def test_los_parametros_de_complejidad_se_validan():
    assert cx.parametros_complejidad({})["repeticiones"] == 5
    with pytest.raises(ValueError):
        cx.parametros_complejidad({"modelado": {"complejidad": {"fracciones": [0.5, 1.5]}}})


# ---------- líneas base ----------
def test_las_lineas_base_separadas_reproducen_las_de_los_experimentos():
    df = _datos()
    y = df["tasa_desercion"].astype(float)
    entrena, prueba = np.arange(0, 150), np.arange(150, len(df))
    t1 = df["tasa_t1"].to_numpy(float)
    naive = cx.NaiveT1().fit(t1[entrena], y.iloc[entrena].to_numpy())
    assert naive.predict(t1[prueba]) == pytest.approx(prediccion_base(df, y, entrena, prueba, "tasa_t1"))
    g = df["caracter"].astype(str) + "|" + df["sector"].astype(str)
    m = cx.MedianaGrupo().fit(g.iloc[entrena], y.iloc[entrena].to_numpy())
    esperado, _ = t2.prediccion_mediana_grupo(df, y, entrena, prueba)
    assert m.predict(g.iloc[prueba]) == pytest.approx(esperado)


# ---------- el script de medición corre con un dataset de juguete ----------
def test_todos_los_componentes_se_miden_con_un_dataset_de_juguete(tmp_path):
    cfg, df = _cfg(tmp_path), _datos()
    pc = cx.parametros_complejidad(cfg)
    comps = cx.componentes(cfg)
    nombres = [c["nombre"] for c in comps]
    assert len(nombres) == len(set(nombres)) == 15 and {c["tipo"] for c in comps} == {"linea_base", "modelo", "preproceso"}
    assert any(n.startswith("KernelPCA") for n in nombres) and any("silueta" in n for n in nombres) and any(n.startswith("Imputación KNN") for n in nombres)
    filas = []
    for c in comps:
        filas += cx.medir_componente(c, df, pc)
    r = pd.DataFrame(filas)
    assert len(r) == 15 * 2 and set(r["fraccion_ies"]) == {0.5, 1.0} and (r["hilos"] == 1).all()
    assert r["ajuste_mediana_s"].notna().all() and (r["ajuste_mediana_s"] >= 0).all() and (r["ajuste_q1_s"] <= r["ajuste_q3_s"]).all()
    assert r.loc[r["fraccion_ies"] == 1.0, "memoria_pico_mib"].notna().all() and r.loc[r["fraccion_ies"] < 1.0, "memoria_pico_mib"].isna().all()  # memoria solo con todas las filas
    n = r.groupby("fraccion_ies")["n_filas"].first()
    assert n[1.0] == len(df) and n[0.5] < n[1.0]
    # las submuestras del script son por IES: cada fracción menor tiene menos IES y las mismas filas por IES
    assert r.groupby("fraccion_ies")["n_ies"].first()[0.5] == int(np.ceil(0.5 * df["cod_ies"].nunique()))


def test_los_exponentes_se_comparan_con_el_teorico():
    n = [100, 200, 400, 800]
    esc = pd.DataFrame([{"componente": "KernelPCA (RBF)", "tipo": "preproceso", "tarea": "T2", "n_filas": x, "ajuste_mediana_s": 1e-7 * x ** 2.5} for x in n] +
                       [{"componente": "PCA (90 % varianza)", "tipo": "preproceso", "tarea": "T2", "n_filas": x, "ajuste_mediana_s": 1e-6 * x ** 2.2} for x in n] +
                       [{"componente": "T1 C solo historia | random_forest", "tipo": "modelo", "tarea": "T1", "n_filas": x, "ajuste_mediana_s": 1e-6 * x ** 1.1} for x in n])
    e = cx.exponentes(esc).set_index("componente")
    assert e.loc["KernelPCA (RBF)", "exponente_empirico"] == pytest.approx(2.5) and e.loc["KernelPCA (RBF)", "comparacion"] == "consistente"
    assert e.loc["PCA (90 % varianza)", "comparacion"] == "por encima del teórico"  # teórico 1, medido 2,2
    assert e.loc["T1 C solo historia | random_forest", "comparacion"] == "consistente"


# ---------- costo del protocolo e informe (sobre los resultados publicados) ----------


def test_la_frontera_de_pareto_deja_los_puntos_no_dominados():
    from desercion_ies.reporting.informe_complejidad import frontera_pareto

    t, e = np.array([1.0, 2.0, 3.0, 2.5]), np.array([0.5, 0.4, 0.45, 0.3])
    assert sorted(frontera_pareto(t, e)) == [0, 1, 3]  # el punto 2 (3,0 · 0,45) lo domina el 3 (2,5 · 0,30); el 1 (2,0 · 0,40) es más barato que el 3
