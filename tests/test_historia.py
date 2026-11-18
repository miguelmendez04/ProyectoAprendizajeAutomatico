"""Variables de historia: solo años anteriores a t (sin fuga del año t ni de posteriores)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.integrate.historia import COLUMNAS_HISTORIA, agregar_historia
from src.modelado.particion import COLUMNAS_HISTORIA as HIST_MODELADO
from src.modelado.particion import columnas_explicativas

DATASET = Path("data/processed/dataset_ies.csv")


def _panel() -> pd.DataFrame:
    """A: 2015-2018 completa; B: 2016, 2018 (hueco en 2017); C: solo 2017."""
    filas = [("A", 2015, 0.10), ("A", 2016, 0.20), ("A", 2017, 0.30), ("A", 2018, 0.40),
             ("B", 2016, 0.50), ("B", 2018, 0.70), ("C", 2017, 0.90)]
    return pd.DataFrame(filas, columns=["cod_ies", "anio", "tasa_desercion"])


def _historia(df: pd.DataFrame) -> pd.DataFrame:
    return agregar_historia(df, ["cod_ies"]).set_index(["cod_ies", "anio"])


def test_valores_conocidos():
    h = _historia(_panel())
    assert np.isnan(h.loc[("A", 2015), "tasa_t1"]) and h.loc[("A", 2015), "n_anios_previos"] == 0
    assert np.isnan(h.loc[("A", 2015), "tasa_media_previa"]) and h.loc[("A", 2015), "falta_historia"]
    assert h.loc[("A", 2016), "tasa_t1"] == pytest.approx(0.10) and np.isnan(h.loc[("A", 2016), "tasa_t2"])
    assert h.loc[("A", 2018), "tasa_t1"] == pytest.approx(0.30) and h.loc[("A", 2018), "tasa_t2"] == pytest.approx(0.20)
    assert h.loc[("A", 2018), "tasa_media_previa"] == pytest.approx((0.10 + 0.20 + 0.30) / 3)
    assert h.loc[("A", 2018), "n_anios_previos"] == 3 and not h.loc[("A", 2018), "falta_historia"]


def test_con_hueco_no_se_toma_otro_anio_como_t_menos_1():
    h = _historia(_panel())
    # B en 2018: no hay fila de 2017 (t-1) -> nula, aunque exista 2016; la media y el conteo sí usan 2016
    assert np.isnan(h.loc[("B", 2018), "tasa_t1"]) and h.loc[("B", 2018), "falta_historia"]
    assert h.loc[("B", 2018), "tasa_t2"] == pytest.approx(0.50)
    assert h.loc[("B", 2018), "tasa_media_previa"] == pytest.approx(0.50) and h.loc[("B", 2018), "n_anios_previos"] == 1


def test_cada_entidad_tiene_su_propia_historia():
    h = _historia(_panel())
    assert np.isnan(h.loc[("C", 2017), "tasa_t1"]) and h.loc[("C", 2017), "n_anios_previos"] == 0  # A y B no se mezclan


def test_las_filas_conservan_su_orden_y_el_resto_de_columnas():
    df = _panel().sample(frac=1, random_state=3).reset_index(drop=True)
    out = agregar_historia(df, ["cod_ies"])
    assert out[["cod_ies", "anio", "tasa_desercion"]].equals(df)
    assert list(out.columns[-5:]) == COLUMNAS_HISTORIA


def test_las_tasas_nulas_no_cuentan_como_historia():
    df = _panel()
    df.loc[(df.cod_ies == "A") & (df.anio == 2016), "tasa_desercion"] = np.nan
    h = _historia(df)
    assert h.loc[("A", 2018), "n_anios_previos"] == 2  # 2015 y 2017
    assert h.loc[("A", 2018), "tasa_media_previa"] == pytest.approx((0.10 + 0.30) / 2)


def test_llave_duplicada_falla():
    df = pd.concat([_panel(), _panel().iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="no es único"):
        agregar_historia(df, ["cod_ies"])


# ---------- sin fuga: ninguna variable usa el año t ni posteriores ----------
def _panel_aleatorio() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    filas = [(f"I{i}", a, float(rng.uniform(0.02, 0.4))) for i in range(6) for a in range(2015, 2022) if rng.random() > 0.15]
    return pd.DataFrame(filas, columns=["cod_ies", "anio", "tasa_desercion"])


def test_la_historia_no_cambia_si_se_alteran_el_año_t_y_los_posteriores():
    """Para cada fila se reemplazan por valores arbitrarios las tasas de su IES con año >= t: su historia no debe cambiar."""
    df = _panel_aleatorio()
    base = agregar_historia(df, ["cod_ies"])
    for i, fila in df.iterrows():
        alterado = df.copy()
        futuro = (alterado["cod_ies"] == fila["cod_ies"]) & (alterado["anio"] >= fila["anio"])
        alterado.loc[futuro, "tasa_desercion"] = 0.999
        nuevo = agregar_historia(alterado, ["cod_ies"])
        cols = COLUMNAS_HISTORIA
        pd.testing.assert_series_equal(base.loc[i, cols].astype(float), nuevo.loc[i, cols].astype(float), check_names=False)


def test_la_historia_si_depende_del_pasado():
    """Contraprueba: alterar el año t-1 debe cambiar `tasa_t1` (si no, la prueba anterior no demostraría nada)."""
    df = _panel()
    alterado = df.copy()
    alterado.loc[(alterado.cod_ies == "A") & (alterado.anio == 2017), "tasa_desercion"] = 0.99
    a, b = _historia(df), _historia(alterado)
    assert a.loc[("A", 2018), "tasa_t1"] != b.loc[("A", 2018), "tasa_t1"]
    assert a.loc[("A", 2018), "tasa_media_previa"] != b.loc[("A", 2018), "tasa_media_previa"]


def test_n_anios_previos_nunca_cuenta_el_futuro():
    h = agregar_historia(_panel_aleatorio(), ["cod_ies"])
    assert (h.groupby("cod_ies")["n_anios_previos"].apply(lambda s: s.is_monotonic_increasing)).all()
    assert (h.loc[h.groupby("cod_ies")["anio"].transform("min") == h["anio"], "n_anios_previos"] == 0).all()
    assert (h["n_anios_previos"] <= h["anio"] - 2015).all()  # a lo sumo un año anterior por cada año transcurrido desde 2015


# ---------- no son explicativas por defecto ----------
def test_las_columnas_de_historia_no_entran_como_explicativas_por_defecto():
    df = agregar_historia(_panel_aleatorio().assign(x_t1=1.0), ["cod_ies"])
    for excluir in (None, []):
        explicativas = columnas_explicativas(df, excluir=excluir)
        assert explicativas == ["x_t1"]
    assert set(HIST_MODELADO) == set(COLUMNAS_HISTORIA)


# ---------- datos reales ----------
@pytest.mark.skipif(not DATASET.exists(), reason="falta data/processed/dataset_ies.csv")
def test_la_historia_del_dataset_publicado_se_recalcula_solo_con_el_pasado():
    d = pd.read_csv(DATASET, dtype={"cod_ies": str})
    assert set(COLUMNAS_HISTORIA) <= set(d.columns)
    recalculado = agregar_historia(d.drop(columns=COLUMNAS_HISTORIA), ["cod_ies"])
    for c in COLUMNAS_HISTORIA:
        pd.testing.assert_series_equal(d[c].astype(float), recalculado[c].astype(float), check_names=False)
    # cada tasa_t1 publicada es, literalmente, la tasa de la fila del año anterior
    previa = d.set_index(["cod_ies", "anio"])["tasa_desercion"]
    esperado = pd.Series(list(zip(d["cod_ies"], d["anio"] - 1))).map(previa.to_dict())
    assert d["tasa_t1"].to_numpy() == pytest.approx(esperado.to_numpy(float), nan_ok=True)
    assert (d.loc[d["anio"] == d["anio"].min(), "tasa_t1"].isna()).all()  # el primer año del dataset no tiene t-1
