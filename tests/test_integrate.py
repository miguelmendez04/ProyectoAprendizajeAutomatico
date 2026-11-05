"""Pruebas de agregación, rezago y construcción del dataset integrado."""
import numpy as np
import pandas as pd
import pytest

from src.integrate.aggregate import agregar_saber11_municipio
from src.integrate.build import construir_dataset_desde_tablas
from src.integrate.join import rezagar_features, unir_con_perdida


def test_agregacion_saber11():
    df = pd.DataFrame(
        {
            "anio": [2018] * 4,
            "cod_divipola_mpio": ["05001"] * 4,
            "punt_global": [200.0, 300.0, 250.0, 350.0],
            "estrato": pd.array([1, 2, 3, pd.NA], dtype="Int64"),
            "colegio_oficial": [True, True, False, True],
        }
    )
    out = agregar_saber11_municipio(df).iloc[0]
    assert out["punt_global_media"] == 275 and out["punt_global_mediana"] == 275
    assert out["pct_estrato_bajo"] == pytest.approx(2 / 3)  # solo estratos conocidos
    assert out["pct_colegio_oficial"] == 0.75 and out["n_estudiantes"] == 4


def test_agregacion_descarta_filas_sin_municipio():
    df = pd.DataFrame(
        {"anio": [2018, 2018], "cod_divipola_mpio": ["05001", None], "punt_global": [1.0, 2.0],
         "estrato": pd.array([1, 1], dtype="Int64"), "colegio_oficial": [True, True]}
    )
    assert len(agregar_saber11_municipio(df)) == 1


def test_rezago_sin_fuga():
    base = pd.DataFrame({"cod_divipola_mpio": "05001", "anio": range(2015, 2021)})
    feats = pd.DataFrame({"cod_divipola_mpio": "05001", "anio": range(2015, 2021), "x": [float(a) for a in range(2015, 2021)]})
    lag = rezagar_features(feats, ["cod_divipola_mpio", "anio"], ["x"], rezago=1)
    out = unir_con_perdida(base, lag, ["cod_divipola_mpio", "anio"], "x", [])
    assert "x_t1" in out.columns and "x" not in out.columns
    por_anio = out.set_index("anio")["x_t1"]
    assert por_anio[2020] == 2019.0           # la fila t recibe el valor de t-1
    assert np.isnan(por_anio[2015])           # primer año: sin dato previo


def test_unir_con_perdida_registra_log():
    izq = pd.DataFrame({"k": ["a", "b", "c"]})
    der = pd.DataFrame({"k": ["a", "b"], "v": [1, 2]})
    log = []
    out = unir_con_perdida(izq, der, ["k"], "demo", log)
    assert len(out) == 3
    assert log == [{"join": "demo", "filas_izq": 3, "filas_der": 2, "filas_resultado": 3, "filas_sin_match": 1}]


def test_unir_con_perdida_rechaza_llave_duplicada_a_la_derecha():
    with pytest.raises(ValueError, match="duplicad"):
        unir_con_perdida(pd.DataFrame({"k": ["a"]}), pd.DataFrame({"k": ["a", "a"], "v": [1, 2]}), ["k"], "x", [])


# ---- construir_dataset_desde_tablas -------------------------------------------------

def _tablas():
    """Dos IES: 1101 en Medellín (05001) y 2833 en Cali (76001). Saber 11 solo existe para Medellín."""
    catalogo = pd.DataFrame(
        {
            "cod_ies": ["1101", "1101", "2833"],
            "cod_snies_programa": ["10", "11", "20"],
            "cod_divipola_mpio": ["05001", "05001", "76001"],
            "cod_divipola_mpio_programa": ["05001", "05001", "76001"],
            "sector": ["1", "1", "2"],
            "caracter": ["4", "4", "3"],
        }
    )
    matricula_programa = pd.DataFrame(
        {
            "cod_ies": ["1101", "1101", "2833", "1101", "1101", "2833"],
            "cod_snies_programa": ["10", "11", "20", "10", "11", "20"],
            "anio": [2017, 2017, 2017, 2018, 2018, 2018],
            "matriculados": [100, 50, 70, 110, 60, 80],
        }
    )
    snies_mpio = pd.DataFrame(
        {"cod_divipola_mpio": ["05001", "76001"] * 2, "anio": [2017, 2017, 2018, 2018],
         "matriculados": [1000, 700, 1100, 800], "ies_con_oferta": [48, 20, 49, 21]}
    )
    saber = pd.DataFrame(
        {"cod_divipola_mpio": ["05001", "05001"], "anio": [2017, 2018], "punt_global_media": [250.0, 255.0],
         "punt_global_mediana": [249.0, 254.0], "pct_estrato_bajo": [0.6, 0.55], "pct_colegio_oficial": [0.7, 0.7],
         "n_estudiantes": [900, 950]}
    )
    objetivo = pd.DataFrame(
        {"cod_ies": ["1101", "2833", "1101", "2833"], "anio": [2018, 2018, 2019, 2019],
         "tasa_desercion": [0.10, 0.20, 0.12, 0.22]}
    )
    return dict(saber_mpio=saber, snies_mpio=snies_mpio, snies_programa=matricula_programa, catalogo=catalogo, objetivo=objetivo)


def test_municipio_sin_saber11_queda_nulo_no_cero():
    df, log = construir_dataset_desde_tablas("municipio", [2018], **_tablas())
    cali = df[df["cod_divipola_mpio"] == "76001"].iloc[0]
    medellin = df[df["cod_divipola_mpio"] == "05001"].iloc[0]
    assert np.isnan(cali["punt_global_media_t1"])
    assert medellin["punt_global_media_t1"] == 250.0  # Saber 2017 para predecir 2018
    saber = next(x for x in log if x["join"] == "saber11")
    assert saber["filas_sin_match"] == 1


def test_unidad_ies_y_municipio_tienen_llaves_unicas_y_distintas():
    t = _tablas()
    ies, _ = construir_dataset_desde_tablas("ies", [2018, 2019], **t)
    mpio, _ = construir_dataset_desde_tablas("municipio", [2018, 2019], **t)
    assert not ies.duplicated(["cod_ies", "anio"]).any() and len(ies) == 4
    assert not mpio.duplicated(["cod_divipola_mpio", "anio"]).any() and len(mpio) == 4
    assert "cod_ies" not in mpio.columns and "cod_divipola_mpio" in ies.columns
    fila = ies[(ies["cod_ies"] == "1101") & (ies["anio"] == 2018)].iloc[0]
    assert fila["matriculados_ies_t1"] == 150  # 100 + 50 en 2017
    assert fila["cod_divipola_mpio"] == "05001" and fila["sector"] == "1"


def test_municipio_agrega_la_tasa_de_las_ies_del_municipio():
    t = _tablas()
    # dos IES con sede en el mismo municipio (Medellín)
    t["catalogo"].loc[t["catalogo"]["cod_ies"] == "2833", ["cod_divipola_mpio", "cod_divipola_mpio_programa"]] = "05001"
    t["objetivo"] = pd.DataFrame({"cod_ies": ["1101", "2833"], "anio": [2018, 2018], "tasa_desercion": [0.10, 0.30]})
    df, _ = construir_dataset_desde_tablas("municipio", [2018], **t)
    fila = df[df["cod_divipola_mpio"] == "05001"].iloc[0]
    assert fila["tasa_desercion"] == pytest.approx(0.20)
    assert "n_ies" not in df.columns  # se calcula con el objetivo del año t: sería fuga


def test_programa_requiere_codigo_de_programa_en_el_objetivo():
    with pytest.raises(ValueError, match="cod_snies_programa"):
        construir_dataset_desde_tablas("programa", [2018], **_tablas())


def test_programa_con_objetivo_por_programa():
    t = _tablas()
    t["objetivo"] = pd.DataFrame({"cod_ies": ["1101"], "cod_snies_programa": ["10"], "anio": [2018], "tasa_desercion": [0.1]})
    df, _ = construir_dataset_desde_tablas("programa", [2018], **t)
    assert df.iloc[0]["matriculados_programa_t1"] == 100 and df.iloc[0]["cod_divipola_mpio"] == "05001"


def test_sin_objetivo_genera_esqueleto_con_tasa_nula():
    t = _tablas()
    t["objetivo"] = None
    df, _ = construir_dataset_desde_tablas("municipio", [2018], **t)
    assert len(df) == 2 and df["tasa_desercion"].isna().all()


def test_filtra_a_la_ventana_de_anios():
    df, _ = construir_dataset_desde_tablas("ies", [2019], **_tablas())
    assert set(df["anio"]) == {2019}


def test_n_estudiantes_cuenta_solo_estudiantes_con_puntaje():
    df = pd.DataFrame(
        {"anio": [2014] * 3, "cod_divipola_mpio": ["05001"] * 3, "punt_global": [200.0, np.nan, 300.0],
         "estrato": pd.array([1, 1, 1], dtype="Int64"), "colegio_oficial": [True, True, True]}
    )
    assert agregar_saber11_municipio(df).iloc[0]["n_estudiantes"] == 2


def test_anios_con_cobertura_incompleta_se_detectan_y_se_excluyen():
    from src.integrate.aggregate import anios_saber_incompletos, excluir_anios_incompletos

    def agregado(anio, n):
        return {"cod_divipola_mpio": "05001", "anio": anio, "punt_global_media": 250.0, "n_estudiantes": n}

    saber = pd.DataFrame([agregado(2017, 540_000), agregado(2018, 32_000), agregado(2019, 560_000), agregado(2020, 15_000)])
    assert anios_saber_incompletos(saber) == [2018, 2020]  # solo calendario B
    notas = []
    limpio = excluir_anios_incompletos(saber, notas)
    assert set(limpio["anio"]) == {2017, 2019}
    assert any("2018" in n and "2020" in n for n in notas)
