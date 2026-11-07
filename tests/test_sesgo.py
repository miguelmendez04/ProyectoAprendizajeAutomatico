"""Análisis de sesgo (src/modelado/sesgo.py) y verificación de fuga temporal."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.modelado import sesgo as S
from src.modelado.particion import cargar_dataset, columnas_explicativas

RAIZ = Path(__file__).resolve().parents[1]


def _df():
    """6 IES x 2015-2021 con distintos paneles; 2 en Bogotá, 1 en Medellín, resto en otros municipios."""
    municipios = {"1": "11001", "2": "11001", "3": "05001", "4": "08001", "5": "13001", "6": "68001"}
    anios = {"1": range(2015, 2022), "2": range(2015, 2022), "3": range(2015, 2022), "4": range(2017, 2022),
             "5": range(2015, 2019), "6": [2015, 2016, 2019, 2020, 2021]}
    filas = []
    for ies, rango in anios.items():
        for a in rango:
            filas.append({"cod_ies": ies, "anio": a, "tasa_desercion": 0.1 if ies in "123" else 0.3, "cod_divipola_mpio": municipios[ies],
                          "sector": "1" if ies in "12" else "2", "caracter": "4", "log_matriculados_spadies_t2": np.log1p(300 * int(ies))})
    return pd.DataFrame(filas)


def test_distribucion_suma_100_por_ciento():
    d = S.distribucion(_df(), "sector")
    assert d["pct_filas"].sum() == pytest.approx(1) and d["pct_ies"].sum() == pytest.approx(1)
    assert d.set_index("sector").loc["1", "ies"] == 2


def test_contraste_detecta_diferencia_y_maneja_vacios():
    c = S.contraste(np.arange(100) + 1000, np.arange(100))
    assert c["p_valor"] < 0.001 and c["prob_superioridad"] > 0.9
    assert np.isnan(S.contraste([], [1.0])["p_valor"])


def test_patron_panel_distingue_completo_entra_tarde_sale_antes_e_intermitente():
    pan = S.patron_panel(_df(), (2015, 2021)).set_index("cod_ies")
    assert pan.loc["1", "patron"] == "completo" and pan.loc["4", "patron"] == "entra tarde"
    assert pan.loc["5", "patron"] == "sale antes" and pan.loc["6", "patron"] == "intermitente"
    assert pan["panel_completo"].sum() == 3


def test_comparar_panel_separa_completos_de_incompletos():
    t = S.comparar_panel(_df(), (2015, 2021)).set_index("grupo")
    assert t.loc["panel completo", "mediana"] == pytest.approx(0.1) and t.loc["panel incompleto", "mediana"] == pytest.approx(0.3)


def test_concentracion_ciudades_cuenta_filas_e_ies():
    c = S.concentracion_ciudades(_df()).set_index("ciudad_principal")
    assert c.loc["Bogotá", "ies"] == 2 and c.loc["Medellín", "ies"] == 1 and c.loc["Cali", "ies"] == 0 or pd.isna(c.loc["Cali", "ies"])
    assert c.loc["Resto del país", "ies"] == 3


def test_dispersion_territorial_compara_ciudades_con_resto():
    df = _df().assign(x_t1=lambda d: d["cod_divipola_mpio"].map({"11001": 1.0, "05001": 1.0}).fillna(0) + d["anio"] * 0.1)
    t = S.dispersion_territorial(df, ["x_t1"])
    assert t.loc[0, "desv_ciudades"] > 0 and t.attrs["municipios_ciudades"] == 2 and t.attrs["municipios_resto"] == 3


def test_contraste_nulos_ignora_los_anios_sin_ningun_dato():
    # 100 IES (50 por sector) x 3 años; 2015 es un bloque estructural sin dato; en 2016 faltan 40 de las 50 privadas
    filas = [{"cod_ies": str(i), "anio": a, "sector": str(1 + i % 2)} for i in range(100) for a in (2015, 2016, 2017)]
    df = pd.DataFrame(filas)
    df["v_t1"] = 1.0
    df.loc[df.anio == 2015, "v_t1"] = np.nan
    privadas = df[(df.sector == "2") & (df.anio == 2016)].index[:40]
    df.loc[privadas, "v_t1"] = np.nan
    r = S.contraste_nulos(df, "v_t1", "sector")
    assert r["filas"] == int((df.anio != 2015).sum()) == 200
    assert r["pct_nulos"] == pytest.approx(40 / 200) and r["p_valor"] < 0.001
    df.loc[df.anio == 2016, "v_t1"] = 1.0  # sin falta fuera del bloque: no hay tabla que contrastar
    assert np.isnan(S.contraste_nulos(df, "v_t1", "sector")["p_valor"])


def test_nulos_por_grupo_calcula_porcentajes():
    df = _df()
    df["v"] = np.where(df.sector == "1", np.nan, 1.0)
    t = S.nulos_por_grupo(df, ["v"], "sector")
    assert t.loc["1", "v"] == 1 and t.loc["2", "v"] == 0


def test_dispersion_por_tamano_trae_la_referencia_binomial():
    t = S.dispersion_por_tamano(_df())
    assert {"desv", "desv_binomial_ref", "razon", "pct_sobre_50"} <= set(t.columns) and (t["desv_binomial_ref"] > 0).all()


# ---- clasificación de exclusiones ----------------------------------------------------------------------------
def _universo_y_exclusiones():
    universo = pd.DataFrame({
        "cod_ies": ["1", "2", "3", "4", "5", "6"], "anio": 2018, "tasa_desercion": [0.1, 0.5, 0.2, 0.3, 0.4, 0.05],
        "matriculados_t2_spadies": [500.0, 50.0, 500.0, 500.0, 500.0, 500.0],
        "posible_cierre": [False, False, True, False, False, False],
    })
    excl = pd.DataFrame({"cod_ies": ["4", "5", "6"], "anio": 2018, "inconsistencia_fuentes": [True, True, False],
                         "regimen_especial": [False, True, True], "excluida": [True, True, True]})
    dataset = pd.DataFrame({"cod_ies": ["1"], "anio": [2018]})
    return universo, excl, dataset


def test_clasificar_exclusion_sigue_el_orden_del_pipeline():
    universo, excl, dataset = _universo_y_exclusiones()
    est = S.clasificar_exclusion(universo, excl, dataset).set_index("cod_ies")["estado"]
    assert est.to_dict() == {"1": "incluida", "2": "bajo denominador mínimo", "3": "posible cierre", "4": "inconsistencia de fuentes",
                             "5": "inconsistencia de fuentes", "6": "régimen especial"}


def test_clasificar_exclusion_falla_si_no_coincide_con_el_dataset():
    universo, excl, _ = _universo_y_exclusiones()
    with pytest.raises(ValueError, match="no coinciden"):
        S.clasificar_exclusion(universo, excl, pd.DataFrame({"cod_ies": ["1", "9"], "anio": 2018}))


def test_cobertura_por_etapa_tiene_las_tres_etapas():
    universo, excl, dataset = _universo_y_exclusiones()
    u = S.clasificar_exclusion(universo, excl, dataset).assign(sector="1")
    c = S.cobertura_por_etapa(u, "sector")
    assert c["etapa"].tolist() == ["sin filtro", "tras mínimo y cierre", "dataset final"]
    assert c["filas"].tolist() == [6, 4, 1]


def test_ratio_fuera_de_rango_clasifica_por_tipo():
    excl = pd.DataFrame({"cod_ies": list("abc"), "anio": 2018, "inconsistencia_fuentes": [True, True, True],
                         "ratio_spadies_snies": [5.0, 0.1, np.nan], "tasa_desercion": [0.3, 0.2, 0.1]})
    ds = pd.DataFrame({"ratio_spadies_snies": [1.0, 2.5, 0.4], "tasa_desercion": [0.1, 0.2, 0.3]})
    t = S.ratio_fuera_de_rango(excl, ds).set_index("grupo")
    assert t.loc["ratio > umbral", "filas"] == 1 and t.loc["ratio < 1/umbral", "filas"] == 1
    assert t.loc["SNIES sin dato dentro de su cobertura", "filas"] == 1 and t.iloc[-1]["filas"] == 2


# ---- fuga temporal ------------------------------------------------------------------------------------------
def _fuentes():
    """Fuentes sintéticas para 2 IES en 2 municipios, años 2013-2019, con un valor distinto por año."""
    años = range(2013, 2020)
    saber = pd.DataFrame([{"cod_divipola_mpio": m, "anio": a, "punt": 100 * k + a} for k, m in enumerate(["11001", "05001"]) for a in años])
    mpio = pd.DataFrame([{"cod_divipola_mpio": m, "anio": a, "matriculados": 1000 * k + a, "ies_con_oferta": a} for k, m in enumerate(["11001", "05001"]) for a in años])
    prog = pd.DataFrame([{"cod_ies": i, "cod_snies_programa": 1, "anio": a, "matriculados": 10 * int(i) + a} for i in ["1", "2"] for a in años])
    td = pd.DataFrame([{"cod_divipola_mpio": m, "anio": a, "td_x": a * 2.0} for m in ["11001", "05001"] for a in años])
    obj = pd.DataFrame({"cod_ies": ["1", "2"], "anio": 2017, "matriculados_t2_spadies": [400.0, 900.0]})
    return saber, mpio, prog, td, obj


def _dataset_correcto():
    saber, mpio, prog, td, obj = _fuentes()
    base = pd.DataFrame({"cod_ies": ["1", "2"], "anio": 2017, "cod_divipola_mpio": ["11001", "05001"]})
    base["punt_t1"] = [100 * 0 + 2016, 100 * 1 + 2016]
    base["mpio_matriculados_t2"] = [0 + 2015, 1000 + 2015]
    base["mpio_ies_con_oferta_t2"] = [2015, 2015]
    base["matriculados_ies_t2"] = [10 + 2015, 20 + 2015]
    base["td_x_t1"] = [2016 * 2.0, 2016 * 2.0]
    base["log_matriculados_spadies_t2"] = np.log1p([400.0, 900.0])
    return base, (saber, mpio, prog, td, obj)


def test_verificar_rezagos_no_encuentra_discrepancias_con_los_rezagos_permitidos():
    base, (saber, mpio, prog, td, obj) = _dataset_correcto()
    saber = saber.rename(columns={"punt": "punt"})
    r = S.verificar_rezagos(base, saber, mpio, prog, td, obj)
    assert len(r) == 6 and (r["discrepancias"] == 0).all() and (r["comparadas"] > 0).all()


@pytest.mark.parametrize("columna, valor", [
    ("punt_t1", [2017, 100 + 2017]),                 # Saber 11 del mismo año t en vez de t-1
    ("td_x_t1", [2018 * 2.0, 2018 * 2.0]),           # TerriData de t+1
    ("matriculados_ies_t2", [10 + 2016, 20 + 2016]),  # matrícula de t-1 en vez de t-2
    ("mpio_matriculados_t2", [2017, 1000 + 2017]),   # matrícula municipal del mismo año
])
def test_verificar_rezagos_detecta_una_explicativa_que_usa_un_anio_no_permitido(columna, valor):
    base, (saber, mpio, prog, td, obj) = _dataset_correcto()
    base[columna] = valor
    r = S.verificar_rezagos(base, saber, mpio, prog, td, obj).set_index("columna")
    assert r.loc[columna, "discrepancias"] == 2
    assert r.drop(columna)["discrepancias"].sum() == 0


def test_explicativas_sin_rezago_senala_los_nombres_sospechosos():
    assert S.explicativas_sin_rezago(["a_t1", "b_t2", "falta_mdm", "tasa_pasada", "x"]) == ["tasa_pasada", "x"]


def test_las_explicativas_por_defecto_declaran_su_rezago():
    """Cada explicativa por defecto del dataset tiene sufijo _t1/_t2 o es un indicador de bloque de años."""
    ruta = RAIZ / "data" / "processed" / "dataset_ies.csv"
    if not ruta.exists():
        pytest.skip("no hay dataset procesado")
    cfg = {"unidad_analisis": "ies", "rutas": {"processed": str(ruta.parent)}}
    df = cargar_dataset(cfg)
    assert S.explicativas_sin_rezago(columnas_explicativas(df)) == []


def test_ninguna_explicativa_usa_un_anio_posterior_al_permitido_en_el_dataset_real():
    """Recalcula cada explicativa desde las tablas intermedias: Saber 11 y TerriData en t-1, matrícula SNIES en t-2."""
    interim = RAIZ / "data" / "interim"
    ruta = RAIZ / "data" / "processed" / "dataset_ies.csv"
    necesarios = ["saber11_municipio.csv", "snies_matricula_mpio.csv", "snies_matricula_programa.csv", "terridata_largo.csv", "spadies.csv"]
    if not ruta.exists() or not all((interim / n).exists() for n in necesarios):
        pytest.skip("faltan data/processed o data/interim (se generan con pipeline.py)")
    from src.config import load_config
    from src.transform.terridata import transformar_terridata

    cfg = load_config(RAIZ / "config" / "config.yaml")
    cod = {"cod_divipola_mpio": str, "cod_ies": str}
    df = pd.read_csv(ruta, dtype={**cod, "sector": str, "caracter": str})
    td = transformar_terridata(pd.read_csv(interim / "terridata_largo.csv", dtype=cod), cfg["anios"])
    r = S.verificar_rezagos(
        df, pd.read_csv(interim / "saber11_municipio.csv", dtype=cod), pd.read_csv(interim / "snies_matricula_mpio.csv", dtype=cod),
        pd.read_csv(interim / "snies_matricula_programa.csv", dtype=cod), td, pd.read_csv(interim / "spadies.csv", dtype=cod),
        cfg.get("rezago_matricula", 2),
    )
    assert len(r) >= 30 and r["comparadas"].min() > 0
    assert r[r["discrepancias"] > 0].empty, r[r["discrepancias"] > 0].to_string()


def test_celdas_rellenadas_hacia_atras_detecta_un_valor_tomado_de_un_anio_posterior():
    # el municipio 11001 solo tiene población desde 2018: el rezago de 2017 (origen 2016) saldría del bfill con el dato de 2018
    largo = pd.DataFrame([{"cod_divipola_mpio": "11001", "anio": a, "columna": c, "valor": 10.0 + a}
                          for a in (2018, 2019, 2020) for c in ("poblacion_hombres", "poblacion_mujeres")])
    df = pd.DataFrame({"cod_divipola_mpio": "11001", "anio": [2017, 2019]})
    r = S.celdas_rellenadas_hacia_atras(largo, [2015, 2016, 2017, 2018, 2019, 2020, 2021], df).set_index("variable")
    assert r.loc["poblacion_total", "celdas_usadas"] == 2
    assert r.loc["poblacion_total", "nulas_hacia_adelante"] == 1 and r.loc["poblacion_total", "rellenadas_con_anio_posterior"] == 1
    completo = pd.DataFrame({"cod_divipola_mpio": "11001", "anio": [2019]})
    assert S.celdas_rellenadas_hacia_atras(largo, [2015, 2016, 2017, 2018, 2019, 2020, 2021], completo)["rellenadas_con_anio_posterior"].sum() == 0
