"""Pruebas del lector y del transformador de TerriData (zips `TerriData_Dim<N>.txt.zip`, formato largo con `|`)."""
import zipfile

import numpy as np
import pandas as pd
import pytest

from src.extract.terridata import leer_terridata
from src.transform.terridata import INDICADORES, transformar_terridata

ENCABEZADO = ("Código Departamento|Departamento|Código Entidad|Entidad|Dimensión|Subcategoría|Indicador|"
              "Dato Numérico|Dato Cualitativo|Año|Mes|Fuente|Unidad de Medida")


def _fila(ent, nombre, subcat, indicador, dato, anio, cual="", dim="X"):
    return f"{ent[:2]}|Dpto|{ent}|{nombre}|{dim}|{subcat}|{indicador}|{dato}|{cual}|{anio}|12|Fuente|Unidad"


def _zip(directorio, dim, filas):
    ruta = directorio / f"TerriData_Dim{dim}.txt.zip"
    with zipfile.ZipFile(ruta, "w") as z:
        z.writestr(f"TerriData_Dim{dim}.txt", "\r\n".join([ENCABEZADO, *filas]) + "\r\n")
    return ruta


def _col(nombre):
    return next(i for i in INDICADORES if i["columna"] == nombre)


def test_catalogo_tiene_las_dimensiones_pedidas():
    columnas = {i["columna"] for i in INDICADORES}
    assert {"poblacion_hombres", "poblacion_mujeres", "nbi_pct", "ipm_censo", "cob_superior", "idf", "mdm"} <= columnas
    assert all({"columna", "dimension", "descripcion", "modo"} <= set(i) for i in INDICADORES)


def test_filtra_nivel_municipal_y_solo_indicadores_del_catalogo(tmp_path):
    ind = _col("cob_superior")
    _zip(tmp_path, 4, [
        _fila("01001", "Colombia", ind["subcategoria"], ind["indicador"], "50,00", 2018),  # nación
        _fila("76000", "Valle del Cauca", ind["subcategoria"], ind["indicador"], "40,00", 2018),  # departamento
        _fila("76001", "Cali", ind["subcategoria"], ind["indicador"], "56,70", 2018),
        _fila("76001", "Cali", "Otra", "Indicador que no se usa", "1,00", 2018),
    ])
    largo = leer_terridata(tmp_path, dimensiones=[4])
    assert largo[["cod_divipola_mpio", "anio", "columna", "valor"]].values.tolist() == [["76001", 2018, "cob_superior", 56.7]]


def test_numeros_con_separador_de_miles_y_coma_decimal(tmp_path):
    ind = _col("poblacion_hombres")
    _zip(tmp_path, 2, [_fila("05001", "Medellín", ind["subcategoria"], ind["indicador"], "1.234.567,00", 2018)])
    assert leer_terridata(tmp_path, dimensiones=[2])["valor"].tolist() == [1234567.0]


def test_dato_cualitativo_sin_numero_se_descarta(tmp_path):
    ind = _col("mdm")
    _zip(tmp_path, 8, [_fila("05001", "Medellín", ind["subcategoria"], ind["indicador"], "", 2018, cual="G3")])
    assert leer_terridata(tmp_path, dimensiones=[8]).empty


def test_zip_ausente_se_omite_y_se_anota(tmp_path):
    notas: list[str] = []
    out = leer_terridata(tmp_path, dimensiones=[4], notas=notas)
    assert out.empty and any("Dim4" in n for n in notas)


def test_duplicados_se_promedian_y_se_cuentan(tmp_path):
    ind = _col("cob_superior")
    f = _fila("76001", "Cali", ind["subcategoria"], ind["indicador"], "50,00", 2018)
    g = _fila("76001", "Cali", ind["subcategoria"], ind["indicador"], "60,00", 2018)
    _zip(tmp_path, 4, [f, g])
    notas: list[str] = []
    out = leer_terridata(tmp_path, dimensiones=[4], notas=notas)
    assert out["valor"].tolist() == [55.0] and any("duplicad" in n for n in notas)


def _largo(filas):
    return pd.DataFrame(filas, columns=["cod_divipola_mpio", "anio", "columna", "valor"])


def test_anual_exige_el_dato_del_anio_y_no_arrastra_valores_viejos():
    largo = _largo([("76001", 2017, "cob_superior", 50.0), ("76001", 2019, "cob_superior", 60.0)])
    out = transformar_terridata(largo, anios=[2018, 2019, 2020]).set_index("anio")
    assert out.loc[2017, "td_cob_superior"] == 50.0 and out.loc[2019, "td_cob_superior"] == 60.0
    assert np.isnan(out.loc[2018, "td_cob_superior"])  # sin dato de 2018 no se inventa


def test_censo_arrastra_el_ultimo_censo_ya_publicado():
    largo = _largo([("76001", 2005, "nbi_pct", 20.0), ("76001", 2018, "nbi_pct", 12.0)])
    out = transformar_terridata(largo, anios=[2016, 2017, 2018, 2019, 2020]).set_index("anio")["td_nbi_pct"]
    assert out.loc[2015:2017].tolist() == [20.0, 20.0, 20.0]  # el censo 2005 rige hasta 2017
    assert out.loc[2018:2019].tolist() == [12.0, 12.0]


def test_poblacion_se_completa_hacia_atras_con_el_primer_dato():
    largo = _largo([("76001", 2018, "poblacion_hombres", 100.0), ("76001", 2018, "poblacion_mujeres", 110.0),
                    ("76001", 2019, "poblacion_hombres", 102.0), ("76001", 2019, "poblacion_mujeres", 112.0)])
    out = transformar_terridata(largo, anios=[2016, 2017, 2018, 2019, 2020]).set_index("anio")["td_poblacion_total"]
    assert out.loc[2015] == 210.0 and out.loc[2017] == 210.0  # relleno hacia atrás con 2018 (proyección base censo 2018)
    assert out.loc[2019] == 214.0


def test_poblacion_total_requiere_hombres_y_mujeres():
    largo = _largo([("76001", 2018, "poblacion_hombres", 100.0)])
    out = transformar_terridata(largo, anios=[2019])
    assert out["td_poblacion_total"].isna().all()
    assert "td_poblacion_hombres" not in out.columns


def test_salida_cubre_los_anios_de_origen_del_rezago_t1():
    largo = _largo([("76001", 2014, "cob_superior", 1.0)])
    out = transformar_terridata(largo, anios=[2015, 2016])
    assert sorted(out["anio"].unique()) == [2014, 2015]  # t-1 de 2015 y 2016
    assert list(out.columns[:2]) == ["cod_divipola_mpio", "anio"] and out.columns[2:].tolist() == ["td_cob_superior"]


# ---- población con la base 2005 (Dim 25) -----------------------------------------------------------

def _b05(mpio, anio, **valores):
    return [(mpio, anio, col, v) for col, v in valores.items()]


def test_poblacion_anterior_a_2018_sale_de_la_base_2005_y_desde_2018_de_la_base_2018():
    filas = []
    for anio in range(2014, 2021):
        filas += _b05("76001", anio, poblacion_hombres_b05=100.0 + anio - 2014, poblacion_mujeres_b05=110.0 + anio - 2014)
    filas += _b05("76001", 2018, poblacion_hombres=200.0, poblacion_mujeres=210.0)
    filas += _b05("76001", 2019, poblacion_hombres=202.0, poblacion_mujeres=212.0)
    out = transformar_terridata(_largo(filas), anios=[2016, 2017, 2018, 2019, 2020]).set_index("anio")["td_poblacion_total"]
    assert out.loc[2015] == 100 + 1 + 110 + 1 and out.loc[2017] == 100 + 3 + 110 + 3  # base 2005
    assert out.loc[2018] == 410 and out.loc[2019] == 414  # base 2018 cuando existe (aunque la 2005 también tenga el año)


def test_pct_rural_y_edades_se_derivan_de_los_conteos_de_la_base_2005():
    filas = _b05("76001", 2016, poblacion_hombres_b05=50.0, poblacion_mujeres_b05=50.0, poblacion_rural_b05=20.0,
                 poblacion_urbana_b05=80.0, hombres_15_19_b05=5.0, mujeres_15_19_b05=3.0,
                 hombres_20_24_b05=4.0, mujeres_20_24_b05=6.0)
    out = transformar_terridata(_largo(filas), anios=[2017]).iloc[0]
    assert out["td_pct_poblacion_rural"] == pytest.approx(20.0)
    assert out["td_pct_pob_15_19"] == pytest.approx(8.0) and out["td_pct_pob_20_24"] == pytest.approx(10.0)


def test_las_columnas_de_la_base_2005_no_salen_en_el_resultado():
    filas = _b05("76001", 2016, poblacion_hombres_b05=50.0, poblacion_mujeres_b05=50.0)
    out = transformar_terridata(_largo(filas), anios=[2017])
    assert not any(c.endswith("_b05") for c in out.columns) and "td_poblacion_total" in out.columns


def test_lee_la_dimension_25(tmp_path):
    h = _col("poblacion_hombres_b05")
    _zip(tmp_path, 25, [_fila("76001", "Cali", h["subcategoria"], h["indicador"], "1.234,00", 2016, dim="25")])
    largo = leer_terridata(tmp_path, dimensiones=[25])
    assert largo[["columna", "valor", "anio"]].values.tolist() == [["poblacion_hombres_b05", 1234.0, 2016]]
