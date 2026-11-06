"""Pruebas del lector de SPADIES (export "IES: Diferenciado": `IES;TIPO;1998-1;…`, dos filas por IES)."""
import json

import pandas as pd
import pytest

from src.extract.spadies import (
    leer_diferenciado,
    leer_nacional,
    leer_spadies,
    tasa_anual,
    tasas_semestrales,
    validar_contra_nacional,
)

PERIODOS = ["2014-1", "2014-2", "2015-1", "2015-2", "2016-1"]


def _diferenciado(ruta, ies: dict, periodos=PERIODOS):
    """`ies`: {codigo: (desertores, matriculados)}. Con BOM, como el export real."""
    texto = "IES;TIPO;" + ";".join(periodos) + "\n"
    for cod, (des, mat) in ies.items():
        texto += f"{cod};DESERTORES;" + ";".join(map(str, des)) + "\n"
        texto += f"{cod};MATRICULADOS;" + ";".join(map(str, mat)) + "\n"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(texto, encoding="utf-8-sig")
    return ruta


def _nacional(ruta, des, mat, periodos=PERIODOS):
    ruta.write_text("TIPO;" + ";".join(periodos) + "\nDESERTORES;" + ";".join(map(str, des))
                    + "\nMATRICULADOS;" + ";".join(map(str, mat)) + "\n", encoding="utf-8")
    return ruta


def _cfg(tmp_path, periodicidad="anual"):
    return {"spadies": {"archivo": str(tmp_path / "spadies_diferenciado.csv"),
                        "nacional": str(tmp_path / "spadies_nacional.csv"), "periodicidad": periodicidad}}


UAO = ([10, 12, 20, 30, 40], [100, 110, 200, 200, 300])
UNAL = ([50, 50, 60, 70, 80], [1000, 1000, 1000, 1000, 1000])


def test_formato_largo_con_bom_y_codigo_en_la_columna_ies(tmp_path):
    ruta = _diferenciado(tmp_path / "spadies_diferenciado.csv", {"1830": UAO, "1101": UNAL})
    out = leer_diferenciado(ruta)
    assert list(out.columns) == ["ies", "periodo", "desertores", "matriculados"]
    assert set(out["ies"]) == {"1830", "1101"} and len(out) == 10
    fila = out[(out["ies"] == "1830") & (out["periodo"] == "2015-1")].iloc[0]
    assert (fila["desertores"], fila["matriculados"]) == (20, 200)


def test_ceros_iniciales_de_desertores_son_dato_faltante_no_cero(tmp_path):
    ruta = _diferenciado(tmp_path / "d.csv", {"1830": ([0, 0, 20, 0, 40], [100, 100, 200, 200, 300])})
    out = leer_diferenciado(ruta).set_index("periodo")
    assert out.loc[["2014-1", "2014-2"], "desertores"].isna().all()
    assert out.loc["2015-2", "desertores"] == 0  # un cero posterior al primer dato sí es real


def test_ies_sin_desertores_en_ningun_periodo_queda_sin_datos(tmp_path):
    ruta = _diferenciado(tmp_path / "d.csv", {"9999": ([0] * 5, [10] * 5)})
    assert leer_diferenciado(ruta)["desertores"].isna().all()


def test_falta_una_fila_de_una_ies_error_claro(tmp_path):
    ruta = tmp_path / "d.csv"
    ruta.write_text("IES;TIPO;2015-1\n1830;DESERTORES;5\n", encoding="utf-8")
    with pytest.raises(KeyError, match="1830.*MATRICULADOS"):
        leer_diferenciado(ruta)


def test_columna_faltante_o_fila_duplicada_error(tmp_path):
    ruta = tmp_path / "d.csv"
    ruta.write_text("TIPO;2015-1\nDESERTORES;5\n", encoding="utf-8")
    with pytest.raises(KeyError, match="IES"):
        leer_diferenciado(ruta)
    ruta.write_text("IES;TIPO;2015-1\n1830;DESERTORES;5\n1830;DESERTORES;6\n1830;MATRICULADOS;9\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicad"):
        leer_diferenciado(ruta)


def test_valor_no_numerico_error(tmp_path):
    ruta = tmp_path / "d.csv"
    ruta.write_text("IES;TIPO;2015-1\n1830;DESERTORES;x\n1830;MATRICULADOS;5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no numéric"):
        leer_diferenciado(ruta)


def test_tasa_semestral_usa_matriculados_dos_periodos_atras_y_los_conserva(tmp_path):
    sem = tasas_semestrales(leer_diferenciado(_diferenciado(tmp_path / "d.csv", {"1830": UAO}))).set_index("periodo")
    assert sem.loc["2015-1", "tasa_desercion"] == pytest.approx(20 / 100)  # matrícula de 2014-1
    assert sem.loc["2015-2", "tasa_desercion"] == pytest.approx(30 / 110)
    assert sem.loc["2016-1", "tasa_desercion"] == pytest.approx(40 / 200)
    assert sem.loc["2015-1", "matriculados_t2"] == 100
    assert pd.isna(sem.loc["2014-1", "tasa_desercion"])  # no hay periodo -2
    assert sem.loc["2015-1", ["anio", "semestre"]].tolist() == [2015, 1]


def test_tasa_semestral_no_cruza_ies(tmp_path):
    sem = tasas_semestrales(leer_diferenciado(_diferenciado(tmp_path / "d.csv", {"1830": UAO, "1101": UNAL})))
    un = sem[(sem["ies"] == "1101") & (sem["periodo"] == "2015-1")].iloc[0]
    assert un["tasa_desercion"] == pytest.approx(60 / 1000)


def test_tasa_semestral_sin_denominador_valido_es_nula(tmp_path):
    ruta = _diferenciado(tmp_path / "d.csv", {"1830": ([5] * 5, [0, 100, 100, 100, 100])})
    sem = tasas_semestrales(leer_diferenciado(ruta)).set_index("periodo")
    assert pd.isna(sem.loc["2015-1", "tasa_desercion"])  # matriculados(2014-1) = 0


def test_tasa_anual_es_ponderada_por_el_denominador_y_exige_ambos_semestres(tmp_path):
    anual = tasa_anual(tasas_semestrales(leer_diferenciado(_diferenciado(tmp_path / "d.csv", {"1830": UAO}))))
    assert list(anual.columns) == ["cod_ies", "anio", "tasa_desercion"]
    # (20 + 30) / (100 + 110): no es el promedio simple de 0,2000 y 0,2727
    assert anual.loc[anual["anio"] == 2015, "tasa_desercion"].iloc[0] == pytest.approx(50 / 210)
    assert 2016 not in anual["anio"].tolist()  # solo existe 2016-1


def test_leer_spadies_anual_y_semestral(tmp_path):
    _diferenciado(tmp_path / "spadies_diferenciado.csv", {"1830": UAO, "1101": UNAL})
    anual = leer_spadies(_cfg(tmp_path), [])
    assert set(anual.columns) == {"cod_ies", "anio", "tasa_desercion"} and len(anual) == 2
    sem = leer_spadies(_cfg(tmp_path, "semestre"), [])
    assert {"cod_ies", "anio", "semestre", "tasa_desercion"} <= set(sem.columns) and len(sem) == 6
    assert sem["tasa_desercion"].between(0, 1).all()


def test_periodicidad_invalida_error(tmp_path):
    _diferenciado(tmp_path / "spadies_diferenciado.csv", {"1830": UAO})
    with pytest.raises(ValueError, match="periodicidad"):
        leer_spadies(_cfg(tmp_path, "trimestre"), [])


def test_escribe_metadata(tmp_path):
    _diferenciado(tmp_path / "spadies_diferenciado.csv", {"1830": UAO, "1101": UNAL})
    leer_spadies(_cfg(tmp_path), [])
    meta = json.loads((tmp_path / "spadies_diferenciado.metadata.json").read_text(encoding="utf-8"))
    assert meta["filas"] == 10


def test_suma_de_ies_dentro_del_total_nacional_no_genera_notas(tmp_path):
    largo = leer_diferenciado(_diferenciado(tmp_path / "d.csv", {"1830": UAO, "1101": UNAL}))
    nac = leer_nacional(_nacional(tmp_path / "n.csv", [100, 100, 100, 110, 130], [2000] * 5))
    assert validar_contra_nacional(largo, nac) == []


def test_suma_de_ies_mayor_que_el_nacional_se_senala(tmp_path):
    largo = leer_diferenciado(_diferenciado(tmp_path / "d.csv", {"1830": UAO, "1101": UNAL}))
    nac = leer_nacional(_nacional(tmp_path / "n.csv", [100] * 5, [2000] * 5))
    notas = validar_contra_nacional(largo, nac)
    assert len(notas) == 1 and "2016-1" in notas[0] and "desertores" in notas[0]  # 40 + 80 = 120 > 100


def test_leer_spadies_agrega_notas_de_cobertura_y_validacion(tmp_path):
    _diferenciado(tmp_path / "spadies_diferenciado.csv", {"1830": UAO, "1101": UNAL})
    _nacional(tmp_path / "spadies_nacional.csv", [1] * 5, [1] * 5)
    notas: list[str] = []
    leer_spadies(_cfg(tmp_path), notas)
    assert any("2 IES" in n for n in notas) and any("supera" in n for n in notas)


def test_tasa_mayor_que_uno_se_senala_en_las_notas(tmp_path):
    _diferenciado(tmp_path / "spadies_diferenciado.csv", {"1830": ([5, 5, 500, 5, 5], [10] * 5)})
    notas: list[str] = []
    leer_spadies(_cfg(tmp_path, "semestre"), notas)
    assert any("mayor que 1" in n for n in notas)


# ---- IES que no llegan al dataset y denominador mínimo ------------------------------------------

def _serie(ies, periodos, desertores, matriculados):
    return pd.DataFrame({"ies": ies, "periodo": periodos, "desertores": desertores, "matriculados": matriculados})


def _periodos(desde, hasta):
    return [f"{a}-{s}" for a in range(desde, hasta + 1) for s in (1, 2)]


def test_ies_sin_tasa_en_la_ventana_se_clasifican_por_motivo():
    from src.extract.spadies import clasificar_ies_sin_tasa

    p = _periodos(2000, 2022)
    n = len(p)
    antigua = _serie("A", p[:18], [5] * 18, [100] * 18)  # reporta hasta 2008-2
    nueva = _serie("B", p[-4:], [5] * 4, [100] * 4)  # primer dato en 2021-1: su primera tasa anual es la de 2022
    un_semestre = _serie("C", p[-6:-3], [5] * 3, [100] * 3)  # datos 2020-1…2021-1: solo 2021-1 tiene denominador
    normal = _serie("D", p, [5] * n, [100] * n)
    largo = pd.concat([antigua, nueva, un_semestre, normal], ignore_index=True)
    motivos = clasificar_ies_sin_tasa(tasas_semestrales(largo), anios=[2015, 2016, 2017, 2018, 2019, 2020, 2021])
    assert motivos["dejó de reportar antes de 2015"] == ["A"]
    assert motivos["empezó a reportar tan tarde que su primera tasa anual es posterior a 2021"] == ["B"]
    assert motivos["sin ninguna tasa anual (menos de dos semestres con denominador)"] == ["C"]
    assert "D" not in sum(motivos.values(), [])


def test_min_matriculados_t2_excluye_ies_anio_con_denominador_pequeno(tmp_path):
    chica = ([5, 5, 5, 5, 5], [10, 10, 10, 10, 10])  # denominador 10
    _diferenciado(tmp_path / "spadies_diferenciado.csv", {"1830": UAO, "9999": chica})
    cfg = _cfg(tmp_path)
    cfg["min_matriculados_t2"] = 100
    notas: list[str] = []
    out = leer_spadies(cfg, notas)
    assert set(out["cod_ies"]) == {"1830"}
    assert any("min_matriculados_t2" in n and "1 IES-año" in n and "9999" in n for n in notas)
    cfg["min_matriculados_t2"] = 0
    assert set(leer_spadies(cfg, [])["cod_ies"]) == {"1830", "9999"}


def test_min_matriculados_t2_por_defecto_es_100(tmp_path):
    _diferenciado(tmp_path / "spadies_diferenciado.csv", {"1830": UAO, "9999": ([5] * 5, [10] * 5)})
    assert set(leer_spadies(_cfg(tmp_path), [])["cod_ies"]) == {"1830"}  # sin la clave en la configuración


def test_min_matriculados_t2_en_semestral_filtra_cada_fila(tmp_path):
    # el denominador de 2015-1 es 99 (matrícula de 2014-1) y el de 2015-2 es 110
    _diferenciado(tmp_path / "spadies_diferenciado.csv", {"1830": ([10, 12, 20, 30, 40], [99, 110, 200, 200, 300])})
    cfg = _cfg(tmp_path, "semestre")
    cfg["min_matriculados_t2"] = 100
    out = leer_spadies(cfg, [])
    assert out[["anio", "semestre"]].values.tolist() == [[2015, 2], [2016, 1]]  # 2015-1 queda fuera (99 < 100)


def test_notas_explican_las_ies_que_no_llegan(tmp_path):
    _diferenciado(tmp_path / "spadies_diferenciado.csv", {"1830": ([10, 12, 20, 30, 40], [100, 110, 200, 200, 300]),
                                                          "8888": ([0, 0, 0, 0, 0], [0, 0, 0, 0, 5])})
    cfg = _cfg(tmp_path)
    cfg["anios"] = [2015, 2016]
    notas: list[str] = []
    leer_spadies(cfg, notas)
    assert any("1 de 2 IES de SPADIES no tienen tasa anual en 2015–2016" in n and "8888" in n for n in notas)
