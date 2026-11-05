"""Pruebas del lector de exports de SPADIES (formato real: `;`, una columna por semestre)."""
import json

import pandas as pd
import pytest

from src.extract.spadies import (
    leer_nacional,
    leer_spadies,
    leer_spadies_ies,
    tasa_anual,
    tasas_semestrales,
    validar_contra_nacional,
)

PERIODOS = ["2014-1", "2014-2", "2015-1", "2015-2", "2016-1"]


def _export(ruta, desertores, matriculados, periodos=PERIODOS):
    texto = "TIPO;" + ";".join(periodos) + "\n"
    texto += "DESERTORES;" + ";".join(map(str, desertores)) + "\n"
    texto += "MATRICULADOS;" + ";".join(map(str, matriculados)) + "\n"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(texto, encoding="utf-8")
    return ruta


def _cfg(tmp_path, periodicidad="anual"):
    return {"spadies": {"directorio": str(tmp_path / "spadies"), "nacional": str(tmp_path / "spadies" / "spadies_nacional.csv"),
                        "periodicidad": periodicidad}}


def _dos_ies(tmp_path):
    d = tmp_path / "spadies"
    _export(d / "spadies_1830_UNIVERSIDAD_AUTONOMA_DE_OCCIDENTE.csv", [10, 12, 20, 30, 40], [100, 100, 200, 200, 300])
    _export(d / "spadies_1101_UNIVERSIDAD_NACIONAL.csv", [50, 50, 60, 70, 80], [1000, 1000, 1000, 1000, 1000])
    _export(d / "spadies_nacional.csv", [100, 100, 100, 110, 130], [2000, 2000, 2000, 2000, 2000])
    return d


def test_formato_largo_con_codigo_tomado_del_nombre_del_archivo(tmp_path):
    ruta = _export(tmp_path / "spadies_1830_UAO.csv", [10, 12, 20, 30, 40], [100, 100, 200, 200, 300])
    out = leer_spadies_ies(tmp_path)
    assert list(out.columns) == ["ies", "periodo", "desertores", "matriculados"]
    assert set(out["ies"]) == {"1830"} and len(out) == 5
    assert out.loc[out["periodo"] == "2015-1", ["desertores", "matriculados"]].iloc[0].tolist() == [20, 200]
    assert ruta.exists()


def test_archivo_que_no_cumple_el_patron_se_ignora_y_el_nacional_no_es_una_ies(tmp_path):
    _export(tmp_path / "spadies_1830_UAO.csv", [1] * 5, [10] * 5)
    _export(tmp_path / "spadies_nacional.csv", [9] * 5, [90] * 5)
    _export(tmp_path / "SPADIES-OTRA.csv", [9] * 5, [90] * 5)
    assert set(leer_spadies_ies(tmp_path)["ies"]) == {"1830"}


def test_directorio_sin_exports_de_ies_error_claro(tmp_path):
    with pytest.raises(FileNotFoundError, match="spadies_<codigo>_<nombre>.csv"):
        leer_spadies_ies(tmp_path)


def test_columnas_o_filas_faltantes_error_claro(tmp_path):
    (tmp_path / "spadies_1830_UAO.csv").write_text("TIPO;2015-1\nDESERTORES;5\n", encoding="utf-8")
    with pytest.raises(KeyError, match="MATRICULADOS"):
        leer_spadies_ies(tmp_path)


def test_valor_no_numerico_error(tmp_path):
    (tmp_path / "spadies_1830_UAO.csv").write_text("TIPO;2015-1\nDESERTORES;x\nMATRICULADOS;5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no numéric"):
        leer_spadies_ies(tmp_path)


def test_ceros_iniciales_de_desertores_son_dato_faltante_no_cero(tmp_path):
    _export(tmp_path / "spadies_1830_UAO.csv", [0, 0, 20, 0, 40], [100, 100, 200, 200, 300])
    out = leer_spadies_ies(tmp_path).set_index("periodo")
    assert out.loc[["2014-1", "2014-2"], "desertores"].isna().all()
    assert out.loc["2015-2", "desertores"] == 0  # un cero posterior al primer dato sí es real


def test_tasa_semestral_usa_matriculados_dos_periodos_atras(tmp_path):
    _export(tmp_path / "spadies_1830_UAO.csv", [10, 12, 20, 30, 40], [100, 110, 200, 200, 300])
    sem = tasas_semestrales(leer_spadies_ies(tmp_path)).set_index("periodo")
    # 2015-1: 20 desertores / 100 matriculados de 2014-1 (dos semestres antes)
    assert sem.loc["2015-1", "tasa_desercion"] == pytest.approx(20 / 100)
    assert sem.loc["2015-2", "tasa_desercion"] == pytest.approx(30 / 110)
    assert sem.loc["2016-1", "tasa_desercion"] == pytest.approx(40 / 200)
    assert pd.isna(sem.loc["2014-1", "tasa_desercion"])  # no hay periodo -2
    assert sem.loc["2015-1", ["anio", "semestre"]].tolist() == [2015, 1]


def test_tasa_semestral_sin_denominador_valido_es_nula(tmp_path):
    _export(tmp_path / "spadies_1830_UAO.csv", [5, 5, 5, 5, 5], [0, 100, 100, 100, 100])
    sem = tasas_semestrales(leer_spadies_ies(tmp_path)).set_index("periodo")
    assert pd.isna(sem.loc["2015-1", "tasa_desercion"])  # matriculados(2014-1) = 0


def test_tasa_anual_promedia_los_dos_semestres_y_exige_ambos(tmp_path):
    _export(tmp_path / "spadies_1830_UAO.csv", [10, 12, 20, 30, 40], [100, 100, 200, 200, 300])
    anual = tasa_anual(tasas_semestrales(leer_spadies_ies(tmp_path)))
    assert list(anual.columns) == ["cod_ies", "anio", "tasa_desercion"]
    fila = anual[anual["anio"] == 2015].iloc[0]
    assert fila["tasa_desercion"] == pytest.approx((20 / 100 + 30 / 100) / 2)
    assert 2016 not in anual["anio"].tolist()  # solo existe 2016-1: el año queda sin tasa


def test_leer_spadies_anual_y_semestral(tmp_path):
    _dos_ies(tmp_path)
    notas: list[str] = []
    anual = leer_spadies(_cfg(tmp_path), notas)
    assert set(anual.columns) == {"cod_ies", "anio", "tasa_desercion"}
    assert set(anual["cod_ies"]) == {"1830", "1101"} and set(anual["anio"]) == {2015}
    sem = leer_spadies(_cfg(tmp_path, "semestre"), notas)
    assert {"cod_ies", "anio", "semestre", "tasa_desercion"} <= set(sem.columns)
    assert len(sem) == 6  # 2 IES x (2015-1, 2015-2, 2016-1); 2014 no tiene periodo -2
    assert sem["tasa_desercion"].between(0, 1).all()


def test_periodicidad_invalida_error(tmp_path):
    _dos_ies(tmp_path)
    with pytest.raises(ValueError, match="periodicidad"):
        leer_spadies(_cfg(tmp_path, "trimestre"), [])


def test_escribe_metadata(tmp_path):
    _dos_ies(tmp_path)
    leer_spadies(_cfg(tmp_path), [])
    meta = json.loads((tmp_path / "spadies" / "spadies_1830_UNIVERSIDAD_AUTONOMA_DE_OCCIDENTE.metadata.json").read_text(encoding="utf-8"))
    assert meta["filas"] == 5  # 5 periodos en formato largo


def test_suma_de_ies_dentro_del_total_nacional_no_genera_notas(tmp_path):
    _dos_ies(tmp_path)
    assert validar_contra_nacional(leer_spadies_ies(tmp_path / "spadies"), leer_nacional(tmp_path / "spadies" / "spadies_nacional.csv")) == []


def test_suma_de_ies_mayor_que_el_nacional_se_senala(tmp_path):
    d = _dos_ies(tmp_path)
    _export(d / "spadies_nacional.csv", [100, 100, 100, 100, 100], [2000, 2000, 2000, 2000, 2000])
    notas = validar_contra_nacional(leer_spadies_ies(d), leer_nacional(d / "spadies_nacional.csv"))
    assert len(notas) == 1 and "2016-1" in notas[0] and "desertores" in notas[0]  # 40 + 80 = 120 > 100; 2015-2 suma 100, no supera


def test_leer_spadies_agrega_notas_de_cobertura_y_validacion(tmp_path):
    d = _dos_ies(tmp_path)
    _export(d / "spadies_nacional.csv", [1] * 5, [1] * 5)
    notas: list[str] = []
    leer_spadies(_cfg(tmp_path), notas)
    assert any("2 IES" in n for n in notas) and any("supera" in n for n in notas)
