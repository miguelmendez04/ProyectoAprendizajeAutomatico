"""Pruebas del lector de exports manuales de SPADIES."""
import json
import shutil

import pandas as pd
import pytest

from src.extract.spadies import leer_spadies

CFG = {
    "spadies": {
        "tasa_en_porcentaje": True,
        "columnas": {"codigo_ies": "cod_ies", "anio": "anio", "tasa_desercion": "tasa_desercion"},
    }
}


def _csv(tmp_path, texto):
    ruta = tmp_path / "spadies.csv"
    ruta.write_text(texto, encoding="utf-8")
    return ruta


def test_lee_y_normaliza_porcentaje_con_coma_decimal(tmp_path):
    ruta = tmp_path / "spadies.csv"
    shutil.copy("tests/fixtures/spadies_ies.csv", ruta)
    out = leer_spadies(ruta, CFG)
    assert list(out.columns) == ["cod_ies", "anio", "tasa_desercion"]
    assert out["tasa_desercion"].tolist() == pytest.approx([0.125, 0.10, 0.0825, 0.0])
    assert out["cod_ies"].tolist() == ["1101", "1101", "2833", "2833"]
    assert out["anio"].dtype == "int64"


def test_escribe_metadata_junto_al_archivo(tmp_path):
    ruta = tmp_path / "spadies.csv"
    shutil.copy("tests/fixtures/spadies_ies.csv", ruta)
    leer_spadies(ruta, CFG)
    meta = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert meta["filas"] == 4


def test_tasa_como_proporcion_no_se_divide(tmp_path):
    cfg = {"spadies": {**CFG["spadies"], "tasa_en_porcentaje": False}}
    ruta = _csv(tmp_path, "codigo_ies,anio,tasa_desercion\n1101,2015,0.125\n")
    assert leer_spadies(ruta, cfg)["tasa_desercion"].tolist() == [0.125]


def test_llave_duplicada_error(tmp_path):
    ruta = _csv(tmp_path, "codigo_ies,anio,tasa_desercion\n1101,2015,5\n1101,2015,6\n")
    with pytest.raises(ValueError, match="duplicad"):
        leer_spadies(ruta, CFG)


@pytest.mark.parametrize("valor", ["150", "-1"])
def test_tasa_fuera_de_rango_error(tmp_path, valor):
    ruta = _csv(tmp_path, f"codigo_ies,anio,tasa_desercion\n1101,2015,{valor}\n")
    with pytest.raises(ValueError, match="rango"):
        leer_spadies(ruta, CFG)


def test_columna_faltante_error_claro(tmp_path):
    ruta = _csv(tmp_path, "codigo_ies,anio\n1101,2015\n")
    with pytest.raises(KeyError, match="tasa_desercion"):
        leer_spadies(ruta, CFG)


def test_lee_excel(tmp_path):
    ruta = tmp_path / "spadies.xlsx"
    pd.DataFrame({"codigo_ies": [1101], "anio": [2015], "tasa_desercion": [12.5]}).to_excel(ruta, index=False)
    out = leer_spadies(ruta, CFG)
    assert out["tasa_desercion"].tolist() == pytest.approx([0.125])
    assert out["cod_ies"].tolist() == ["1101"]
