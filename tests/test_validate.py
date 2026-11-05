"""Pruebas de las reglas de calidad (pandera) y del diccionario de datos."""
import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from src.validate.dictionary import describir_columnas, generar_diccionario
from src.validate.schema import construir_esquema, validar


def _valido() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cod_divipola_mpio": ["05001", "11001", "05001"],
            "anio": [2018, 2018, 2019],
            "tasa_desercion": [0.10, 0.20, 0.15],
            "punt_global_media_t1": [250.0, 260.0, 255.0],
            "pct_estrato_bajo_t1": [0.5, 0.4, 0.5],
            "n_estudiantes_t1": [900, 800, 950],
            "mpio_matriculados_t1": [1000, 2000, 1100],
        }
    )


def test_dataset_valido_pasa():
    assert len(validar(_valido(), "municipio", max_nulos_pct=0.5)) == 3


def test_tasa_mayor_a_1_falla():
    df = _valido()
    df.loc[0, "tasa_desercion"] = 1.5
    with pytest.raises(SchemaErrors):
        validar(df, "municipio", max_nulos_pct=0.5)


def test_puntaje_fuera_de_rango_falla():
    df = _valido()
    df.loc[0, "punt_global_media_t1"] = 900.0
    with pytest.raises(SchemaErrors):
        validar(df, "municipio", max_nulos_pct=0.5)


def test_llave_duplicada_falla():
    df = pd.concat([_valido(), _valido().head(1)], ignore_index=True)
    with pytest.raises(SchemaErrors):
        validar(df, "municipio", max_nulos_pct=0.5)


def test_exceso_de_nulos_falla():
    df = _valido()
    df.loc[[0, 1], "pct_estrato_bajo_t1"] = None  # 2/3 de nulos
    with pytest.raises(SchemaErrors):
        validar(df, "municipio", max_nulos_pct=0.5)


def test_matriculados_menor_que_desertores_falla():
    df = _valido()
    df["matriculados"] = [100, 100, 100]
    df["desertores"] = [10, 200, 10]
    with pytest.raises(SchemaErrors):
        validar(df, "municipio", max_nulos_pct=0.5)


def test_objetivo_nulo_solo_se_acepta_si_no_es_obligatorio():
    df = _valido()
    df["tasa_desercion"] = float("nan")
    with pytest.raises(SchemaErrors):
        validar(df, "municipio", max_nulos_pct=0.5)
    validar(df, "municipio", max_nulos_pct=0.5, objetivo_obligatorio=False)


def test_unidad_ies_usa_su_propia_llave():
    df = pd.DataFrame({"cod_ies": ["1", "1"], "anio": [2018, 2018], "tasa_desercion": [0.1, 0.1]})
    with pytest.raises(SchemaErrors):
        validar(df, "ies", max_nulos_pct=0.5)
    assert construir_esquema("municipio", 0.5) is not None


def test_diccionario_contiene_todas_las_columnas(tmp_path):
    df = _valido()
    fuentes, descripciones = describir_columnas(list(df.columns))
    ruta = generar_diccionario(df, fuentes, descripciones, tmp_path / "data_dictionary.md")
    texto = ruta.read_text(encoding="utf-8")
    for col in df.columns:
        assert texto.count(f"| `{col}` |") == 1
    assert "0.0%" in texto  # % de nulos
    assert "Saber 11" in texto  # fuente inferida de las columnas conocidas
