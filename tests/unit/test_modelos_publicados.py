"""Los tres modelos guardados en `models/` cargan y predicen (en un entorno limpio falló `tarea2_mediana` por falta de pyarrow)."""

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

from desercion_ies.inference import predecir as pr

RAIZ = Path(__file__).resolve().parents[2]
MODELOS = RAIZ / "models"
EJEMPLOS = RAIZ / "examples"


def _fila_2021() -> pd.DataFrame:
    df = pd.read_csv(
        RAIZ / "data" / "processed" / "dataset_ies.csv",
        dtype={"cod_ies": str, "cod_divipola_mpio": str, "sector": str, "caracter": str},
    )
    return df[df["anio"] == 2021].head(1).reset_index(drop=True)


@pytest.mark.parametrize("clave", pr.CLAVES_PUBLICADAS)
def test_los_modelos_publicados_cargan_y_predicen_una_fila(clave):
    modelo, meta = pr.cargar_modelo(clave, MODELOS)
    salida = pr.predecir_tabla(modelo, _fila_2021(), meta["nombre"])
    assert list(salida.columns) == [
        "cod_ies",
        "anio",
        "cod_divipola_mpio",
        "modelo",
        "tasa_predicha",
    ]
    assert len(salida) == 1 and 0 <= salida["tasa_predicha"].iloc[0] <= 1


def test_pyarrow_esta_declarado_en_las_dependencias():
    import tomllib

    deps = tomllib.loads((RAIZ / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "dependencies"
    ]
    assert any(d.startswith("pyarrow==") for d in deps)
    assert "pyarrow==" in (RAIZ / "requirements.txt").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("tarea", "archivo", "id_esperado"),
    [(1, "historia_2022.csv", "cod_ies"), (2, "ies_nuevas.csv", "caso")],
)
def test_los_ejemplos_de_examples_se_predicen_y_conservan_su_identificador(
    tarea, archivo, id_esperado, tmp_path
):
    salida = tmp_path / "p.csv"
    ap = argparse.ArgumentParser()
    pr.configurar(ap)
    args = ap.parse_args(
        ["--tarea", str(tarea), "--entrada", str(EJEMPLOS / archivo), "--salida", str(salida)]
        + ["--modelos-dir", str(MODELOS)]
    )
    assert pr.ejecutar(args) == 0
    p = pd.read_csv(salida)
    assert id_esperado in p.columns and len(p) == 5
    assert p["tasa_predicha"].between(0, 1).all()


def test_conservar_copia_columnas_pedidas_y_rechaza_las_que_no_existen():
    modelo, meta = pr.cargar_modelo("tarea2_mediana", MODELOS)
    df = _fila_2021().assign(nombre_ies="X", caso="c1")
    s = pr.predecir_tabla(modelo, df, meta["nombre"], conservar=("nombre_ies",))
    assert {"caso", "nombre_ies"} <= set(s.columns)
    with pytest.raises(pr.EntradaInvalida, match="--conservar"):
        pr.predecir_tabla(modelo, df, meta["nombre"], conservar=("no_existe",))


def test_los_json_de_models_coinciden_con_los_archivos():
    for clave in pr.CLAVES_PUBLICADAS:
        meta = json.loads((MODELOS / f"{clave}.json").read_text(encoding="utf-8"))
        assert meta["archivo"] == f"{clave}.joblib"
