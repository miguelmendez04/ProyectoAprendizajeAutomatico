"""Modelos guardados en `models/`: coinciden con el pipeline de evaluación sobre el dataset real y sus metadatos con los informes."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from desercion_ies.config import load_config
from desercion_ies.evaluation import ajuste as aj
from desercion_ies.evaluation import evaluacion_final as ef
from desercion_ies.evaluation import tarea2 as t2
from desercion_ies.evaluation.experimentos import parametros_experimentos
from desercion_ies.features.particion import cargar_dataset
from desercion_ies.inference import entrenar as en
from desercion_ies.inference import predecir as pr
from desercion_ies.reproducibilidad import leer_checksums, sha256_archivo

RAIZ = Path(__file__).resolve().parents[2]
MODELOS = RAIZ / "models"

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def real():
    if (
        not (MODELOS / "tarea1_ridge.joblib").exists()
        or not (RAIZ / "data/processed/dataset_ies.csv").exists()
    ):
        pytest.skip("faltan models/ o data/processed")
    cfg = load_config(RAIZ / "configs/config.yaml")
    return cfg, cargar_dataset(cfg)


@pytest.mark.parametrize("clave", en.CLAVES)
def test_los_metadatos_de_los_modelos_publicados_son_consistentes(clave, real):
    if not (MODELOS / f"{clave}.json").exists():
        pytest.skip("faltan models/")
    meta = json.loads((MODELOS / f"{clave}.json").read_text(encoding="utf-8"))
    assert meta["sha256_modelo"] == sha256_archivo(MODELOS / f"{clave}.joblib")
    assert (
        meta["dataset"]["sha256"]
        == leer_checksums(RAIZ / "data/checksums.json")["procesado"][
            "data/processed/dataset_ies.csv"
        ]["sha256"]
    )
    assert meta["dataset"]["filas"] == len(real[1]) and meta["validacion"]["mae"] > 0
    assert len(meta["commit"]) == 40 and {"python", "scikit-learn", "xgboost", "numpy"} <= set(
        meta["versiones"]
    )


def test_el_mae_de_validacion_guardado_es_el_de_los_informes(real):
    resumen = pd.read_csv(RAIZ / "reports/tables/ajuste_resumen.csv").set_index(
        ["configuracion", "variante"]
    )
    t1 = json.loads((MODELOS / "tarea1_ridge.json").read_text(encoding="utf-8"))["validacion"]
    assert t1["mae"] == resumen.loc[("T1 C solo historia | ridge", "ajustado"), "mae_media"]
    base = json.loads((MODELOS / "tarea2_mediana.json").read_text(encoding="utf-8"))["validacion"]
    assert base["mae"] == resumen.loc[("T2 base_mediana_grupo", "linea_base"), "mae_media"]


def test_la_prediccion_sobre_el_dataset_coincide_con_la_del_pipeline_de_evaluacion(real):
    cfg, df = real
    p_exp, p_t2 = parametros_experimentos(cfg), t2.parametros_tarea2(cfg)
    y = df["tasa_desercion"].astype(float)
    finales = {m["nombre"]: m for m in ef.modelos_finales(cfg)}
    todas = np.arange(len(df))
    for clave, (nombre, _) in en.FINALES.items():
        m = finales[nombre]
        modelo, _ = pr.cargar_modelo(clave, MODELOS)
        X, num, cat = aj.preparar(m["spec"], cfg, df)
        # el camino de la evaluación: ajustar con todas las filas y predecir las mismas filas
        esperado = ef._tarea(m, 0, 0, 0, 0, todas, todas, X, y, num, cat, p_exp, p_t2, "h")[3]
        filas = (
            df["tasa_t1"].notna().to_numpy() if m["tarea"] == "T1" else np.ones(len(df), dtype=bool)
        )
        obtenido = modelo.predecir(df[filas].reset_index(drop=True))
        assert np.allclose(obtenido, np.asarray(esperado)[filas], rtol=0, atol=1e-12), clave
    filas_mediana, _ = t2.prediccion_mediana_grupo(df, y, todas, todas)
    mediana, _ = pr.cargar_modelo("tarea2_mediana", MODELOS)
    assert np.allclose(mediana.predecir(df), filas_mediana, rtol=0, atol=1e-12)
