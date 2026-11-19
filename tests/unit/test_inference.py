"""`desercion entrenar` y `desercion predecir`: entrenamiento con datos de juguete, validación de entradas y coherencia con la evaluación."""

import json
import logging
import os

import numpy as np
import pandas as pd
import pytest
import yaml

from desercion_ies import cli
from desercion_ies.evaluation import ajuste as aj
from desercion_ies.evaluation import evaluacion_final as ef
from desercion_ies.evaluation import tarea2 as t2
from desercion_ies.evaluation.experimentos import parametros_experimentos
from desercion_ies.evaluation.protocolo import particiones_repetidas
from desercion_ies.inference import entrenar as en
from desercion_ies.inference import predecir as pr
from desercion_ies.inference.modelos import EntradaInvalida, MedianaCaracterSector
from tests.unit.test_sesgo_sensibilidad import _cfg, _datos


@pytest.fixture(scope="module")
def entrenados(tmp_path_factory):
    """Los tres modelos entrenados con el panel sintético (una sola vez para todo el módulo)."""
    tmp = tmp_path_factory.mktemp("modelos")
    cfg, df = _cfg(tmp), _datos()
    ajuste = pd.read_csv(cfg["modelado"]["sesgo_sensibilidad"]["ajuste"])
    destino = tmp / "models"
    archivos = en.entrenar_modelos(cfg, df, ajuste, destino)
    return {"cfg": cfg, "df": df, "ajuste": ajuste, "destino": destino, "archivos": archivos}


# ---------- entrenar ----------
def test_entrenar_guarda_un_joblib_y_un_json_por_modelo(entrenados):
    d = entrenados["destino"]
    assert set(entrenados["archivos"]) == {"tarea1_ridge", "tarea2_ridge", "tarea2_mediana"}
    for clave in entrenados["archivos"]:
        assert (d / f"{clave}.joblib").exists() and (d / f"{clave}.json").exists()


def test_los_metadatos_traen_lo_pedido(entrenados):
    d = entrenados["destino"]
    meta = json.loads((d / "tarea1_ridge.json").read_text(encoding="utf-8"))
    for campo in (
        "fecha_entrenamiento_utc",
        "commit",
        "dataset",
        "versiones",
        "hiperparametros",
        "columnas",
        "validacion",
        "sha256_modelo",
    ):
        assert campo in meta
    assert {"python", "scikit-learn", "xgboost", "numpy", "pandas"} <= set(meta["versiones"])
    assert (
        meta["columnas"]["obligatorias"] == ["tasa_t1"]
        and "tasa_t2" in meta["columnas"]["opcionales"]
    )
    assert meta["dataset"]["filas"] == len(entrenados["df"]) and meta["hiperparametros"] == {
        "alpha": 5.0
    }
    from desercion_ies.reproducibilidad import sha256_archivo

    assert meta["sha256_modelo"] == sha256_archivo(d / "tarea1_ridge.joblib")
    t2m = json.loads((d / "tarea2_ridge.json").read_text(encoding="utf-8"))
    assert (
        t2m["hiperparametros"] == {"alpha": 2.0, "k": 5}
        and "sector" in t2m["columnas"]["obligatorias"]
    )


def test_la_validacion_publicada_se_toma_de_ajuste_resumen_si_existe(entrenados, tmp_path):
    resumen = pd.DataFrame(
        [
            {"configuracion": "T1 C solo historia | ridge", "variante": "ajustado", "mae_media": 0.0245, "mae_ic_inf": 0.0212, "mae_ic_sup": 0.0278, "n_ajustes": 25,
             "mejora_base_pct": 4.4, "p_wilcoxon_base": 1e-6, "p_t_base": 0.011},
            {"configuracion": "T1 base_ingenua", "variante": "linea_base", "mae_media": 0.0256, "mae_ic_inf": 0.0225, "mae_ic_sup": 0.0288, "n_ajustes": 25,
             "mejora_base_pct": 0.0, "p_wilcoxon_base": np.nan, "p_t_base": np.nan},
        ]
    )  # fmt: skip
    en.entrenar_modelos(
        entrenados["cfg"],
        entrenados["df"],
        entrenados["ajuste"],
        tmp_path / "m",
        resumen=resumen,
        fuente="ajuste_resumen.csv",
    )
    v = json.loads((tmp_path / "m" / "tarea1_ridge.json").read_text(encoding="utf-8"))["validacion"]
    assert (
        v["mae"] == 0.0245
        and v["mae_ic95"] == [0.0212, 0.0278]
        and v["linea_base"] == "T1 base_ingenua"
        and v["mae_linea_base"] == 0.0256
    )


# ---------- coherencia con el pipeline de evaluación ----------
def _pliegue(df):
    entrena, prueba = particiones_repetidas(df, 3, [301])[0][0]
    return entrena, prueba


def test_la_prediccion_coincide_con_la_del_pipeline_de_evaluacion_para_las_mismas_filas(entrenados):
    cfg, df, ajuste = entrenados["cfg"], entrenados["df"], entrenados["ajuste"]
    entrena, prueba = _pliegue(df)
    y = df["tasa_desercion"].astype(float)
    p_exp, p_t2 = parametros_experimentos(cfg), t2.parametros_tarea2(cfg)
    finales = {m["nombre"]: m for m in ef.modelos_finales(cfg, ajuste)}
    for clave, (nombre, _) in en.FINALES.items():
        m = finales[nombre]
        X, num, cat = aj.preparar(m["spec"], cfg, df)
        esperado = ef._tarea(m, 0, 0, 301, 0, entrena, prueba, X, y, num, cat, p_exp, p_t2, "h")[
            3
        ]  # el camino de la evaluación
        modelo = en.entrenar_modelo(
            m, cfg, df.iloc[entrena].reset_index(drop=True)
        )  # el camino de `desercion entrenar`
        # la evaluación también predice las filas sin tasa_t1 (con la mediana); `predecir` las rechaza, así que se comparan las demás
        con_tasa = (
            df.iloc[prueba]["tasa_t1"].notna().to_numpy()
            if m["tarea"] == "T1"
            else np.ones(len(prueba), dtype=bool)
        )
        assert con_tasa.sum() > 10
        obtenido = modelo.predecir(df.iloc[prueba][con_tasa].reset_index(drop=True))
        assert np.array_equal(obtenido, np.asarray(esperado)[con_tasa]), clave


def test_la_mediana_por_caracter_y_sector_coincide_con_la_regla_de_la_evaluacion(entrenados):
    df = entrenados["df"]
    entrena, prueba = _pliegue(df)
    y = df["tasa_desercion"].astype(float)
    esperado, _ = t2.prediccion_mediana_grupo(df, y, entrena, prueba)
    modelo = en.entrenar_mediana(df.iloc[entrena].reset_index(drop=True))
    assert np.array_equal(modelo.predecir(df.iloc[prueba].reset_index(drop=True)), esperado)


def test_un_grupo_desconocido_usa_la_mediana_global():
    df = pd.DataFrame(
        {"sector": ["1", "1", "2"], "caracter": ["a", "a", "b"], "tasa_desercion": [0.1, 0.3, 0.5]}
    )
    m = MedianaCaracterSector().fit(df, df["tasa_desercion"])
    nuevo = pd.DataFrame({"sector": ["1", "9"], "caracter": ["a", "z"]})
    assert m.predict(nuevo).tolist() == [
        0.2,
        0.3,
    ]  # mediana del grupo conocido y mediana global (0,3)


def test_las_predicciones_quedan_entre_0_y_1(entrenados):
    modelo, _ = pr.cargar_modelo("tarea1_ridge", entrenados["destino"])
    extremo = pd.DataFrame({"tasa_t1": [0.0, 1.0, 0.5]})
    p = modelo.predecir(extremo)
    assert ((p >= 0) & (p <= 1)).all()


# ---------- predecir: entradas mal formadas ----------
@pytest.fixture
def t1(entrenados):
    return pr.cargar_modelo("tarea1_ridge", entrenados["destino"])[0]


@pytest.fixture
def t2ridge(entrenados):
    return pr.cargar_modelo("tarea2_ridge", entrenados["destino"])[0]


def test_la_tarea_1_sin_tasa_t1_sugiere_la_tarea_2(t1):
    with pytest.raises(EntradaInvalida, match=r"exige la columna `tasa_t1`.*--tarea 2"):
        t1.predecir(pd.DataFrame({"cod_ies": ["1"], "anio": [2021]}))


def test_la_tarea_1_con_tasa_t1_vacia_sugiere_la_tarea_2_y_dice_las_filas(t1):
    with pytest.raises(EntradaInvalida, match=r"vacía en 2 fila.*filas 4, 5.*--tarea 2"):
        t1.predecir(pd.DataFrame({"tasa_t1": [0.1, 0.2, np.nan, np.nan]}))


def test_la_tarea_1_rechaza_porcentajes_y_valores_no_numericos(t1):
    with pytest.raises(EntradaInvalida, match="entre 0 y 1"):
        t1.predecir(pd.DataFrame({"tasa_t1": [0.1, 12.5]}))
    with pytest.raises(EntradaInvalida, match=r"`tasa_t1` debe ser numérica.*'alta'"):
        t1.predecir(pd.DataFrame({"tasa_t1": [0.1, "alta"]}))


def test_la_tarea_2_lista_las_columnas_que_faltan(t2ridge):
    with pytest.raises(
        EntradaInvalida, match=r"faltan columnas obligatorias para la Tarea 2: .*`x1`"
    ):
        t2ridge.predecir(pd.DataFrame({"sector": ["1"], "caracter": ["1"]}))


def test_la_tarea_2_rechaza_categorias_vacias_y_numeros_malos(entrenados, t2ridge):
    fila = entrenados["df"].iloc[[0]].copy()
    vacia = fila.assign(sector="")
    with pytest.raises(EntradaInvalida, match="`sector` tiene valores vacíos"):
        t2ridge.predecir(vacia)
    mala = fila.assign(x1="no es un número")
    with pytest.raises(EntradaInvalida, match="`x1` debe ser numérica"):
        t2ridge.predecir(mala)


def test_una_entrada_sin_filas_se_rechaza(t1):
    with pytest.raises(EntradaInvalida, match="no tiene filas"):
        t1.predecir(pd.DataFrame({"tasa_t1": []}))


def test_un_codigo_nuevo_de_categoria_avisa_pero_predice(entrenados, t2ridge, caplog):
    caplog.set_level(logging.WARNING)
    fila = entrenados["df"].iloc[[0]].copy().assign(caracter="99")
    assert len(t2ridge.predecir(fila)) == 1
    assert "no vio al entrenar" in caplog.text and "99" in caplog.text


def test_leer_entrada_rechaza_archivos_ausentes_o_vacios(tmp_path):
    with pytest.raises(EntradaInvalida, match="no existe"):
        pr.leer_entrada(tmp_path / "no_esta.csv")
    vacio = tmp_path / "vacio.csv"
    vacio.write_text("", encoding="utf-8")
    with pytest.raises(EntradaInvalida, match="vacío"):
        pr.leer_entrada(vacio)


def test_un_modelo_alterado_o_ausente_no_se_carga(entrenados, tmp_path):
    import shutil

    copia = tmp_path / "copia"
    shutil.copytree(entrenados["destino"], copia)
    with open(copia / "tarea1_ridge.joblib", "ab") as f:
        f.write(b"basura")
    with pytest.raises(EntradaInvalida, match="SHA-256"):
        pr.cargar_modelo("tarea1_ridge", copia)
    with pytest.raises(EntradaInvalida, match="desercion entrenar"):
        pr.cargar_modelo("tarea1_ridge", tmp_path / "vacia")


# ---------- de punta a punta por la CLI ----------
def test_entrenar_y_predecir_por_la_cli_con_datos_de_juguete(entrenados, tmp_path, monkeypatch):
    cfg, df = dict(entrenados["cfg"]), entrenados["df"]
    procesado = tmp_path / "processed"
    procesado.mkdir()
    df.to_csv(procesado / "dataset_ies.csv", index=False)
    cfg.update(unidad_analisis="ies", rutas={"processed": str(procesado)})
    cfg["modelado"] = {
        **cfg["modelado"],
        "ajuste_temporal": {
            **cfg["modelado"]["ajuste_temporal"],
            "salidas": {"ajuste_resumen": str(tmp_path / "no_hay.csv")},
        },
    }
    config = tmp_path / "toy.yaml"
    config.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    modelos = tmp_path / "modelos"
    monkeypatch.chdir(
        tmp_path
    )  # la CLI cambia al directorio raíz del proyecto; monkeypatch lo restaura
    assert cli.main(["entrenar", "--config", str(config), "--salida", str(modelos)]) == 0
    assert (modelos / "tarea2_ridge.joblib").exists()

    entrada = tmp_path / "entrada.csv"
    df[df["anio"] == 2021][
        [
            "cod_ies",
            "anio",
            "tasa_t1",
            "tasa_t2",
            "tasa_media_previa",
            "n_anios_previos",
            "falta_historia",
        ]
    ].to_csv(entrada, index=False)
    salida = tmp_path / "pred.csv"
    args = [
        "predecir",
        "--tarea",
        "1",
        "--entrada",
        str(entrada),
        "--salida",
        str(salida),
        "--modelos-dir",
        str(modelos),
    ]
    assert cli.main(args) == 0
    p = pd.read_csv(salida)
    assert (
        list(p.columns) == ["cod_ies", "anio", "modelo", "tasa_predicha"]
        and len(p) == df["anio"].eq(2021).sum()
    )
    assert p["tasa_predicha"].between(0, 1).all()

    sin_tasa = tmp_path / "sin_tasa.csv"
    pd.DataFrame({"cod_ies": ["1"]}).to_csv(sin_tasa, index=False)
    args[args.index(str(entrada))] = str(sin_tasa)
    assert cli.main(args) == 2  # entrada no válida: código 2, sin traceback
    cli.configurar_logging(0, False)
    os.chdir(tmp_path)
