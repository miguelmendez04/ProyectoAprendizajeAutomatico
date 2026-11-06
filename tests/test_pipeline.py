"""Prueba end-to-end del pipeline con archivos crudos mínimos (formato real de datos.gov.co)."""
import pandas as pd
import pytest
import yaml

import pipeline
from src.validate.schema import validar

NIVELES = ["tecnica_profesional", "tecnologica", "universitaria", "especializacion", "maestria", "doctorado"]


def _escribir_crudos(raw):
    """Crea data/raw mínimo: 2 municipios (Medellín 05001, Cali 76001), 2 IES, años 2016-2018."""
    (raw / "dane").mkdir(parents=True)
    (raw / "snies").mkdir()
    (raw / "saber11").mkdir()
    (raw / "spadies").mkdir()

    pd.DataFrame(
        {"cod_dpto": ["05", "76"], "dpto": ["ANTIOQUIA", "VALLE DEL CAUCA"], "cod_mpio": ["05001", "76001"],
         "nom_mpio": ["MEDELLÍN", "CALI"], "tipo_municipio": ["Municipio"] * 2,
         "longitud": ["-75,5", "-76,5"], "latitud": ["6,2", "3,4"]}
    ).to_csv(raw / "dane" / "divipola.csv", index=False)

    # Saber 11 (periodo 2017 -> predice 2018); códigos sin cero a la izquierda
    pd.DataFrame(
        {"estu_consecutivo": ["a", "b", "c", "d"], "periodo": ["20172"] * 4, "cole_cod_mcpio_ubicacion": ["5001", "5001", "76001", "0"],
         "punt_global": [250, 270, 300, 200], "fami_estratovivienda": ["Estrato 1", "Estrato 4", "Estrato 2", "Sin Estrato"],
         "cole_naturaleza": ["OFICIAL", "NO OFICIAL", "OFICIAL", "OFICIAL"]}
    ).to_csv(raw / "saber11" / "saber11.csv", index=False)

    filas = []
    for anio in ("2016", "2017", "2018"):
        for cod, nombre in (("5001", "MEDELLÍN"), ("76001", "CALI")):
            filas.append({"a_o": anio, "c_digo_delmunicipio": cod, "nombre_del_municipio": nombre,
                          **{n: "10" for n in NIVELES}, "ies_con_oferta": "5"})
    pd.DataFrame(filas).to_csv(raw / "snies" / "matricula_mpio.csv", index=False)

    prog = []
    for anio in ("2016", "2017", "2018"):
        for ies, programa, mpio in (("1101", "10", "5001"), ("2833", "20", "76001")):
            prog.append({"c_digo_de_la_instituci_n": ies, "c_digo_snies_delprograma": programa,
                         "c_digo_del_municipio_ies": mpio, "c_digo_del_municipio_programa": mpio,
                         "id_sector": "1", "id_caracter": "4", "a_o": anio, "semestre": "1",
                         "id_g_nero": "1", "matriculados_2015": "1500"})
    pd.DataFrame(prog).to_csv(raw / "snies" / "matricula_programa.csv", index=False)


def _config(tmp_path, unidad, periodicidad="anual"):
    raw = tmp_path / "data" / "raw"
    cfg = {
        "unidad_analisis": unidad,
        "anios": [2018],
        "rutas": {"raw": str(raw), "interim": str(tmp_path / "data" / "interim"),
                  "processed": str(tmp_path / "data" / "processed"), "reports": str(tmp_path / "reports")},
        "datasets": {},
        "terridata": {"directorio": str(raw / "terridata")},
        "spadies": {"archivo": str(raw / "spadies" / "spadies_diferenciado.csv"),
                    "nacional": str(raw / "spadies" / "spadies_nacional.csv"), "periodicidad": periodicidad},
        "max_nulos_pct": 0.5,
    }
    ruta = tmp_path / "config.yaml"
    ruta.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return ruta, raw


def _spadies(raw):
    """Export diferenciado 2016-1 a 2018-2; la tasa anual de 2018 es 0,105 (IES 1101) y 0,2 (IES 2833); matrícula de referencia 2000 frente a 1500 de SNIES (ratio 1,3)."""
    periodos = ["2016-1", "2016-2", "2017-1", "2017-2", "2018-1", "2018-2"]
    filas = ["IES;TIPO;" + ";".join(periodos)]
    for ies, desertores in (("1101", 210), ("2833", 400)):
        filas.append(f"{ies};DESERTORES;" + ";".join([str(desertores)] * 6))
        filas.append(f"{ies};MATRICULADOS;" + ";".join(["2000"] * 6))
    (raw / "spadies" / "spadies_diferenciado.csv").write_text("\n".join(filas) + "\n", encoding="utf-8-sig")


@pytest.mark.parametrize("unidad", ["municipio", "ies"])
def test_pipeline_end_to_end_con_fixtures(tmp_path, unidad):
    ruta, raw = _config(tmp_path, unidad)
    _escribir_crudos(raw)
    _spadies(raw)

    salida = pipeline.main(str(ruta), saltar_extract=True)

    df = pd.read_csv(salida, dtype={"cod_divipola_mpio": str, "cod_ies": str})
    assert salida.name == f"dataset_{unidad}.csv" and len(df) == 2
    validar(df, unidad, max_nulos_pct=0.5)
    assert df["tasa_desercion"].round(3).tolist() in ([0.105, 0.2], [0.2, 0.105])
    assert df["punt_global_media_t1"].notna().all()  # Saber 2017 llegó como variable t-1
    assert df["tasa_desercion"].notna().all() and df["mpio_matriculados_t2"].notna().all()  # matrícula 2016 = t-2
    log = (tmp_path / "reports" / "etl_log.md").read_text(encoding="utf-8")
    assert "| saber11 |" in log and "filas_sin_match" in log
    assert "Saber 11: 1 fila(s) sin código de municipio válido" in log  # el código "0" se cuenta, no se pierde en silencio
    assert (tmp_path / "reports" / "data_dictionary.md").exists()


def test_pipeline_sin_spadies_genera_dataset_preliminar(tmp_path):
    ruta, raw = _config(tmp_path, "municipio")
    _escribir_crudos(raw)  # sin export de SPADIES

    salida = pipeline.main(str(ruta), saltar_extract=True)

    df = pd.read_csv(salida, dtype={"cod_divipola_mpio": str})
    assert df["tasa_desercion"].isna().all() and len(df) == 2
    assert "SPADIES" in (tmp_path / "reports" / "etl_log.md").read_text(encoding="utf-8")


def test_codigos_fuera_de_divipola_se_cuentan():
    from src.integrate.build import codigos_fuera_de_divipola

    divipola = pd.DataFrame({"cod_divipola_mpio": ["05001"]})
    df = pd.DataFrame({"cod_divipola_mpio": ["05001", "99999x", "76001", None]})
    assert codigos_fuera_de_divipola(df, divipola) == 2  # los nulos no cuentan


def test_bitacora_conserva_notas_manuales_entre_corridas(tmp_path):
    ruta, raw = _config(tmp_path, "municipio")
    _escribir_crudos(raw)
    bitacora = tmp_path / "reports" / "etl_log.md"
    bitacora.parent.mkdir(parents=True)
    bitacora.write_text("# Bitácora\n\nDecisión manual: ventana 2015-2021.\n", encoding="utf-8")

    pipeline.main(str(ruta), saltar_extract=True)
    pipeline.main(str(ruta), saltar_extract=True)

    texto = bitacora.read_text(encoding="utf-8")
    assert texto.count("Decisión manual: ventana 2015-2021.") == 1
    assert texto.count("| saber11 |") == 1  # el bloque automático se reemplaza, no se acumula


def test_cambios_de_codigo_se_aplican_a_todas_las_fuentes(tmp_path):
    ruta, raw = _config(tmp_path, "municipio")
    cfg = yaml.safe_load(ruta.read_text(encoding="utf-8"))
    cfg["cambios_codigo"] = {"76001": "76002"}  # municipio con código cambiado
    ruta.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    _escribir_crudos(raw)
    _spadies(raw)

    df = pd.read_csv(pipeline.main(str(ruta), saltar_extract=True), dtype={"cod_divipola_mpio": str})

    assert set(df["cod_divipola_mpio"]) == {"05001", "76002"}
    assert df["punt_global_media_t1"].notna().all()  # Saber 11 y SNIES siguieron al municipio renombrado


def test_unidad_ies_semestral_agrega_semestre_a_la_llave(tmp_path):
    ruta, raw = _config(tmp_path, "ies", periodicidad="semestre")
    _escribir_crudos(raw)
    _spadies(raw)

    df = pd.read_csv(pipeline.main(str(ruta), saltar_extract=True), dtype={"cod_ies": str})

    assert sorted(df["semestre"].unique()) == [1, 2] and len(df) == 4  # 2 IES x 2 semestres de 2018
    assert df["punt_global_media_t1"].notna().all()  # el contexto territorial anual se une por el municipio sede


def test_unidad_programa_sin_spadies_por_programa_error_claro(tmp_path):
    ruta, raw = _config(tmp_path, "programa")
    _escribir_crudos(raw)
    _spadies(raw)
    with pytest.raises(ValueError, match="cod_snies_programa"):
        pipeline.main(str(ruta), saltar_extract=True)


def test_ies_con_varias_sedes_se_asigna_a_la_de_mas_programas_y_se_senala(tmp_path):
    ruta, raw = _config(tmp_path, "ies")
    _escribir_crudos(raw)
    _spadies(raw)
    prog = pd.read_csv(raw / "snies" / "matricula_programa.csv", dtype=str)
    extra = prog[prog["c_digo_de_la_instituci_n"] == "1101"].copy()
    extra["c_digo_snies_delprograma"] = "11"
    extra["c_digo_del_municipio_ies"] = "76001"  # un programa más, en Cali
    extra["c_digo_del_municipio_programa"] = "76001"
    pd.concat([prog, extra, extra.assign(c_digo_snies_delprograma="12")]).to_csv(raw / "snies" / "matricula_programa.csv", index=False)

    df = pd.read_csv(pipeline.main(str(ruta), saltar_extract=True), dtype={"cod_ies": str, "cod_divipola_mpio": str})

    assert df.set_index("cod_ies").loc["1101", "cod_divipola_mpio"] == "76001"  # 2 programas en Cali contra 1 en Medellín
    log = (tmp_path / "reports" / "etl_log.md").read_text(encoding="utf-8")
    assert "IES con programas en más de un municipio (1: 1101)" in log


def _terridata(raw):
    """TerriData solo para Cali (76001): cobertura superior 2017 y población 2018; Medellín no aparece."""
    from tests.test_terridata import _col, _fila, _zip

    (raw / "terridata").mkdir()
    cob, hom, muj = _col("cob_superior"), _col("poblacion_hombres"), _col("poblacion_mujeres")
    _zip(raw / "terridata", 4, [_fila("76001", "Cali", cob["subcategoria"], cob["indicador"], "56,70", 2017)])
    _zip(raw / "terridata", 2, [
        _fila("76001", "Cali", hom["subcategoria"], hom["indicador"], "1.000,00", 2018),
        _fila("76001", "Cali", muj["subcategoria"], muj["indicador"], "1.100,00", 2018),
    ])


def test_pipeline_integra_terridata_con_rezago_t1_y_reporta_ies_sin_match(tmp_path):
    ruta, raw = _config(tmp_path, "ies")
    _escribir_crudos(raw)
    _spadies(raw)
    _terridata(raw)

    df = pd.read_csv(pipeline.main(str(ruta), saltar_extract=True), dtype={"cod_ies": str}).set_index("cod_ies")

    assert df.loc["2833", "td_cob_superior_t1"] == pytest.approx(56.7)  # dato de 2017 para el objetivo 2018
    assert df.loc["2833", "td_poblacion_total_t1"] == 2100  # 2018 completado hacia atrás para 2017
    assert pd.isna(df.loc["1101", "td_cob_superior_t1"])  # Medellín no está en TerriData
    log = (tmp_path / "reports" / "etl_log.md").read_text(encoding="utf-8")
    assert "TerriData: 1 de 2 IES sin municipio sede en TerriData (1101→05001)" in log
    assert "| terridata |" in log


def test_variables_terridata_con_demasiados_nulos_se_descartan_y_se_anotan(tmp_path):
    from src.integrate.build import depurar_terridata

    df = pd.DataFrame({"cod_ies": ["1", "2", "3"], "cod_divipola_mpio": ["05001", "76001", "76001"],
                       "td_a_t1": [1.0, None, None], "td_b_t1": [1.0, 2.0, 3.0]})
    td = pd.DataFrame({"cod_divipola_mpio": ["76001"], "anio": [2017], "td_a": [1.0]})
    notas: list[str] = []
    out = depurar_terridata(df, td, "ies", max_nulos_pct=0.5, notas=notas)
    assert "td_a_t1" not in out.columns and "td_b_t1" in out.columns
    assert any("td_a_t1 (67%)" in n for n in notas)
    assert any("1 de 3 IES sin municipio sede" in n for n in notas)
