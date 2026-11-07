"""Opciones de mitigación de sesgo: pesos, panel incompleto, umbral de denominador, métricas por subgrupo y particiones."""
import numpy as np
import pandas as pd
import pytest

from src.modelado.geografia import agregar_geografia, denominador
from src.modelado.mitigacion import (
    ESQUEMAS_PESOS,
    agregar_panel_incompleto,
    filtrar_denominador_minimo,
    metricas_por_subgrupo,
    pesos_desde_config,
    pesos_muestrales,
)
from src.modelado.particion import (
    cargar_dataset,
    columnas_explicativas,
    kfold_desde_config,
    particion_dejando_fuera,
    particion_group_kfold,
    particion_group_kfold_estratificado,
    particion_por_departamento,
    particiones_desde_config,
    variables_modelado,
    variables_modelado_desde_config,
)

MUNICIPIOS = ["11001", "05001", "76001", "08001", "13001", "68001", "54001", "17001"]


def _df(n_ies=40):
    """n IES x 2015-2021; sector y carácter alternados; la IES i tiene 100*(i+1) de denominador."""
    filas = []
    for i in range(n_ies):
        for a in range(2015, 2022):
            filas.append({
                "cod_ies": str(1000 + i), "anio": a, "tasa_desercion": 0.05 + 0.002 * i, "cod_divipola_mpio": MUNICIPIOS[i % len(MUNICIPIOS)],
                "sector": str(1 + i % 2), "caracter": str(1 + (i // 2) % 4), "log_matriculados_spadies_t2": np.log1p(100 * (i + 1)),
            })
    return pd.DataFrame(filas)


# ---- pesos --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("esquema", ESQUEMAS_PESOS)
def test_todos_los_esquemas_de_pesos_tienen_media_uno(esquema):
    w = pesos_muestrales(_df(), esquema)
    assert len(w) == len(_df()) and w.mean() == pytest.approx(1.0) and (w > 0).all()


def test_pesos_ninguno_son_todos_uno():
    assert (pesos_muestrales(_df(), "ninguno") == 1).all()


def test_pesos_por_matriculados_son_proporcionales_al_denominador():
    df = _df()
    w = pesos_muestrales(df, "matriculados")
    assert (w / denominador(df).to_numpy()).std() == pytest.approx(0, abs=1e-9)
    r = pesos_muestrales(df, "raiz_matriculados")
    assert r.max() / r.min() < w.max() / w.min()  # la raíz concentra menos


def test_pesos_inverso_grupo_dan_igual_peso_total_a_cada_grupo():
    df = _df()
    # grupos desbalanceados: se repiten filas del sector 1 caracter 1
    df = pd.concat([df, df[(df.sector == "1") & (df.caracter == "1")]], ignore_index=True)
    w = pesos_muestrales(df, "inverso_grupo")
    por_grupo = pd.Series(w).groupby([df["sector"], df["caracter"]]).sum()
    assert por_grupo.std() == pytest.approx(0, abs=1e-9)


def test_pesos_inverso_ies_dan_igual_peso_total_a_cada_ies():
    df = _df()
    df = df[~((df.cod_ies == "1000") & (df.anio > 2016))]  # una IES con solo 2 años
    w = pesos_muestrales(df, "inverso_ies")
    por_ies = pd.Series(w).groupby(df["cod_ies"].to_numpy()).sum()
    assert por_ies.std() == pytest.approx(0, abs=1e-9)


def test_pesos_esquema_invalido_o_sin_denominador_falla():
    with pytest.raises(ValueError, match="esquema"):
        pesos_muestrales(_df(), "aleatorio")
    con_nulo = _df()
    con_nulo.loc[0, "log_matriculados_spadies_t2"] = np.nan
    with pytest.raises(ValueError, match="denominador"):
        pesos_muestrales(con_nulo, "matriculados")


def test_pesos_desde_config_por_defecto_son_uno():
    assert (pesos_desde_config(_df(), {}) == 1).all()
    assert pesos_desde_config(_df(), {"modelado": {"pesos_muestrales": "inverso_ies"}}).std() == pytest.approx(0, abs=1e-9)  # panel completo: todos 1


# ---- panel incompleto ---------------------------------------------------------------------------------------
def test_panel_incompleto_marca_las_ies_con_menos_anios():
    df = _df(5)
    df = df[~((df.cod_ies == "1001") & (df.anio.isin([2018, 2019])))]
    out = agregar_panel_incompleto(df)
    assert out.loc[out.cod_ies == "1001", "n_anios_ies"].unique().tolist() == [5]
    assert out.loc[out.cod_ies == "1001", "panel_incompleto"].all()
    assert not out.loc[out.cod_ies == "1000", "panel_incompleto"].any()
    assert "panel_incompleto" not in df.columns  # no modifica la entrada


def test_panel_incompleto_respeta_la_ventana():
    df = _df(3)
    df = df[~((df.cod_ies == "1002") & (df.anio < 2017))]
    assert agregar_panel_incompleto(df, (2017, 2021))["panel_incompleto"].sum() == 0  # en 2017-2021 todas están completas
    assert agregar_panel_incompleto(df, (2015, 2021))["panel_incompleto"].sum() == 5


# ---- umbral de denominador ----------------------------------------------------------------------------------
def test_filtrar_denominador_minimo_deja_solo_los_mayores_o_iguales():
    df = _df(10)  # denominadores 100..1000
    out = filtrar_denominador_minimo(df, 500)
    assert denominador(out).min() == pytest.approx(500) and len(out) == 6 * 7 and out.cod_ies.nunique() == 6


def test_filtrar_denominador_por_debajo_del_pipeline_falla_con_mensaje_claro():
    with pytest.raises(ValueError, match="regenérelo"):
        filtrar_denominador_minimo(_df(), 50, minimo_pipeline=100)


# ---- cargar_dataset con las opciones ------------------------------------------------------------------------
def _csv(tmp_path, df):
    df.to_csv(tmp_path / "dataset_ies.csv", index=False)
    return {"unidad_analisis": "ies", "rutas": {"processed": str(tmp_path)}, "min_matriculados_t2": 100}


def test_cargar_dataset_por_defecto_no_agrega_columnas_ni_filtra(tmp_path):
    base = _df(10)
    cfg = _csv(tmp_path, base)
    out = cargar_dataset(cfg)
    assert list(out.columns) == list(base.columns) and len(out) == len(base)


def test_cargar_dataset_con_umbral_y_panel_desde_la_config(tmp_path):
    cfg = _csv(tmp_path, _df(10))
    cfg["modelado"] = {"umbral_denominador": 500, "indicador_panel_incompleto": True}
    out = cargar_dataset(cfg)
    assert denominador(out).min() == pytest.approx(500) and {"n_anios_ies", "panel_incompleto"} <= set(out.columns)
    num, _ = variables_modelado_desde_config(out, cfg)
    assert "panel_incompleto" in num
    sin, _ = variables_modelado(out)
    assert "panel_incompleto" not in sin  # por defecto no entra


# ---- métricas por subgrupo ----------------------------------------------------------------------------------
def test_metricas_por_subgrupo_calcula_mae_rmse_y_sesgo():
    df = _df(8)
    y = df["tasa_desercion"].to_numpy()
    pred = y + np.where(df["sector"] == "1", 0.02, -0.04)  # error +0,02 en oficial, −0,04 en privada
    m = metricas_por_subgrupo(df, y, pred, columnas=["sector"])
    sec = m[m.variable == "sector"].set_index("grupo")
    assert sec.loc["oficial", "mae"] == pytest.approx(0.02) and sec.loc["oficial", "sesgo_medio"] == pytest.approx(0.02)
    assert sec.loc["privada", "mae"] == pytest.approx(0.04) and sec.loc["privada", "sesgo_medio"] == pytest.approx(-0.04)
    assert sec.loc["privada", "rmse"] == pytest.approx(0.04)
    total = m[m.variable == "total"].iloc[0]
    assert total["filas"] == len(df) and total["mae"] == pytest.approx(0.03)


def test_metricas_por_subgrupo_agrega_geografia_y_acepta_pesos():
    df = _df(8)
    y = df["tasa_desercion"].to_numpy()
    pred = y + np.where(df["cod_divipola_mpio"] == "11001", 0.1, 0.0)
    m = metricas_por_subgrupo(df, y, pred, columnas=["ciudad_principal"]).set_index(["variable", "grupo"])
    assert m.loc[("ciudad_principal", "Bogotá"), "mae"] == pytest.approx(0.1) and m.loc[("ciudad_principal", "Cali"), "mae"] == pytest.approx(0)
    ponderado = metricas_por_subgrupo(df, y, pred, columnas=[], peso=np.where(df["cod_divipola_mpio"] == "11001", 0.0, 1.0))
    assert ponderado.iloc[0]["mae"] == pytest.approx(0)  # Bogotá con peso 0


def test_metricas_por_subgrupo_exige_misma_longitud():
    with pytest.raises(ValueError, match="misma longitud"):
        metricas_por_subgrupo(_df(4), np.zeros(3), np.zeros(3))


def test_geografia_asigna_region_ciudad_y_tramo():
    g = agregar_geografia(_df(8))
    assert g.loc[g.cod_divipola_mpio == "11001", "region"].unique().tolist() == ["Bogotá D.C."]
    assert set(g["ciudad_principal"]) == {"Bogotá", "Medellín", "Cali", "Resto del país"}
    assert g.loc[g.cod_ies == "1000", "tramo_tamano"].astype(str).unique().tolist() == ["menos de 200"]


# ---- particiones alternativas -------------------------------------------------------------------------------
def test_estratificado_no_reparte_ies_y_prueba_cada_fila_una_vez():
    df = _df(40)
    pliegues = list(particion_group_kfold_estratificado(df, n_splits=5))
    assert len(pliegues) == 5
    for entrena, prueba in pliegues:
        assert set(df.iloc[entrena].cod_ies).isdisjoint(df.iloc[prueba].cod_ies)
    assert sorted(np.concatenate([p for _, p in pliegues])) == list(range(len(df)))


def test_estratificado_mantiene_la_mezcla_de_sector_en_cada_pliegue():
    df = _df(40)  # 20 IES de cada sector
    for _, prueba in particion_group_kfold_estratificado(df, n_splits=5, estratos=["sector"]):
        ies = df.iloc[prueba].drop_duplicates("cod_ies")
        assert (ies.sector == "1").sum() == 4 and (ies.sector == "2").sum() == 4


def test_estratificado_es_reproducible_con_semilla():
    df = _df(40)
    a = [p.tolist() for _, p in particion_group_kfold_estratificado(df, 5, semilla=3)]
    b = [p.tolist() for _, p in particion_group_kfold_estratificado(df, 5, semilla=3)]
    assert a == b


def test_por_departamento_nunca_reparte_un_departamento():
    df = _df(40)
    for entrena, prueba in particion_por_departamento(df, n_splits=4):
        dep = lambda idx: set(df.iloc[idx].cod_divipola_mpio.str[:2])  # noqa: E731
        assert dep(entrena).isdisjoint(dep(prueba))


def test_por_departamento_falla_si_hay_ies_sin_municipio():
    df = _df(10)
    df.loc[0, "cod_divipola_mpio"] = np.nan
    with pytest.raises(ValueError, match="sin municipio"):
        list(particion_por_departamento(df, n_splits=3))


def test_dejando_fuera_bogota_prueba_solo_en_bogota():
    df = _df(16)
    entrena, prueba = particion_dejando_fuera(df, ("11",))
    assert (df.iloc[prueba].cod_divipola_mpio.str[:2] == "11").all()
    assert not (df.iloc[entrena].cod_divipola_mpio.str[:2] == "11").any()
    assert len(entrena) + len(prueba) == len(df)
    _, tres = particion_dejando_fuera(df, ("11001", "05001", "76001"))
    assert set(df.iloc[tres].cod_divipola_mpio) == {"11001", "05001", "76001"}


def test_dejando_fuera_un_prefijo_inexistente_falla():
    with pytest.raises(ValueError, match="vacío"):
        particion_dejando_fuera(_df(8), ("99",))


def test_kfold_desde_config_por_defecto_es_el_group_kfold_de_siempre():
    df = _df(20)
    base = [p.tolist() for _, p in particion_group_kfold(df, 5)]
    assert [p.tolist() for _, p in kfold_desde_config(df, {})] == base
    assert [p.tolist() for _, p in kfold_desde_config(df, {"particion": {"estratificar_por": []}})] == base
    pliegues, temporal = particiones_desde_config(df, {})
    assert [p.tolist() for _, p in pliegues] == base and len(temporal[0]) > 0


def test_kfold_desde_config_elige_estratificado_o_departamento_y_valida():
    df = _df(40)
    estr = [p.tolist() for _, p in kfold_desde_config(df, {"particion": {"estratificar_por": ["sector"]}})]
    assert estr != [p.tolist() for _, p in particion_group_kfold(df, 5)]
    assert len(list(kfold_desde_config(df, {"particion": {"agrupar_por": "departamento", "n_splits": 3}}))) == 3
    with pytest.raises(ValueError, match="agrupar_por"):
        kfold_desde_config(df, {"particion": {"agrupar_por": "region"}})
    with pytest.raises(ValueError, match="estratificar_por"):
        kfold_desde_config(df, {"particion": {"agrupar_por": "departamento", "estratificar_por": ["sector"]}})


def test_n_anios_ies_nunca_entra_en_las_explicativas_por_defecto():
    """`n_anios_ies` cuenta años posteriores al de cada fila (fuga hacia el futuro): no puede ser explicativa."""
    df = agregar_panel_incompleto(_df(10))
    assert "n_anios_ies" in df.columns
    assert "n_anios_ies" not in columnas_explicativas(df)
    assert "n_anios_ies" not in columnas_explicativas(df, excluir=[])  # ni siquiera sin lista de exclusión
    num, _ = variables_modelado(df)
    assert "n_anios_ies" not in num
    num, _ = variables_modelado(df, incluir_panel_incompleto=True)
    assert "n_anios_ies" not in num and "panel_incompleto" in num


def test_n_anios_ies_tampoco_entra_con_la_opcion_de_panel_desde_la_config(tmp_path):
    cfg = _csv(tmp_path, _df(10))
    cfg["modelado"] = {"indicador_panel_incompleto": True}
    out = cargar_dataset(cfg)
    num, _ = variables_modelado_desde_config(out, cfg)
    assert "n_anios_ies" in out.columns and "n_anios_ies" not in num
