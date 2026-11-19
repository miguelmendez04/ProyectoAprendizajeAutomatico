"""Protocolo v3: conjunto fijado, pliegues repetidos idénticos para todo, ingenuas, rezagada sin t-1, IC y pruebas pareadas."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy import stats

import desercion_ies.evaluation.protocolo as pr
from desercion_ies.etl.integrate.historia import agregar_historia
from desercion_ies.evaluation.formulaciones import variables_formulacion
from desercion_ies.evaluation.protocolo import (
    BASE_INGENUA,
    BASE_INGENUA_REZAGADA,
    BASE_MEDIANA,
    comparar_pareado,
    ejecutar_protocolo,
    media_e_intervalo,
    parametros_protocolo,
    particiones_repetidas,
    prediccion_base,
    resumen,
)

# Conjunto FIJADO ANTES de correr el protocolo v3 (si se cambia, que sea una decisión explícita y documentada, no un ajuste a resultados)
CONJUNTO_FIJADO = [
    ("A_contexto", "ridge", "logit", "info_mutua"),
    ("B_historia_contexto", "xgboost", "logit", "ninguna"),
    ("B_historia_contexto", "ridge", "logit", "ninguna"),
    ("B_solo_historia", "ridge", "logit", "ninguna"),
    ("C_correccion", "ridge", "ninguna", "ninguna"),
    ("C_correccion", "random_forest", "ninguna", "ninguna"),
    ("C_correccion_solo_historia", "ridge", "ninguna", "ninguna"),
    ("C_correccion_solo_historia", "random_forest", "ninguna", "ninguna"),
    ("B_historia_rezagada_contexto", "xgboost", "logit", "ninguna"),
    ("B_historia_rezagada_contexto", "ridge", "logit", "ninguna"),
    ("B_solo_historia_rezagada", "ridge", "logit", "ninguna"),
    ("C_correccion_rezagada", "ridge", "ninguna", "ninguna"),
    ("C_correccion_rezagada", "random_forest", "ninguna", "ninguna"),
    ("C_correccion_solo_historia_rezagada", "ridge", "ninguna", "ninguna"),
    ("C_correccion_solo_historia_rezagada", "random_forest", "ninguna", "ninguna"),
]


def _datos(n_ies: int = 40, semilla: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    previa, filas = [], []
    for i in range(n_ies):
        base = rng.uniform(0.05, 0.25)
        for a in (2015, 2016):
            previa.append(
                {
                    "cod_ies": str(1000 + i),
                    "anio": a,
                    "tasa_desercion": float(np.clip(base + rng.normal(0, 0.02), 0.01, 0.9)),
                }
            )
        for a in range(2017, 2021):
            if rng.random() < 0.08:
                continue
            filas.append(
                {
                    "cod_ies": str(1000 + i),
                    "anio": a,
                    "sector": str(1 + i % 2),
                    "caracter": str(1 + i % 4),
                    "tasa_desercion": float(np.clip(base + rng.normal(0, 0.02), 0.01, 0.9)),
                    "x1": rng.normal(),
                    "x2": rng.normal(),
                    "falta_saber11": a == 2020,
                }
            )
    df = pd.DataFrame(filas)
    # algunas IES nuevas sin previa: sus primeras filas quedan sin historia
    previa = pd.DataFrame(previa)
    previa = previa[~previa["cod_ies"].isin([str(1000 + i) for i in range(0, 4)])]
    return agregar_historia(df, ["cod_ies"], previa=previa)


def _cfg(configuraciones=None, semillas=(1, 2)) -> dict:
    configuraciones = configuraciones or [
        {
            "nombre": "A",
            "formulacion": "A_contexto",
            "modelo": "ridge",
            "objetivo": "ninguna",
            "seleccion": "ninguna",
        },
        {
            "nombre": "B",
            "formulacion": "B_historia_contexto",
            "modelo": "ridge",
            "objetivo": "logit",
            "seleccion": "ninguna",
        },
        {
            "nombre": "B solo historia",
            "formulacion": "B_solo_historia",
            "modelo": "ridge",
            "objetivo": "ninguna",
            "seleccion": "info_mutua",
        },
        {
            "nombre": "C",
            "formulacion": "C_correccion",
            "modelo": "ridge",
            "objetivo": "ninguna",
            "seleccion": "ninguna",
        },
        {
            "nombre": "B rez",
            "formulacion": "B_historia_rezagada_contexto",
            "modelo": "ridge",
            "objetivo": "ninguna",
            "seleccion": "ninguna",
        },
        {
            "nombre": "C rez",
            "formulacion": "C_correccion_rezagada",
            "modelo": "ridge",
            "objetivo": "ninguna",
            "seleccion": "ninguna",
        },
    ]
    return {
        "modelado": {
            "excluir_variables": [],
            "experimentos": {"semilla": 3, "k_info_mutua": 15},
            "protocolo_v3": {
                "n_splits": 3,
                "semillas": list(semillas),
                "configuraciones": configuraciones,
            },
        }
    }


# ---------- el conjunto está fijado ----------
def test_el_conjunto_de_configuraciones_esta_fijado():
    cfg = yaml.safe_load(Path("configs/config.yaml").read_text(encoding="utf-8"))
    p = parametros_protocolo(cfg)
    assert [
        (c["formulacion"], c["modelo"], c["objetivo"], c["seleccion"]) for c in p["configuraciones"]
    ] == CONJUNTO_FIJADO
    assert (
        len(p["semillas"]) == 5 and p["n_splits"] == 5 and p["nivel_confianza"] == 0.95
    )  # 25 ajustes por configuración


def test_nombres_duplicados_o_de_lineas_base_fallan():
    c = _cfg()["modelado"]["protocolo_v3"]["configuraciones"]
    with pytest.raises(ValueError, match="únicos"):
        parametros_protocolo(_cfg(c + [dict(c[0])]))
    with pytest.raises(ValueError, match="únicos"):
        parametros_protocolo(_cfg([{**c[0], "nombre": BASE_INGENUA}]))


# ---------- pliegues repetidos ----------
def test_las_repeticiones_son_reproducibles_distintas_y_agrupadas_por_ies():
    df = _datos()
    a, b = particiones_repetidas(df, 3, [1, 2]), particiones_repetidas(df, 3, [1, 2])
    for rep_a, rep_b in zip(a, b, strict=False):
        for (_ta, pa), (_tb, pb) in zip(rep_a, rep_b, strict=False):
            assert np.array_equal(pa, pb)  # misma semilla, mismos pliegues
    assert not all(
        np.array_equal(p1, p2) for (_, p1), (_, p2) in zip(a[0], a[1], strict=False)
    )  # semillas distintas, pliegues distintos
    for rep in a:
        for entrena, prueba in rep:
            assert set(df.iloc[entrena]["cod_ies"]).isdisjoint(df.iloc[prueba]["cod_ies"])
        assert sorted(np.concatenate([p for _, p in rep])) == list(
            range(len(df))
        )  # cada fila se evalúa una vez por repetición


def test_todas_las_configuraciones_y_lineas_base_usan_los_mismos_pliegues_en_cada_repeticion(
    monkeypatch,
):
    llamadas = []
    original = pr.particiones_repetidas

    def espia(*a, **k):
        llamadas.append(1)
        return original(*a, **k)

    monkeypatch.setattr(pr, "particiones_repetidas", espia)
    res = ejecutar_protocolo(_cfg(), _datos())
    assert len(llamadas) == 1  # se generan una sola vez
    assert (res.groupby(["repeticion", "pliegue"])["hash_prueba"].nunique() == 1).all()
    assert (res.groupby(["repeticion", "pliegue"])["n_prueba"].nunique() == 1).all()
    assert (
        res.groupby("configuracion").size().eq(2 * 3).all()
    )  # 2 repeticiones x 3 pliegues por configuración y línea base
    assert set(res["configuracion"]) >= {
        BASE_INGENUA,
        BASE_INGENUA_REZAGADA,
        BASE_MEDIANA,
        "A",
        "C rez",
    }


# ---------- líneas base ----------
def test_la_ingenua_usa_tasa_t1_y_la_rezagada_tasa_t2_con_mediana_de_respaldo():
    df = _datos()
    y = df["tasa_desercion"]
    entrena, prueba = np.arange(0, 100), np.arange(100, len(df))
    mediana = np.median(y.iloc[entrena])
    for columna in ("tasa_t1", "tasa_t2"):
        pred = prediccion_base(df, y, entrena, prueba, columna)
        v = df[columna].to_numpy()[prueba]
        assert pred == pytest.approx(np.where(np.isnan(v), mediana, v))
    # la historia previa a la ventana da t-1 al primer año: el primer año de una IES con previa NO cae a la mediana
    primero = df[(df["anio"] == 2017) & df["tasa_t1"].notna()]
    assert len(primero) > 0


# ---------- historia rezagada: no usa t-1 ----------
def test_las_formulaciones_rezagadas_no_llevan_columnas_con_t_menos_1():
    df, cfg = _datos(), _cfg()
    prohibidas = {"tasa_t1", "tasa_media_previa", "n_anios_previos", "falta_historia"}
    for conjunto in ("historia_rezagada_contexto", "solo_historia_rezagada"):
        num, cat = variables_formulacion(df, cfg, conjunto)
        assert not prohibidas & set(num + cat)
        assert "tasa_t2" in num and "tasa_media_rezagada" in num


def test_el_resultado_rezagado_no_cambia_si_se_alteran_las_columnas_de_t_menos_1():
    df, cfg = (
        _datos(),
        _cfg(
            [
                c
                for c in _cfg()["modelado"]["protocolo_v3"]["configuraciones"]
                if "rez" in c["nombre"]
            ]
        ),
    )
    alterado = df.copy()
    rng = np.random.default_rng(9)
    for c in ("tasa_t1", "tasa_media_previa"):
        alterado[c] = rng.uniform(0, 1, len(df))
    a, b = ejecutar_protocolo(cfg, df), ejecutar_protocolo(cfg, alterado)
    for nombre in ("B rez", "C rez"):
        ma = (
            a[a["configuracion"] == nombre].sort_values(["repeticion", "pliegue"])["mae"].to_numpy()
        )
        mb = (
            b[b["configuracion"] == nombre].sort_values(["repeticion", "pliegue"])["mae"].to_numpy()
        )
        assert ma == pytest.approx(mb)  # ni como variable ni como base de la corrección
    # contraprueba: la ingenua normal sí cambia (usa tasa_t1)
    ia = a[a["configuracion"] == BASE_INGENUA]["mae"].to_numpy()
    ib = b[b["configuracion"] == BASE_INGENUA]["mae"].to_numpy()
    assert not np.allclose(ia, ib)


# ---------- intervalo de confianza y pruebas ----------
def test_el_ic_usa_la_varianza_corregida_de_nadeau_y_bengio():
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    media, lo, hi = media_e_intervalo(v, rho=0.25, nivel=0.95)
    h = stats.t.ppf(0.975, 4) * np.sqrt((1 / 5 + 0.25) * 2.5)
    assert media == 3.0 and (lo, hi) == pytest.approx((3.0 - h, 3.0 + h))
    # sin corrección (rho = 0) es el IC t clásico, más estrecho
    _, lo0, hi0 = media_e_intervalo(v, rho=0.0)
    assert hi0 - lo0 < hi - lo
    assert hi0 - lo0 == pytest.approx(2 * stats.t.ppf(0.975, 4) * np.sqrt(2.5 / 5))


def test_el_ic_ignora_los_nan():
    media, lo, hi = media_e_intervalo([1.0, np.nan, 3.0, 5.0], rho=0.25)
    assert media == 3.0 and lo < 3.0 < hi


def test_la_prueba_pareada_tiene_el_signo_la_proporcion_y_los_p_correctos():
    dif = np.linspace(0.001, 0.02, 25)  # 25 diferencias positivas: gana la configuración en todos
    c = comparar_pareado(dif, rho=0.25)
    assert (
        c["n_pares"] == 25 and c["prop_gana"] == 1.0 and c["dif_media"] == pytest.approx(dif.mean())
    )
    assert c["p_wilcoxon"] == pytest.approx(stats.wilcoxon(dif).pvalue) and c["p_wilcoxon"] < 1e-6
    t = dif.mean() / np.sqrt((1 / 25 + 0.25) * dif.var(ddof=1))
    assert c["p_t_corregido"] == pytest.approx(2 * stats.t.sf(abs(t), 24))
    # la corrección hace la t más conservadora que la t clásica
    assert c["p_t_corregido"] > 2 * stats.t.sf(abs(dif.mean() / np.sqrt(dif.var(ddof=1) / 25)), 24)
    neg = comparar_pareado(-dif, rho=0.25)
    assert neg["prop_gana"] == 0.0 and neg["dif_media"] < 0


def test_la_prueba_pareada_sin_diferencias_no_falla():
    c = comparar_pareado(np.zeros(25), rho=0.25)
    assert np.isnan(c["p_wilcoxon"]) and np.isnan(c["p_t_corregido"]) and c["prop_gana"] == 0.0


def test_el_resumen_compara_cada_configuracion_con_la_ingenua():
    res = ejecutar_protocolo(_cfg(), _datos())
    r = resumen(res).set_index("configuracion")
    assert r.loc[BASE_INGENUA, "mejora_ing_pct"] == pytest.approx(0)
    mae = res.pivot(index=["repeticion", "pliegue"], columns="configuracion", values="mae")
    dif = (mae[BASE_INGENUA] - mae["A"]).mean()
    assert r.loc["A", "dif_ing"] == pytest.approx(dif)
    assert r.loc["A", "mejora_ing_pct"] == pytest.approx(100 * dif / mae[BASE_INGENUA].mean())
    assert r.loc["A", "n_ajustes"] == 6 and r.loc["A", "gana_ing"] == pytest.approx(
        (mae[BASE_INGENUA] > mae["A"]).mean()
    )


def test_el_informe_y_la_figura_se_generan(tmp_path):
    from desercion_ies.reporting.informe_v3 import escribir_informe, figura_ic

    cfg = _cfg(
        [
            {
                "nombre": n,
                "formulacion": f,
                "modelo": "ridge",
                "objetivo": "ninguna",
                "seleccion": "ninguna",
            }
            for n, f in (
                ("A contexto | ridge logit info_mutua", "A_contexto"),
                ("B historia+contexto | ridge logit", "B_historia_contexto"),
                ("B solo historia | ridge logit", "B_solo_historia"),
                ("C corrección con contexto | ridge", "C_correccion"),
                ("C corrección solo historia | ridge", "C_correccion_solo_historia"),
                ("B historia+contexto | ridge logit [rezagada]", "B_historia_rezagada_contexto"),
                ("B solo historia | ridge logit [rezagada]", "B_solo_historia_rezagada"),
                ("C corrección con contexto | ridge [rezagada]", "C_correccion_rezagada"),
                (
                    "C corrección solo historia | ridge [rezagada]",
                    "C_correccion_solo_historia_rezagada",
                ),
            )
        ]
    )
    res = ejecutar_protocolo(cfg, _datos())
    r = resumen(res)
    meta = {
        "filas": 100,
        "ies": 40,
        "ventana": [2017, 2020],
        "n_splits": 3,
        "semillas": [1, 2],
        "n_ajustes": 6,
        "n_configuraciones": 9,
        "sklearn": "x",
        "xgboost": "y",
        "tiempo_total_s": 1.0,
    }
    # los pares de comparación del informe necesitan los nombres completos de la configuración real; aquí se usan los mismos
    texto = escribir_informe(
        res.assign(configuracion=res["configuracion"]), r, meta, tmp_path / "r.md"
    )
    assert "Tabla principal" in texto and "¿Se sostienen las conclusiones de la v2?" in texto
    figura_ic(r, tmp_path / "f.png")
    assert (tmp_path / "f.png").stat().st_size > 1000
