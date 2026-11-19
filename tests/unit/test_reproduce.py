"""Punto de entrada `desercion reproducir`, verificación de checksums, configuración del modo rápido y comparación con lo publicado."""

import hashlib
import json
import logging
from pathlib import Path

import pandas as pd
import pytest
import yaml

from desercion_ies import reproduce as rep
from desercion_ies import reproducibilidad as rp
from desercion_ies.config import load_config


@pytest.fixture(autouse=True)
def _mensajes_info(caplog):
    caplog.set_level(logging.INFO)  # la salida de reproducir va por logging


# ---------- checksums ----------
def _manifiesto(tmp_path, contenidos: dict[str, bytes], grupo="procesado") -> Path:
    for rel, c in contenidos.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(c)
    man = {
        grupo: {
            rel: {"sha256": hashlib.sha256(c).hexdigest(), "bytes": len(c)}
            for rel, c in contenidos.items()
        }
    }
    ruta = tmp_path / "checksums.json"
    ruta.write_text(json.dumps(man), encoding="utf-8")
    return ruta


def test_sha256_coincide_con_hashlib_y_lee_por_bloques(tmp_path):
    datos = b"abc" * 100_000
    (tmp_path / "x.bin").write_bytes(datos)
    assert rp.sha256_archivo(tmp_path / "x.bin", bloque=1000) == hashlib.sha256(datos).hexdigest()


def test_la_verificacion_distingue_ok_distinto_y_ausente(tmp_path):
    ruta = _manifiesto(
        tmp_path, {"data/a.csv": b"hola", "data/b.csv": b"mundo", "data/c.csv": b"!"}
    )
    (tmp_path / "data/b.csv").write_bytes(b"mundo alterado")
    (tmp_path / "data/c.csv").unlink()
    estados = {
        f["archivo"]: f["estado"] for f in rp.verificar_checksums("procesado", tmp_path, ruta)
    }
    assert estados == {"data/a.csv": "ok", "data/b.csv": "distinto", "data/c.csv": "ausente"}
    ok, texto = rp.resumen_verificacion(rp.verificar_checksums("procesado", tmp_path, ruta))
    assert not ok and "b.csv: distinto" in texto and "c.csv: ausente" in texto


def test_la_verificacion_pasa_si_todo_coincide(tmp_path):
    ruta = _manifiesto(tmp_path, {"data/a.csv": b"hola"})
    ok, texto = rp.resumen_verificacion(rp.verificar_checksums("procesado", tmp_path, ruta))
    assert ok and "1 archivos" in texto


def test_generar_checksums_incluye_tamano_y_metadatos(tmp_path):
    (tmp_path / "raw.csv").write_bytes(b"12345")
    (tmp_path / "final.csv").write_bytes(b"xyz")
    man = rp.generar_checksums(tmp_path, {"raw.csv": {"fuente": "f", "url": "u"}}, ["final.csv"])
    assert man["crudos"]["raw.csv"]["bytes"] == 5 and man["crudos"]["raw.csv"]["fuente"] == "f"
    assert man["procesado"]["final.csv"]["sha256"] == hashlib.sha256(b"xyz").hexdigest()


# ---------- modo rápido: la configuración ----------
def test_la_configuracion_rapida_redirige_las_salidas_y_no_muta_la_original():
    cfg = load_config()
    antes = yaml.safe_dump(cfg)
    r = rp.config_rapida(cfg, "reproduccion")
    assert yaml.safe_dump(cfg) == antes

    def rutas(x):
        if isinstance(x, str):
            yield x
        elif isinstance(x, dict):
            for v in x.values():
                yield from rutas(v)
        elif isinstance(x, list):
            for v in x:
                yield from rutas(v)

    assert not [
        s for s in rutas(r) if s.startswith("reports/")
    ]  # ni una ruta de entrada o salida apunta a lo publicado
    assert (
        r["modelado"]["ajuste_temporal"]["salidas"]["ajuste"]
        == "reproduccion/reports/tables/ajuste.csv"
    )
    assert (
        r["modelado"]["sesgo_sensibilidad"]["ajuste"] == "reproduccion/reports/tables/ajuste.csv"
    )  # lo que lee una etapa es lo que escribió la anterior


def test_la_configuracion_rapida_conserva_pliegues_y_usa_subconjuntos_de_semillas_y_grillas():
    cfg = load_config()
    r = rp.config_rapida(cfg, "x")["modelado"]
    m = cfg["modelado"]
    assert (
        r["protocolo_v3"]["n_splits"] == m["protocolo_v3"]["n_splits"]
        and r["tarea2"]["n_splits"] == m["tarea2"]["n_splits"]
    )
    assert r["ajuste_temporal"]["n_splits_externo"] == m["ajuste_temporal"]["n_splits_externo"]
    # las semillas conservadas son las PRIMERAS de las publicadas: por eso las cifras por pliegue coinciden
    assert r["protocolo_v3"]["semillas"] == m["protocolo_v3"]["semillas"][
        : len(r["protocolo_v3"]["semillas"])
    ] and 0 < len(r["protocolo_v3"]["semillas"]) < len(m["protocolo_v3"]["semillas"])
    assert r["tarea2"]["semilla_cribado"] == m["tarea2"]["semilla_cribado"]
    assert r["tarea2"]["semillas_confirmacion"] == m["tarea2"]["semillas_confirmacion"][:1]
    assert (
        r["ajuste_temporal"]["semillas_externas"] == m["ajuste_temporal"]["semillas_externas"][:1]
    )
    assert r["experimentos"]["semilla"] == m["experimentos"]["semilla"]
    for k in ("objetivo", "imputacion", "seleccion", "modelo"):
        assert set(r["experimentos"][k]) <= set(m["experimentos"][k])
    for k, v in r["tarea2"]["grilla"].items():
        assert set(v) <= set(m["tarea2"]["grilla"][k])
    nombres = {c["nombre"] for c in m["protocolo_v3"]["configuraciones"]}
    assert (
        {c["nombre"] for c in r["protocolo_v3"]["configuraciones"]} == set(rp.V3_RAPIDO) <= nombres
    )
    assert (
        len(r["complejidad"]["fracciones"]) >= 3 and r["complejidad"]["fracciones"][-1] == 1.0
    )  # el exponente log-log necesita 3 puntos
    assert r["ajuste_temporal"]["n_iter"] < m["ajuste_temporal"]["n_iter"]


def test_la_configuracion_rapida_tolera_configuraciones_parciales():
    r = rp.config_rapida(
        {
            "modelado": {"complejidad": {"repeticiones": 5, "fracciones": [0.25, 1.0]}},
            "rutas": {"raw": "data/raw"},
        },
        "s",
    )
    assert r["modelado"]["complejidad"]["repeticiones"] == 2


# ---------- comparación con lo publicado ----------
def test_la_comparacion_exige_igualdad_exacta_solo_en_claves_comunes(tmp_path):
    base = pd.DataFrame(
        {
            "configuracion": ["a", "a", "b"],
            "pliegue": [0, 1, 0],
            "mae": [0.1, 0.2, 0.3],
            "variante": ["defecto", "defecto", "ajustado"],
        }
    )
    base.to_csv(tmp_path / "pub.csv", index=False)
    nuevo = base.iloc[[0, 1, 2]].copy()
    nuevo.loc[2, "mae"] = 0.35
    nuevo = pd.concat(
        [
            nuevo,
            pd.DataFrame(
                {"configuracion": ["z"], "pliegue": [0], "mae": [9.0], "variante": ["defecto"]}
            ),
        ]
    )  # clave que no está en lo publicado
    nuevo.to_csv(tmp_path / "nuevo.csv", index=False)
    c = rp.comparar_csv(tmp_path / "nuevo.csv", tmp_path / "pub.csv")
    assert (
        c["filas_comunes"] == 3 and c["iguales"] == 2 and c["diferencia_max"] == pytest.approx(0.05)
    )
    assert (
        c["casi_iguales"] == 2
    )  # una diferencia real (0,05) no cabe en la tolerancia de punto flotante
    d = rp.comparar_csv(tmp_path / "nuevo.csv", tmp_path / "pub.csv", {"variante": "defecto"})
    assert d["filas_comunes"] == 2 and d["iguales"] == 2 and d["diferencia_max"] == 0.0


# ---------- punto de entrada ----------


def test_el_punto_de_entrada_rechaza_etapas_desconocidas():
    with pytest.raises(SystemExit):
        rep.main(["--etapa", "inexistente"])


def test_las_etapas_llaman_a_los_scripts_existentes_con_la_configuracion_rapida(
    tmp_path, monkeypatch
):
    llamadas = []
    monkeypatch.setattr(rep, "correr", lambda cmd, log: llamadas.append(cmd))
    monkeypatch.setattr(rep, "verificar", lambda grupo, raiz=".": (True, "ok"))
    monkeypatch.setattr(rep, "cifras", lambda e, base, extra="": "cifras")
    salida = tmp_path / "rapido"
    codigo = rep.main(["--etapa", "tarea1", "--rapido", "--salida", str(salida), "--reanudar"])
    assert codigo == 0
    etapas = [c[c.index("evaluar") + 1] for c in llamadas]
    assert etapas == ["v1", "v2", "tarea1"]  # los comandos `desercion evaluar`, en orden
    assert all(c[c.index("--config") + 1] == str(salida / "config_rapida.yaml") for c in llamadas)
    cfg = yaml.safe_load((salida / "config_rapida.yaml").read_text(encoding="utf-8"))
    assert "rapido/reports/tables/experimentos_v1.csv" in cfg["modelado"]["experimentos"][
        "salidas"
    ]["resultados"].replace("\\", "/")


def test_reanudar_se_pasa_solo_a_los_scripts_que_lo_aceptan(tmp_path, monkeypatch):
    llamadas = []
    monkeypatch.setattr(rep, "correr", lambda cmd, log: llamadas.append(cmd))
    monkeypatch.setattr(rep, "verificar", lambda grupo, raiz=".": (True, "ok"))
    monkeypatch.setattr(rep, "cifras", lambda e, base, extra="": "")
    for etapa, script, acepta in [("ajuste", "ajuste", True), ("tarea2", "tarea2", False)]:
        llamadas.clear()
        rep.main(
            [
                "--etapa",
                etapa,
                "--rapido",
                "--salida",
                str(tmp_path / etapa),
                "--reanudar",
                "--sin-verificar",
            ]
        )
        assert (
            llamadas[0][llamadas[0].index("evaluar") + 1] == script
            and ("--reanudar" in llamadas[0]) is acepta
        )


def test_una_etapa_sin_el_dataset_verificado_falla_con_un_mensaje_claro(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setattr(rep, "correr", lambda cmd, log: pytest.fail("no debe correr nada"))
    monkeypatch.setattr(
        rep,
        "verificar",
        lambda grupo, raiz=".": (False, "data/processed/dataset_ies.csv: distinto"),
    )
    codigo = rep.main(
        ["--etapa", "tarea1", "--rapido", "--salida", str(tmp_path / "r"), "--sin-notebooks"]
    )
    assert codigo == 1
    assert "no coincide con el publicado" in caplog.text


def test_el_etl_sin_crudos_se_omite_en_todo_y_falla_si_se_pide_solo(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(
        rp,
        "verificar_checksums",
        lambda grupo, raiz=".", ruta=rp.CHECKSUMS: [
            {"archivo": "data/raw/x.csv", "estado": "ausente", "esperado": "0", "observado": None}
        ],
    )
    monkeypatch.setattr(rep, "correr", lambda cmd, log: pytest.fail("no debe correr nada"))
    assert rep.main(["--etapa", "etl", "--rapido", "--salida", str(tmp_path / "r")]) == 1
    assert "datos crudos" in caplog.text


def test_el_eda_sin_crudos_se_omite_en_todo_y_falla_si_se_pide_solo(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(
        rp,
        "verificar_checksums",
        lambda grupo, raiz=".", ruta=rp.CHECKSUMS: [
            {"archivo": "data/raw/x.csv", "estado": "ausente", "esperado": "0", "observado": None}
        ],
    )
    monkeypatch.setattr(rep, "correr", lambda cmd, log: pytest.fail("no debe correr nada"))
    assert rep.main(["--etapa", "eda", "--rapido", "--salida", str(tmp_path / "r")]) == 1
    assert "necesita los datos crudos" in caplog.text
    monkeypatch.setattr(
        rep, "verificar", lambda grupo, raiz=".": (False, "no importa")
    )  # en `todo` solo se omiten etl y eda; el resto se detiene al no poder verificar el dataset
    assert rep.main(["--etapa", "todo", "--rapido", "--salida", str(tmp_path / "r2")]) == 1
    salida = caplog.text
    assert "etl          omitida" in salida and "eda          omitida" in salida


def test_el_ruido_de_punto_flotante_cuenta_como_casi_igual_pero_no_como_igual_exacto(tmp_path):
    base = pd.DataFrame({"configuracion": ["a", "a"], "pliegue": [0, 1], "mae": [0.1, 0.2]})
    base.to_csv(tmp_path / "pub.csv", index=False)
    ruido = base.copy()
    ruido.loc[0, "mae"] += 3e-15
    ruido.to_csv(tmp_path / "nuevo.csv", index=False)
    c = rp.comparar_csv(tmp_path / "nuevo.csv", tmp_path / "pub.csv")
    assert c["iguales"] == 1 and c["casi_iguales"] == 2 and c["diferencia_max"] < 1e-9
