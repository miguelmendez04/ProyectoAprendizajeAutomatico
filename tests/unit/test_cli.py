"""CLI `desercion`, rutas del paquete, requirements generado desde pyproject y estructura del repositorio."""

import importlib
import logging
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from desercion_ies import cli, paths

RAIZ = Path(__file__).resolve().parents[2]


# ---------- rutas ----------
def test_la_raiz_se_detecta_sin_depender_del_directorio_actual(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert paths.encontrar_raiz() == RAIZ
    assert paths.PROJECT_ROOT == RAIZ and (paths.CONFIG_FILE).is_file()
    for ruta in (
        paths.CONFIGS,
        paths.TABLES,
        paths.FIGURES,
        paths.REPORTS_ETL,
        paths.REPORTS_HISTORICO,
        paths.NOTEBOOKS,
        paths.DOCS,
    ):
        assert RAIZ in ruta.parents and ruta.is_dir(), ruta


def test_la_variable_de_entorno_manda_sobre_la_deteccion(tmp_path, monkeypatch):
    monkeypatch.setenv("DESERCION_ROOT", str(tmp_path))
    assert paths.encontrar_raiz() == tmp_path.resolve()


# ---------- argumentos ----------
def test_los_globales_se_extraen_de_cualquier_posicion():
    resto, verbose, quiet = cli._extraer_globales(
        ["evaluar", "-v", "tarea2", "--config", "x.yaml", "--verbose", "-q"]
    )
    assert resto == ["evaluar", "tarea2", "--config", "x.yaml"] and verbose == 2 and quiet


def test_el_logging_va_a_la_salida_estandar_y_respeta_el_nivel(capsys):
    cli.configurar_logging(0, False)
    logging.getLogger("desercion_ies.prueba").info("mensaje informativo")
    logging.getLogger("desercion_ies.prueba").debug("mensaje de depuración")
    salida = capsys.readouterr().out
    assert "mensaje informativo" in salida and "depuración" not in salida
    cli.configurar_logging(1, False)
    logging.getLogger("desercion_ies.prueba").debug("detalle")
    assert "detalle" in capsys.readouterr().out
    cli.configurar_logging(0, True)
    logging.getLogger("desercion_ies.prueba").info("silenciado")
    assert capsys.readouterr().out == ""
    cli.configurar_logging(0, False)


def test_las_rutas_del_usuario_son_relativas_a_donde_invoco_y_las_predeterminadas_a_la_raiz(
    tmp_path,
):
    import argparse

    a = argparse.Namespace(config="mi.yaml", salida="reproduccion", entrada="datos.csv")
    cli._a_absoluta(a, tmp_path, {"config": "configs/config.yaml", "salida": "reproduccion"})
    assert a.config == str((tmp_path / "mi.yaml").resolve())
    assert a.entrada == str(
        (tmp_path / "datos.csv").resolve()
    )  # sin valor predeterminado: lo puso el usuario
    assert (
        a.salida == "reproduccion"
    )  # igual al predeterminado: se queda relativo a la raíz del proyecto


def test_evaluar_ofrece_las_siete_etapas_y_cada_una_expone_configurar_y_ejecutar():
    assert set(cli.EVALUAR) == {"v1", "v2", "tarea1", "tarea2", "ajuste", "sesgo", "complejidad"}
    for etapa, (modulo, _) in cli.EVALUAR.items():
        if etapa == "complejidad":
            continue  # su importación pesada se prueba en el subproceso de abajo
        m = importlib.import_module(modulo)
        assert callable(m.configurar) and callable(m.ejecutar), etapa


def test_la_ayuda_de_una_etapa_sale_con_codigo_0_y_lista_sus_opciones():
    r = subprocess.run(
        [sys.executable, "-m", "desercion_ies", "evaluar", "ajuste", "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONPATH": str(RAIZ / "src"), "PYTHONIOENCODING": "utf-8"},
    )
    assert r.returncode == 0 and "--reanudar" in r.stdout and "--solo-informe" in r.stdout


def test_la_cli_funciona_desde_otro_directorio_y_verifica_los_checksums(tmp_path):
    if (
        not (RAIZ / "data" / "checksums.json").exists()
        or not (RAIZ / "data" / "processed" / "dataset_ies.csv").exists()
    ):
        pytest.skip("falta el manifiesto o el dataset")
    r = subprocess.run(
        [sys.executable, "-m", "desercion_ies", "verificar"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(RAIZ / "src"), "PYTHONIOENCODING": "utf-8"},
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert (
        "procesado" in r.stdout
        and "dataset_ies.csv" in r.stdout
        and "verificación correcta" in r.stdout
    )


def test_verificar_falla_si_un_archivo_difiere(tmp_path, monkeypatch, caplog):
    from desercion_ies import reproducibilidad as rp

    caplog.set_level(logging.INFO)
    monkeypatch.setattr(
        rp,
        "verificar_checksums",
        lambda grupo, raiz=".", ruta=None: [
            {"archivo": f"{grupo}.csv", "estado": "distinto" if grupo == "procesado" else "ausente"}
        ],
    )
    assert cli._verificar([]) == 1
    assert "FALLÓ" in caplog.text
    monkeypatch.setattr(
        rp,
        "verificar_checksums",
        lambda grupo, raiz=".", ruta=None: [
            {"archivo": f"{grupo}.csv", "estado": "ausente" if grupo == "crudos" else "ok"}
        ],
    )
    assert cli._verificar([]) == 0  # sin los crudos, el dataset verificado basta
    assert cli._verificar(["--estricto"]) == 1  # con --estricto también deben estar los crudos


def test_main_despacha_a_la_etapa_pedida(monkeypatch):
    llamadas = []

    class Falsa:
        __doc__ = "etapa falsa"

        @staticmethod
        def configurar(ap):
            ap.add_argument("--config", default="configs/config.yaml")
            ap.add_argument("--reanudar", action="store_true")

        @staticmethod
        def ejecutar(args):
            llamadas.append(args)

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda nombre: Falsa if nombre.endswith("runners.tarea2") else importlib.__import__(nombre),
    )
    monkeypatch.chdir(RAIZ)
    assert cli.main(["evaluar", "tarea2", "--reanudar", "-q"]) == 0
    assert (
        len(llamadas) == 1
        and llamadas[0].reanudar is True
        and llamadas[0].config == "configs/config.yaml"
    )
    cli.configurar_logging(0, False)


def test_evaluar_complejidad_fija_un_hilo_antes_de_cargar_la_etapa(monkeypatch):
    for v in cli.HILOS:
        monkeypatch.delenv(v, raising=False)
    cli.limitar_hilos()
    assert all(os.environ[v] == "1" for v in cli.HILOS)


# ---------- empaquetado y estructura ----------
def test_requirements_txt_esta_generado_desde_pyproject():
    sys.path.insert(0, str(RAIZ / "tools"))
    try:
        import generar_requirements
    finally:
        sys.path.pop(0)
    assert (RAIZ / "requirements.txt").read_text(
        encoding="utf-8"
    ) == generar_requirements.contenido()


def test_pyproject_declara_el_paquete_el_comando_y_el_marcador_de_integracion():
    p = tomllib.loads((RAIZ / "pyproject.toml").read_text(encoding="utf-8"))
    assert p["project"]["name"] == "desercion-ies" and p["project"]["requires-python"] == ">=3.12"
    assert p["project"]["scripts"]["desercion"] == "desercion_ies.cli:main"
    assert any(m.startswith("integration") for m in p["tool"]["pytest"]["ini_options"]["markers"])
    assert all("==" in d for d in p["project"]["dependencies"])  # versiones fijas
    assert {"pytest", "ruff", "pre-commit", "nbconvert"} <= {
        d.split("==")[0] for d in p["project"]["optional-dependencies"]["dev"]
    }


def test_no_quedan_scripts_de_etapa_en_la_raiz_ni_rutas_de_la_estructura_anterior():
    for viejo in (
        "pipeline.py",
        "reproducir.py",
        "pytest.ini",
        "run_experimentos.py",
        "run_tarea2.py",
        "config",
        "src/modelado",
        "src/extract",
    ):
        assert not (RAIZ / viejo).exists(), viejo
    assert not list(RAIZ.glob("run_*.py"))
    for nuevo in (
        "configs/config.yaml",
        "docs/guia_etl.md",
        "reports/01_etl.md",
        "reports/tables",
        "reports/figures",
        "reports/historico",
        "reports/etl",
        "src/desercion_ies/etl/pipeline.py",
        "src/desercion_ies/evaluation/protocolo.py",
        "src/desercion_ies/analysis/complejidad.py",
    ):
        assert (RAIZ / nuevo).exists(), nuevo


def test_el_codigo_no_usa_el_paquete_anterior():
    import re

    patron = re.compile(r"\b(src\.modelado|from src\b|import src\b)")
    ofensores = [
        str(p.relative_to(RAIZ))
        for carpeta in ("src", "tests", "tools")
        for p in (RAIZ / carpeta).rglob("*.py")
        if patron.search(p.read_text(encoding="utf-8")) and p != Path(__file__).resolve()
    ]
    assert ofensores == []


def test_las_llamadas_a_logging_del_paquete_son_validas():
    """Un `print(..., sep=...)` convertido a `log.info` a medias no falla hasta que corre la etapa: aquí se revisa la forma de cada llamada."""
    import ast

    problemas = []
    for p in (RAIZ / "src").rglob("*.py"):
        for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr in ("debug", "info", "warning", "error")
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "log"
            ):
                if {k.arg for k in n.keywords} - {"exc_info", "extra"}:
                    problemas.append(f"{p.name}:{n.lineno} argumentos con nombre no admitidos")
                if len(n.args) > 1 and not (
                    isinstance(n.args[0], ast.Constant) and "%" in n.args[0].value
                ):
                    problemas.append(f"{p.name}:{n.lineno} varios argumentos sin formato %")
    assert problemas == []
