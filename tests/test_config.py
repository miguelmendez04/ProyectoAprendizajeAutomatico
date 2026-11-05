"""Pruebas de carga y validación de config.yaml."""
import pytest
import yaml

from src.config import load_config


def test_carga_config_valida():
    cfg = load_config("config/config.yaml")
    assert cfg["unidad_analisis"] in {"ies", "programa", "municipio"}


def test_unidad_invalida_lanza_valueerror(tmp_path):
    ruta = tmp_path / "config.yaml"
    ruta.write_text(yaml.safe_dump({"unidad_analisis": "colegio"}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(ruta)
