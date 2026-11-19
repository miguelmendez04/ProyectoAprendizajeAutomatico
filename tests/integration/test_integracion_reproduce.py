"""Pruebas de integración de `test_reproduce`: necesitan data/processed, data/interim o los resultados publicados (se omiten si faltan)."""

import logging
from pathlib import Path

import pandas as pd
import pytest
import yaml

from desercion_ies import reproduce as rep
from desercion_ies import reproducibilidad as rp

# ---------- checksums ----------


@pytest.mark.integration
def test_el_manifiesto_publicado_coincide_con_el_dataset_del_repositorio():
    if (
        not Path("data/checksums.json").exists()
        or not Path("data/processed/dataset_ies.csv").exists()
    ):
        pytest.skip("falta el manifiesto o el dataset")
    ok, texto = rep.verificar("procesado")
    assert ok, texto
    man = rp.leer_checksums()
    assert len(man["crudos"]) == 14 and all(
        len(v["sha256"]) == 64 and v.get("url") and v.get("fecha_descarga")
        for v in man["crudos"].values()
    )


@pytest.mark.integration
def test_solo_verificar_imprime_el_estado_de_cada_archivo(caplog):
    caplog.set_level(logging.INFO)
    if not Path("data/checksums.json").exists():
        pytest.skip("falta el manifiesto")
    assert rep.main(["--solo-verificar"]) == 0
    salida = caplog.text
    assert "procesado" in salida and "dataset_ies.csv" in salida and "crudos" in salida


@pytest.mark.integration
def test_rapido_corre_de_punta_a_punta_con_datos_de_juguete(tmp_path):
    """Una etapa completa (complejidad) por el punto de entrada: subproceso real, configuración rápida y dataset sintético; las salidas quedan en la carpeta pedida."""
    from tests.unit.test_complejidad import _cfg as cfg_complejidad
    from tests.unit.test_sesgo_sensibilidad import _datos

    toy = cfg_complejidad(tmp_path)
    (tmp_path / "processed").mkdir()
    _datos().to_csv(tmp_path / "processed" / "dataset_ies.csv", index=False)
    toy.update(
        unidad_analisis="ies",
        rutas={
            "raw": str(tmp_path / "raw"),
            "interim": str(tmp_path / "interim"),
            "processed": str(tmp_path / "processed"),
            "reports": "reports",
        },
    )
    toy["modelado"]["complejidad"]["fracciones"] = [0.5, 0.75, 1.0]
    toy["modelado"]["complejidad"]["kmeans_k"] = (
        6  # los nombres de los componentes llevan estos parámetros y la tabla teórica los espera como en config.yaml
    )
    toy["modelado"]["experimentos"]["k_info_mutua"] = 15
    toy["modelado"]["complejidad"]["salidas"] = {
        k: f"reports/{n}"
        for k, n in {
            "entorno": "tables/complejidad_entorno.json",
            "complejidad": "tables/complejidad.csv",
            "escalado": "tables/complejidad_escalado.csv",
            "exponentes": "tables/complejidad_exponentes.csv",
            "protocolo": "tables/complejidad_protocolo.csv",
            "informe": "07_complejidad.md",
            "figura_costo_error": "figures/complejidad_costo_error.png",
            "figura_escalado": "figures/complejidad_escalado.png",
        }.items()
    }
    toy["modelado"]["ajuste_temporal"]["salidas"] = {
        "ajuste_resumen": "reports/tables/ajuste_resumen.csv"
    }  # el informe de complejidad lee el error de los modelos
    cfg_path = tmp_path / "toy.yaml"
    cfg_path.write_text(yaml.safe_dump(toy, allow_unicode=True), encoding="utf-8")
    salida = tmp_path / "salida"
    publicado_antes = (
        Path("reports/tables/complejidad.csv").read_bytes()
        if Path("reports/tables/complejidad.csv").exists()
        else None
    )
    codigo = rep.main(
        [
            "--etapa",
            "complejidad",
            "--rapido",
            "--config",
            str(cfg_path),
            "--salida",
            str(salida),
            "--sin-verificar",
        ]
    )
    assert codigo == 0
    c = pd.read_csv(salida / "reports/tables/complejidad.csv")
    assert len(c) == 15 and (c["ajuste_mediana_s"] > 0).all()  # los 15 componentes se midieron
    assert (salida / "reports/tables/complejidad_escalado.csv").exists() and (
        salida / "logs/evaluar_complejidad.log"
    ).exists()
    if publicado_antes is not None:
        assert (
            Path("reports/tables/complejidad.csv").read_bytes() == publicado_antes
        )  # lo publicado no se tocó
