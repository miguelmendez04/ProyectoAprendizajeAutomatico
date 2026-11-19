"""Pruebas de integración de `test_complejidad`: necesitan data/processed, data/interim o los resultados publicados (se omiten si faltan)."""

from pathlib import Path

import pandas as pd
import pytest

from desercion_ies.analysis.costo_protocolo import costo_protocolo


@pytest.mark.integration
def test_el_costo_del_protocolo_cuenta_ajustes_y_toma_los_tiempos_de_los_informes():
    import yaml

    if not Path("reports/tables/ajuste.csv").exists():
        pytest.skip("faltan los resultados anteriores")
    cfg = yaml.safe_load(Path("configs/config.yaml").read_text(encoding="utf-8"))
    p = costo_protocolo(cfg)
    assert len(p) == 9 and (p["ajustes"] > 0).all() and (p["tiempo_pared_s"] > 0).all()
    a = cfg["modelado"]["ajuste_temporal"]
    ajuste = pd.read_csv("reports/tables/ajuste.csv")
    n_def, n_aj = (
        int((ajuste["variante"] == "defecto").sum()),
        int((ajuste["variante"] == "ajustado").sum()),
    )
    fila = p[p["etapa"].str.startswith("Ajuste anidado")].iloc[0]
    assert fila["ajustes"] == n_def + n_aj * (
        a["n_iter"] * a["n_splits_interno"] + 1
    )  # 1 ajuste por defecto + (30 × 3 internos + 1 reajuste) por cada tarea externa
    v1 = pd.read_csv("reports/tables/experimentos_v1.csv")
    assert (
        p.iloc[0]["ajustes"] == (v1["tipo"] == "modelo").sum()
        and p.iloc[0]["tiempo_pared_s"] == 337.0
    )


@pytest.mark.integration
def test_el_informe_se_genera_desde_los_resultados_publicados(tmp_path):
    import yaml

    from desercion_ies.reporting.informe_complejidad import generar_informe

    base = Path("reports/tables")
    archivos = {
        "entorno": "complejidad_entorno.json",
        "complejidad": "complejidad.csv",
        "escalado": "complejidad_escalado.csv",
        "exponentes": "complejidad_exponentes.csv",
        "protocolo": "complejidad_protocolo.csv",
    }
    if not all((base / v).exists() for v in archivos.values()):
        pytest.skip("faltan los CSV de complejidad")
    cfg = yaml.safe_load(Path("configs/config.yaml").read_text(encoding="utf-8"))
    s = {k: base / v for k, v in archivos.items()}
    s.update(
        {
            "informe": tmp_path / "r.md",
            "figura_costo_error": tmp_path / "a.png",
            "figura_escalado": tmp_path / "b.png",
        }
    )
    texto = generar_informe(cfg, s)
    assert (
        "## Conclusión" in texto
        and "Complejidad teórica" in texto
        and "Costo del protocolo" in texto
    )
    assert (tmp_path / "a.png").stat().st_size > 1000 and (tmp_path / "b.png").stat().st_size > 1000
