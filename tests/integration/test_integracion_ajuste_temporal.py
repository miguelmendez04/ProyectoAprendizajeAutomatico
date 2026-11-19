"""Pruebas de integración de `test_ajuste_temporal`: necesitan data/processed, data/interim o los resultados publicados (se omiten si faltan)."""

import pytest

MUNICIPIOS = ["11001", "05001", "76001", "68001", "13001", "23001", "08001", "66001"]


@pytest.mark.integration
def test_el_informe_se_genera_desde_los_csv_publicados(tmp_path):
    from pathlib import Path

    import yaml

    from desercion_ies.reporting.informe_ajuste import generar_informe

    base = Path("reports/tables")
    if (
        not (base / "ajuste_resumen.csv").exists()
        or not (base / "temporal_predicciones.csv").exists()
    ):
        pytest.skip("faltan los CSV de reports/tables")
    cfg = yaml.safe_load(Path("configs/config.yaml").read_text(encoding="utf-8"))
    s = {
        k: base / v
        for k, v in {
            "ajuste_resumen": "ajuste_resumen.csv",
            "hiperparametros": "ajuste_hiperparametros.csv",
            "temporal": "temporal.csv",
            "temporal_grupos": "temporal_grupos.csv",
            "temporal_predicciones": "temporal_predicciones.csv",
            "arranque_frio_resumen": "arranque_frio_municipio_resumen.csv",
            "arranque_frio_pliegues": "arranque_frio_pliegues.csv",
        }.items()
    }
    s.update(
        {
            "informe": tmp_path / "r.md",
            "figura_ajuste": tmp_path / "a.png",
            "figura_temporal": tmp_path / "t.png",
        }
    )
    texto = generar_informe(
        cfg, s, {"t_ajuste_s": 1.0, "t_temporal_s": 1.0, "t_frio_s": 1.0, "t_total_s": 3.0}
    )
    assert (
        "## Conclusión" in texto
        and "Nota de diagnóstico" in texto
        and "Arranque en frío estricto" in texto
    )
    assert (tmp_path / "a.png").stat().st_size > 1000 and (tmp_path / "t.png").stat().st_size > 1000
