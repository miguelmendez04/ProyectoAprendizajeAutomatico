"""Pruebas de integración de `test_sesgo_sensibilidad`: necesitan data/processed, data/interim o los resultados publicados (se omiten si faltan)."""

from pathlib import Path

import pytest

HISTORIA = {"tasa_t1", "tasa_t2", "tasa_media_previa"}


@pytest.mark.integration
def test_el_informe_se_genera_desde_los_csv_publicados(tmp_path):
    import yaml

    from desercion_ies.reporting.informe_sesgo import generar_informe

    base = Path("reports/tables")
    archivos = {
        "subgrupos": "subgrupos.csv",
        "sensibilidad": "sensibilidad.csv",
        "colapso": "colapso_ridge.csv",
        "colapso_z": "colapso_ridge_z.csv",
        "colapso_peores": "colapso_ridge_peores.csv",
        "shap_top": "shap_top.csv",
        "shap_grupos": "shap_grupos.csv",
        "permutacion": "permutacion.csv",
        "coeficientes": "coeficientes_ridge.csv",
        "hiperparametros": "hiperparametros_finales.csv",
    }
    if not all((base / v).exists() for v in archivos.values()):
        pytest.skip("faltan los CSV de reports/tables")
    cfg = yaml.safe_load(Path("configs/config.yaml").read_text(encoding="utf-8"))
    s = {k: base / v for k, v in archivos.items()}
    s.update(
        {
            "informe": tmp_path / "r.md",
            "figura_subgrupos": tmp_path / "a.png",
            "figura_shap_t1": tmp_path / "b.png",
            "figura_shap_t2": tmp_path / "c.png",
            "figura_coeficientes": tmp_path / "d.png",
        }
    )
    copia = tmp_path / "05_sesgo_dataset.md"
    original = Path("reports/05_sesgo_dataset.md")
    copia.write_bytes(original.read_bytes())
    import desercion_ies.reporting.informe_sesgo as inf

    actualizar = inf.actualizar_analisis_sesgo
    inf.actualizar_analisis_sesgo = lambda ruta, es, en: actualizar(
        copia, es, en
    )  # no toca el archivo real
    try:
        texto = generar_informe(cfg, s, {"t_total_s": 1.0, "cortes_tamano": [0, 500, 3000, 9000]})
    finally:
        inf.actualizar_analisis_sesgo = actualizar
    assert "## Conclusión" in texto and "colapso" in texto and "En lenguaje simple" in texto
    assert all((tmp_path / f).stat().st_size > 1000 for f in ("a.png", "b.png", "c.png", "d.png"))
    assert "sesgo_final:inicio" in copia.read_text(encoding="utf-8")
