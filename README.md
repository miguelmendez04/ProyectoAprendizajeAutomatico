# ETL del dataset de deserción en educación superior (Colombia)

Proyecto del curso Aprendizaje Automático (Maestría IA y Ciencia de Datos, UAO).
Especificación: `guia_etl_desercion.md`. Plan: `docs/superpowers/plans/`.

## Uso
```
pip install -r requirements.txt
pytest
python pipeline.py            # extract -> transform -> integrate -> validate
```
La unidad de análisis (`ies | programa | municipio`) y los IDs de datos.gov.co se configuran en `config/config.yaml`.
