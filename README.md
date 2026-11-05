# ETL del dataset de deserción en educación superior (Colombia)

Proyecto del curso Aprendizaje Automático (Maestría IA y Ciencia de Datos, UAO).
Especificación: `guia_etl_desercion.md` · Plan: `docs/superpowers/plans/2026-11-05-etl-desercion.md`.

## Uso
```
pip install -r requirements.txt
pytest                              # 78 pruebas
python pipeline.py                  # extract -> transform -> integrate -> validate
python pipeline.py --saltar-extract # reutiliza data/raw/ ya descargado
```
La unidad de análisis (`ies | programa | municipio`), la ventana de años y los IDs de datos.gov.co se configuran en `config/config.yaml`.

## Regenerar `data/raw/` (no se versiona)
- **Automático** (`pipeline.py` sin `--saltar-extract`): Saber 11 (`kgxf-xxbe`, ~3,4 M filas, ~25 min), matrícula SNIES y DIVIPOLA. datos.gov.co falla de forma intermitente (500/503); la descarga reintenta sola.
- **Manual**:
  - `data/raw/spadies/desercion_ies.csv` — export de la consulta pública de SPADIES (variable objetivo). Columnas esperadas en `config.yaml: spadies.columnas`. Sin este archivo el pipeline genera un dataset **preliminar** con `tasa_desercion` nula.
  - `data/raw/dane/dane_poblacion_nbi.csv` y `data/raw/terridata/terridata.csv` (formatos en `config.yaml`). Si faltan, la fuente se omite y se anota en `reports/etl_log.md`.

## Salidas
`data/processed/dataset_<unidad>.csv` · `reports/data_dictionary.md` · `reports/etl_log.md` (la parte posterior a `<!-- pipeline:inicio -->` se regenera en cada corrida; lo anterior es manual) · `notebooks/01_eda.ipynb` · `reports/diagrama_pipeline.md`.
