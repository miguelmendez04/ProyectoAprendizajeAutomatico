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
  - `data/raw/spadies/spadies_diferenciado.csv` — SPADIES → Analíticas → Deserción Anual → tipo de cálculo IES → IES: Diferenciado (variable objetivo; todas las IES en un archivo) y `spadies_nacional.csv` (total nacional, solo contexto y control). Formato y cálculo de la tasa en `reports/etl_log.md`. Sin el diferenciado el pipeline genera un dataset **preliminar** con `tasa_desercion` nula.
  - `data/raw/terridata/TerriData_Dim{2,4,7,8,14,16,25}.txt.zip` — descargas de TerriData (DNP) por dimensión: población, educación, finanzas públicas, desempeño municipal, pobreza y mercado laboral. Fuente de población y NBI/IPM (ya no se usan archivos del DANE). La lectura tarda ~1 min (la dimensión 2 pesa ~1 GB) y se guarda en `data/interim/terridata_largo.csv`. Si faltan, la fuente se omite y se anota en `reports/etl_log.md`.

## Modelado
`src/modelado/particion.py` trae `particion_group_kfold` (por `cod_ies`) y `particion_temporal` (2015–2019 / 2020–2021), con parámetros en `config.yaml: particion`. `min_matriculados_t2` (por defecto 100) excluye IES-año con denominador pequeño. `excluir_posible_cierre` (por defecto true) quita las filas de los últimos semestres de IES que dejaron de reportar, y `reports/tasas_altas.csv` lista las tasas > 50 %. `src/modelado/objetivo.py` ofrece la transformación logit del objetivo (`modelado.transformacion_objetivo`). El dataset no incluye variables de mercado laboral (ver `reports/etl_log.md`).

## Salidas
`data/processed/dataset_<unidad>.csv` · `reports/data_dictionary.md` · `reports/etl_log.md` (la parte posterior a `<!-- pipeline:inicio -->` se regenera en cada corrida; lo anterior es manual) · `notebooks/01_eda.ipynb` · `reports/diagrama_pipeline.md`.
