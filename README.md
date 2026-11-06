# ETL del dataset de deserción en educación superior (Colombia)

Proyecto del curso Aprendizaje Automático (Maestría IA y Ciencia de Datos, UAO).
Especificación: `guia_etl_desercion.md` · Bitácora de decisiones y calidad de datos: `reports/etl_log.md`.

## Autores
- **Simón Colonia** — rol: *[por completar]*
- **Miguel Ángel Méndez** — rol: *[por completar]*

Seguimiento del proyecto: *[enlace a Trello por completar]*

## Uso
```
pip install -r requirements.txt
pytest                              # 173 pruebas
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
Utilidades en `src/modelado/` (parámetros en `config/config.yaml`, secciones `modelado` y `particion`):
- **Partición** (`particion.py`): `particion_group_kfold` (por `cod_ies`) y `particion_temporal` (entrena 2015–2019, prueba 2020–2021). `cargar_dataset(cfg, excluir_sena=..., ventana_modelado=...)` aplica las sensibilidades: `modelado.ventana_modelado` (`[2015, 2021]` por defecto; `[2017, 2021]` evita los años sin MDM ni SNIES) y `modelado.excluir_sena` (quita al SENA al cargar).
- **Variables**: `variables_modelado(df)` devuelve las **numéricas y las categóricas por separado** (`sector` y `caracter`, con etiquetas según SNIES) y `construir_preprocesador` arma el `ColumnTransformer` (imputación, estandarización y one-hot). El tamaño por defecto es `log_matriculados_spadies_t2` (denominador de SPADIES t-2, completo 2015–2021; información de t-2, sin fuga); `log_matriculados_ies_t2` (SNIES) es la alternativa. Quedan fuera por defecto `td_poblacion_total_t1` (cambio de base de las proyecciones 2005 → 2018), `ratio_spadies_snies` y la matrícula bruta. Los indicadores `falta_saber11`, `falta_mdm` y `falta_snies` marcan los bloques de años sin dato (MDM y SNIES: 2015–2016; Saber 11: 2019 y 2021) y entran como numéricas.
- **Objetivo** (`objetivo.py`): transformación logit opcional (`modelado.transformacion_objetivo`).

Filtros del ETL (todos configurables en `config.yaml`): `min_matriculados_t2` (por defecto 100), `excluir_posible_cierre`, `excluir_inconsistencias` (ratio SPADIES/SNIES fuera de 1/3–3) y `excluir_regimen_especial` (lista explícita de 16 escuelas militares y policiales), todos `true`; `es_sena` marca al SENA (9110) y `excluir_sena` (`false`) lo saca. El detalle está en `reports/exclusiones_calidad.csv` y las tasas > 50 % en `reports/tasas_altas.csv`. El dataset no incluye variables de mercado laboral (ver `reports/etl_log.md`).

## Salidas
`data/processed/dataset_<unidad>.csv` · `reports/data_dictionary.md` · `reports/etl_log.md` (la parte posterior a `<!-- pipeline:inicio -->` se regenera en cada corrida; lo anterior es manual) · `notebooks/01_eda.ipynb` · `reports/diagrama_pipeline.md`.
