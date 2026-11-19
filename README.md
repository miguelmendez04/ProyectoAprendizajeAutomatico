# ETL del dataset de deserción en educación superior (Colombia)

Proyecto del curso Aprendizaje Automático (Maestría IA y Ciencia de Datos, UAO).
Especificación: `guia_etl_desercion.md` · Bitácora de decisiones y calidad de datos: `reports/etl_log.md`.

## Autores
- **Miguel Ángel Méndez Rodríguez** — Data Engineer y Project Manager.
- **Simón Colonia** — Model Developer y Evaluation Lead.
- **Reporting Lead** — compartido.

Seguimiento del proyecto: [tablero de Trello](https://trello.com/b/sgxvalz4/proyecto-ml-deserci%C3%B3n-en-ies-uao).

## Uso
```
pip install -r requirements.txt
pytest                              # 348 pruebas
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
- **Experimentos** (`experimentos.py`, `informe_experimentos.py`; `python run_experimentos.py`): grilla de 36 configuraciones (objetivo ninguna/logit × imputación mediana/KNN × selección ninguna/información mutua/PCA × Ridge/Random Forest/XGBoost, configurable en `modelado.experimentos`) con GroupKFold por IES de 5 pliegues idénticos para todas, más las líneas base mediana e ingenua (tasa del año anterior de la IES). Escribe `reports/resultados/experimentos_v1.csv` (por pliegue) y `experimentos_v1_resumen.csv`, `reports/resultados_v1.md` y `reports/figuras/mae_configuraciones.png`.
- **Formulaciones con historia** (`formulaciones.py`, `informe_formulaciones.py`; `python run_formulaciones.py`): A contexto, B historia + contexto, B solo historia y C corrección sobre la ingenua (+ C solo historia), con los mismos pliegues que la v1; reporta el MAE total y por filas con/sin historia. Las columnas de historia (`tasa_t1`, `tasa_t2`, `tasa_media_previa`, `n_anios_previos`, `falta_historia`, solo años anteriores a t) las agrega el ETL (`incluir_historia`) y no son explicativas por defecto. Salidas: `reports/resultados/experimentos_v2.csv`, `reports/resultados_v2.md`, `reports/figuras/mae_formulaciones.png`.
- **Protocolo v3** (`protocolo.py`, `informe_v3.py`; `python run_protocolo_v3.py`): evaluación rigurosa de un conjunto de 15 configuraciones fijado antes de correr, con GroupKFold por IES repetido 5 veces (25 ajustes), IC 95 % con la corrección de Nadeau y Bengio, prueba de Wilcoxon pareada contra la ingenua y MAE por filas con/sin historia. Incluye la variante «historia rezagada» (sin la tasa de t-1) y su ingenua rezagada. La historia del dataset usa las tasas anuales de SPADIES desde 1999 (`historia_previa_a_ventana`); las filas sin historia (34) se listan en `reports/sin_historia.csv`.
- **Tarea 2: estimación sin historia** (`metodos.py`, `tarea2.py`, `informe_tarea2.py`; `python run_tarea2.py`; notebook `notebooks/04_tarea2.ipynb`): predice la tasa solo con el contexto (formulación A) con GroupKFold por IES. Líneas base: mediana y mediana por carácter × sector. Métodos del profesor como transformers de scikit-learn (KernelPCA RBF, autoencoder n→16→8→16→n con parada temprana, clusters K-means con k por silueta que agregan el cluster y las distancias a los centroides). Protocolo en dos etapas: cribado de 120 configuraciones (una semilla) y confirmación 5 × 5 con semillas distintas, IC 95 % y t corregida. Salidas: `reports/resultados_tarea2.md`, `reports/resultados/experimentos_tarea2_*.csv`, `reports/figuras/tarea2_metodologias.png` y `clusters.png`.
- **Ajuste, particiones temporales y arranque en frío** (`ajuste.py`, `temporal.py`, `arranque_frio.py`, `informe_ajuste.py`; `python run_ajuste_temporal.py`; notebook `notebooks/05_ajuste_temporal.ipynb`): ajuste de hiperparámetros con validación anidada (externa GroupKFold por IES 5 × 5, interna RandomizedSearchCV de 30 iteraciones), particiones temporales (prepandemia y estrés, con bootstrap por IES) y arranque en frío estricto agrupando por municipio. Salidas: `reports/resultados_ajuste_temporal.md`, `reports/resultados/ajuste.csv`, `temporal.csv`, `arranque_frio_municipio.csv`, `reports/figuras/ajuste_vs_defecto.png` y `temporal_por_anio.png`.
- **Sesgo** (`sesgo.py`, `geografia.py`, `mitigacion.py`): análisis de sesgo del dataset y opciones de mitigación, **todas apagadas por defecto** (el dataset y los valores por defecto no cambian): `modelado.pesos_muestrales` (`ninguno` | `matriculados` | `raiz_matriculados` | `inverso_grupo` | `inverso_ies`), `modelado.umbral_denominador` (p. ej. 200), `modelado.indicador_panel_incompleto`, `particion.agrupar_por` (`departamento`) y `particion.estratificar_por` (`[sector, caracter]`); además `particion_dejando_fuera` (prueba en Bogotá) y `metricas_por_subgrupo` (MAE por sector, carácter, región, ciudad y tamaño). Informe en `reports/analisis_sesgo.md` y figuras en `notebooks/02_sesgo.ipynb` (`python notebooks/construir_sesgo.py`).

Filtros del ETL (todos configurables en `config.yaml`): `min_matriculados_t2` (por defecto 100), `excluir_posible_cierre`, `excluir_inconsistencias` (ratio SPADIES/SNIES fuera de 1/3–3) y `excluir_regimen_especial` (lista explícita de 16 escuelas militares y policiales), todos `true`; `es_sena` marca al SENA (9110) y `excluir_sena` (`false`) lo saca. El detalle está en `reports/exclusiones_calidad.csv` y las tasas > 50 % en `reports/tasas_altas.csv`. El dataset no incluye variables de mercado laboral (ver `reports/etl_log.md`).

## Salidas
`data/processed/dataset_<unidad>.csv` · `reports/data_dictionary.md` · `reports/etl_log.md` (la parte posterior a `<!-- pipeline:inicio -->` se regenera en cada corrida; lo anterior es manual) · `notebooks/01_eda.ipynb` · `notebooks/02_sesgo.ipynb` · `reports/analisis_sesgo.md` · `reports/diagrama_pipeline.md` · `reports/resultados_v1.md` · `reports/resultados_v2.md` · `reports/resultados_v3.md` · `reports/resultados_tarea2.md` · `reports/resultados_ajuste_temporal.md` · `reports/sin_historia.csv` · `reports/resultados/` · `reports/figuras/`.
