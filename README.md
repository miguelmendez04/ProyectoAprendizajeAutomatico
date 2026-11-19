# Deserción en educación superior en Colombia: ETL y modelado (IES × año, 2015–2021)

> **English summary.** ETL and machine-learning experiments to estimate the dropout rate of Colombian higher-education institutions (IES × year, 2015–2021; 1,928 rows, 295 institutions) from SPADIES (target), SNIES, Saber 11 and TerriData. **Task 1** (with the institution's own history): does a model beat the naive forecast "this year's rate = last year's rate"? **Task 2** (no history, "cold start"): can the municipal and institutional context alone beat the median by character × sector? Evaluation: GroupKFold by institution (5 × 5), 95 % CIs with the Nadeau–Bengio correction, nested hyper-parameter tuning, temporal splits, bias analysis, and measured computational cost. **Main findings:** the best Task 1 model (ridge correcting the naive forecast) improves MAE by 4.4 % over the naive baseline (0.0245 vs 0.0256), small but significant under both tests; in Task 2 no model beats the median-by-group baseline with statistical firmness (best: ridge, +0.9 %, p = 0.60). **Reproduce:** `pip install -r requirements.txt` then `python reproducir.py --etapa todo --rapido` (< 10 min, writes to `reproduccion/`, does not touch published results) or `--etapa todo` (≈ 1.5 h). Code: MIT; data keep the terms of their official sources ([`docs/DATOS.md`](docs/DATOS.md)).

Proyecto del curso Aprendizaje Automático (Maestría IA y Ciencia de Datos, UAO). Entrega: 26 de noviembre de 2026 (tag `v1.0-entrega`).
Especificación: `guia_etl_desercion.md` · Bitácora de decisiones y calidad de datos: `reports/etl_log.md`.

## Autores
- **Miguel Ángel Méndez Rodríguez** — Data Engineer y Project Manager.
- **Simón Colonia** — Model Developer y Evaluation Lead.
- **Reporting Lead** — compartido.

Seguimiento del proyecto: [tablero de Trello](https://trello.com/b/sgxvalz4/proyecto-ml-deserci%C3%B3n-en-ies-uao).

## El problema y las dos tareas
La **tasa de deserción** anual de cada IES (fuente: SPADIES; desertores de t / matriculados de t−2, ponderada por semestre) se estima con variables de contexto rezagadas (Saber 11 y TerriData en t−1, matrícula SNIES en t−2). La unidad es IES × año; el dataset final tiene 1.928 filas de 295 IES.
- **Tarea 1 — con historia de la IES.** Se conoce la tasa del año anterior. Línea base: la **ingenua** (tasa de t−1). Mejor formulación: *C solo historia*, un modelo que corrige la ingenua con la historia de la propia IES.
- **Tarea 2 — sin historia («arranque en frío»).** Solo el contexto (formulación A), incluidos KernelPCA, autoencoder y clusters. Línea base: la **mediana por carácter × sector**.

## Resultados principales
MAE (menor es mejor) con validación anidada: GroupKFold por IES 5 × 5 = 25 ajustes; IC 95 % con la corrección de Nadeau y Bengio. Fuente: `reports/resultados_ajuste_temporal.md` (`reports/resultados/ajuste_resumen.csv`). «p» es la prueba t corregida (y Wilcoxon) frente a la línea base.

| Tarea | Modelo | MAE [IC 95 %] | Mejora frente a la línea base | p (t corregida; Wilcoxon) |
|---|---|---|---|---|
| 1 | **Ingenua** (tasa de t−1) | 0,0256 [0,0225; 0,0288] | — | — |
| 1 | **Ridge de corrección, C solo historia** (ajustado) | 0,0245 [0,0212; 0,0278] | +4,4 % | 0,011; < 0,001 |
| 2 | **Mediana por carácter × sector** | 0,0400 [0,0345; 0,0455] | — | — |
| 2 | Mejor modelo: ridge, logit, información mutua, sin clusters (ajustado) | 0,0396 [0,0345; 0,0448] | +0,9 % | 0,596; 0,127 |

**Lectura.** En la Tarea 1 el ridge de corrección supera a la ingenua con significancia en ambas pruebas, pero la ventaja es pequeña (0,001 de MAE) y en las particiones temporales (prepandemia y 2020–2021) ningún modelo se distingue de la ingenua. En la Tarea 2 el contexto no mejora a la mediana por carácter × sector con firmeza estadística; xgboost y random forest empeoran con hiperparámetros por defecto y quedan «sin diferencia firme» tras el ajuste. Costo: el ridge de la Tarea 1 se ajusta en ~9 ms; random forest, en ~1,3 s sin menos error (`reports/resultados_complejidad.md`).

## Cómo reproducir
**Requisitos:** Python 3.14, las versiones fijas de `requirements.txt` y el dataset `data/processed/dataset_ies.csv` (ya está en el repositorio). Los datos crudos **no** son necesarios para el modelado; solo para la etapa `etl` (ver [`docs/DATOS.md`](docs/DATOS.md): URL, pasos de descarga, nombre y carpeta de cada archivo, fecha y checksums).

```
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
python reproducir.py --solo-verificar         # SHA-256 del dataset (y de los crudos, si están)
python reproducir.py --etapa todo --rapido    # verificación en < 10 min; escribe en reproduccion/
python reproducir.py --etapa todo             # corrida completa (≈ 1,5 h)
python -m pytest -q                            # 399 pruebas (≈ 2 min)
```
Opciones: `--etapa {etl,eda,tarea1,tarea2,ajuste,sesgo,complejidad,todo}`, `--rapido`, `--reanudar` (reutiliza lo ya calculado donde el script lo permite: ajuste anidado, sesgo y complejidad guardan puntos de control), `--sin-notebooks`, `--sin-verificar`. Al terminar imprime el tiempo y las cifras clave de cada etapa. Cada etapa llama a los scripts existentes (`pipeline.py`, `run_experimentos.py`, `run_formulaciones.py`, `run_protocolo_v3.py`, `run_tarea2.py`, `run_ajuste_temporal.py`, `run_sesgo_sensibilidad.py`, `run_complejidad.py`) con sus semillas, sin cambiar su lógica.

| Etapa | Qué corre | Completa (≈) | `--rapido` (≈) |
|---|---|---|---|
| `etl` | `pipeline.py --saltar-extract` (necesita los 14 crudos) | 2 min | 2 min |
| `eda` | cuadernos 01 y 02 | 2 min | 2 min |
| `tarea1` | v1 (36 configuraciones), v2 (formulaciones), v3 (protocolo 5 × 5) | 31 min | 0,5 min |
| `tarea2` | cribado + confirmación + clusters | 15 min | 1,5 min |
| `ajuste` | anidado + temporales + arranque en frío | 31 min | 1,5 min |
| `sesgo` | subgrupos, sensibilidad, colapso, SHAP | 2 min | 1 min |
| `complejidad` | medición con un hilo | 3 min | 2 min |

`--etapa todo --rapido` tardó **585 s (9,8 min)** con el ETL y el EDA incluidos; desde un clon limpio, sin los crudos (`etl` y `eda` se omiten), **381 s (6,3 min)**. Los tiempos son de la máquina original (**Windows 11, Intel Core i5-12500H, 16 hilos, 31,6 GiB**) y **dependen de la máquina**: las etapas pesadas usan 16 procesos en paralelo. La corrida completa regenera además los cuadernos 04–07 (`--sin-notebooks` lo evita).

**Modo `--rapido`.** Solo cambia la *configuración*: menos semillas (se conservan las primeras de las publicadas), grillas más chicas, `n_iter` = 3 en la búsqueda interna, 100 réplicas de bootstrap y 2 repeticiones en la medición. Escribe todo en `reproduccion/` (no se versiona) y no toca `reports/`. Al final **compara contra los CSV publicados**: las cifras por pliegue de las configuraciones comunes coinciden **exactamente** (v1, v3, Tarea 2, ajuste por defecto, arranque en frío), salvo dos diferencias esperadas y documentadas en [`docs/REPRODUCIBILIDAD.md`](docs/REPRODUCIBILIDAD.md) (la v2 con historia y el ajuste con `n_iter` reducido). Los tiempos medidos (complejidad) nunca coinciden: dependen de la máquina.

## Mapa de informes y cuadernos
| Orden | Cuaderno | Pregunta que responde | Informe(s) |
|---|---|---|---|
| 01 | `notebooks/01_eda.ipynb` | ¿Cómo es el dataset, qué calidad tiene y qué se excluyó? | `reports/data_dictionary.md`, `etl_log.md`, `notebooks/conclusiones_eda.md` |
| 02 | `notebooks/02_sesgo.ipynb` | ¿Hay sesgo de selección, supervivencia, geografía o fuga temporal? | `reports/analisis_sesgo.md` |
| 03 | — (no existe: la numeración salta; v1–v3 se documentan solo en informes) | ¿Qué configuraciones y formulaciones funcionan, con qué protocolo? | `resultados_v1.md`, `resultados_v2.md` (historia desde 2015; superada), `resultados_v3.md` (referencia vigente) |
| 04 | `notebooks/04_tarea2.ipynb` | Sin historia, ¿ayudan KernelPCA, el autoencoder o los clusters? | `resultados_tarea2.md` |
| 05 | `notebooks/05_ajuste_temporal.ipynb` | ¿El ajuste de hiperparámetros, la pandemia o el arranque en frío cambian las conclusiones? | `resultados_ajuste_temporal.md` |
| 06 | `notebooks/06_sesgo_interpretabilidad.ipynb` | ¿Dónde falla más el modelo, qué tan robusto es y qué variables usa? | `resultados_sesgo_sensibilidad.md` |
| 07 | `notebooks/07_complejidad.ipynb` | ¿Cuánto cuesta cada modelo y cómo escala con más filas? | `resultados_complejidad.md` |

Todos los cuadernos se ejecutan de punta a punta sin errores (`jupyter nbconvert --execute`); ver [`docs/REPRODUCIBILIDAD.md`](docs/REPRODUCIBILIDAD.md).

## Estructura del repositorio
```
reproducir.py            punto de entrada único (etapas, --rapido, --reanudar)
pipeline.py              ETL: extract → transform → integrate → validate
run_*.py                 un script por etapa de modelado (v1, v2, v3, Tarea 2, ajuste, sesgo, complejidad)
config/config.yaml       toda la configuración (fuentes, filtros, grillas, semillas, salidas)
src/{extract,transform,integrate,validate,homologation}/   ETL
src/modelado/            partición, métodos, protocolo, ajuste, sesgo, complejidad e informes
src/reproducibilidad.py  checksums, configuración del modo rápido, comparación con lo publicado
data/raw/                crudos (no versionados; ver docs/DATOS.md)
data/processed/          dataset_ies.csv (versionado)   ·   data/checksums.json (SHA-256)
reports/                 informes .md, resultados/*.csv, figuras/
notebooks/               01 a 07 (+ construir_*.py que los generan)
tests/                   pruebas (pytest)
docs/                    DATOS.md, REPRODUCIBILIDAD.md
```

## Limitaciones principales
- **Tamaño y alcance:** 1.928 filas de 295 IES; la unidad es la institución, no el estudiante. Las conclusiones no se trasladan a programas ni a personas.
- **Ventaja pequeña de los modelos:** en la Tarea 1 es de ~4 %, y ningún modelo supera a la ingenua con IC que excluya 0 en las particiones temporales; en la Tarea 2 no hay mejora firme sobre la mediana por carácter × sector.
- **Disponibilidad de la historia:** la Tarea 1 supone conocer la tasa de t−1; el escenario conservador («historia rezagada») está en `resultados_v3.md`.
- **Sesgo de selección y supervivencia:** solo IES con denominador suficiente y sin posible cierre; el error es mayor en algunos subgrupos (ver `resultados_sesgo_sensibilidad.md`). El análisis de fuga cubre 32 explicativas, no todos los caminos posibles.
- **Datos que cambian:** las fuentes de datos.gov.co se actualizan; por eso se fijan SHA-256 y fecha de descarga. SPADIES y TerriData se descargan a mano.
- **Complejidad medida en una sola máquina** (CPU híbrida, un hilo): sirve para comparar componentes entre sí, no para predecir tiempos en otra; con 4 puntos los exponentes de escalado son aproximados.
- **Variables de mercado laboral** no disponibles a nivel municipal para la ventana.

## Licencia y citación
El **código** es MIT ([`LICENSE`](LICENSE)). Los **datos** provienen de sus fuentes oficiales (ICFES, MEN/SNIES, DANE, DNP/TerriData y SPADIES) y conservan sus propias condiciones; las cinco fuentes de datos.gov.co son CC BY-SA 4.0 y las de SPADIES y TerriData no se verificaron (ver [`docs/DATOS.md`](docs/DATOS.md)). Para citar: [`CITATION.cff`](CITATION.cff).

---

## Detalles del ETL y del modelado
### Regenerar `data/raw/` (no se versiona; detalle en `docs/DATOS.md`)
- **Automático** (`pipeline.py` sin `--saltar-extract`): Saber 11 (`kgxf-xxbe`, ~3,4 M filas, ~25 min), matrícula SNIES y DIVIPOLA. datos.gov.co falla de forma intermitente (500/503); la descarga reintenta sola.
- **Manual**:
  - `data/raw/spadies/spadies_diferenciado.csv` — SPADIES → Analíticas → Deserción Anual → tipo de cálculo IES → IES: Diferenciado (variable objetivo; todas las IES en un archivo) y `spadies_nacional.csv` (total nacional, solo contexto y control). Formato y cálculo de la tasa en `reports/etl_log.md`. Sin el diferenciado el pipeline genera un dataset **preliminar** con `tasa_desercion` nula.
  - `data/raw/terridata/TerriData_Dim{2,4,7,8,14,16,25}.txt.zip` — descargas de TerriData (DNP) por dimensión: población, educación, finanzas públicas, desempeño municipal, pobreza y mercado laboral. Fuente de población y NBI/IPM (ya no se usan archivos del DANE). La lectura tarda ~1 min (la dimensión 2 pesa ~1 GB) y se guarda en `data/interim/terridata_largo.csv`. Si faltan, la fuente se omite y se anota en `reports/etl_log.md`.

### Modelado
Utilidades en `src/modelado/` (parámetros en `config/config.yaml`, secciones `modelado` y `particion`):
- **Partición** (`particion.py`): `particion_group_kfold` (por `cod_ies`) y `particion_temporal` (entrena 2015–2019, prueba 2020–2021). `cargar_dataset(cfg, excluir_sena=..., ventana_modelado=...)` aplica las sensibilidades: `modelado.ventana_modelado` (`[2015, 2021]` por defecto; `[2017, 2021]` evita los años sin MDM ni SNIES) y `modelado.excluir_sena` (quita al SENA al cargar).
- **Variables**: `variables_modelado(df)` devuelve las **numéricas y las categóricas por separado** (`sector` y `caracter`, con etiquetas según SNIES) y `construir_preprocesador` arma el `ColumnTransformer` (imputación, estandarización y one-hot). El tamaño por defecto es `log_matriculados_spadies_t2` (denominador de SPADIES t-2, completo 2015–2021; información de t-2, sin fuga); `log_matriculados_ies_t2` (SNIES) es la alternativa. Quedan fuera por defecto `td_poblacion_total_t1` (cambio de base de las proyecciones 2005 → 2018), `ratio_spadies_snies` y la matrícula bruta. Los indicadores `falta_saber11`, `falta_mdm` y `falta_snies` marcan los bloques de años sin dato (MDM y SNIES: 2015–2016; Saber 11: 2019 y 2021) y entran como numéricas.
- **Objetivo** (`objetivo.py`): transformación logit opcional (`modelado.transformacion_objetivo`).
- **Experimentos** (`experimentos.py`, `informe_experimentos.py`; `python run_experimentos.py`): grilla de 36 configuraciones (objetivo ninguna/logit × imputación mediana/KNN × selección ninguna/información mutua/PCA × Ridge/Random Forest/XGBoost, configurable en `modelado.experimentos`) con GroupKFold por IES de 5 pliegues idénticos para todas, más las líneas base mediana e ingenua (tasa del año anterior de la IES). Escribe `reports/resultados/experimentos_v1.csv` (por pliegue) y `experimentos_v1_resumen.csv`, `reports/resultados_v1.md` y `reports/figuras/mae_configuraciones.png`.
- **Formulaciones con historia** (`formulaciones.py`, `informe_formulaciones.py`; `python run_formulaciones.py`): A contexto, B historia + contexto, B solo historia y C corrección sobre la ingenua (+ C solo historia), con los mismos pliegues que la v1; reporta el MAE total y por filas con/sin historia. Las columnas de historia (`tasa_t1`, `tasa_t2`, `tasa_media_previa`, `n_anios_previos`, `falta_historia`, solo años anteriores a t) las agrega el ETL (`incluir_historia`) y no son explicativas por defecto. Salidas: `reports/resultados/experimentos_v2.csv`, `reports/resultados_v2.md`, `reports/figuras/mae_formulaciones.png`.
- **Protocolo v3** (`protocolo.py`, `informe_v3.py`; `python run_protocolo_v3.py`): evaluación rigurosa de un conjunto de 15 configuraciones fijado antes de correr, con GroupKFold por IES repetido 5 veces (25 ajustes), IC 95 % con la corrección de Nadeau y Bengio, prueba de Wilcoxon pareada contra la ingenua y MAE por filas con/sin historia. Incluye la variante «historia rezagada» (sin la tasa de t-1) y su ingenua rezagada. La historia del dataset usa las tasas anuales de SPADIES desde 1999 (`historia_previa_a_ventana`); las filas sin historia (34) se listan en `reports/sin_historia.csv`.
- **Tarea 2: estimación sin historia** (`metodos.py`, `tarea2.py`, `informe_tarea2.py`; `python run_tarea2.py`; notebook `notebooks/04_tarea2.ipynb`): predice la tasa solo con el contexto (formulación A) con GroupKFold por IES. Líneas base: mediana y mediana por carácter × sector. Métodos del profesor como transformers de scikit-learn (KernelPCA RBF, autoencoder n→16→8→16→n con parada temprana, clusters K-means con k por silueta que agregan el cluster y las distancias a los centroides). Protocolo en dos etapas: cribado de 120 configuraciones (una semilla) y confirmación 5 × 5 con semillas distintas, IC 95 % y t corregida. Salidas: `reports/resultados_tarea2.md`, `reports/resultados/experimentos_tarea2_*.csv`, `reports/figuras/tarea2_metodologias.png` y `clusters.png`.
- **Ajuste, particiones temporales y arranque en frío** (`ajuste.py`, `temporal.py`, `arranque_frio.py`, `informe_ajuste.py`; `python run_ajuste_temporal.py`; notebook `notebooks/05_ajuste_temporal.ipynb`): ajuste de hiperparámetros con validación anidada (externa GroupKFold por IES 5 × 5, interna RandomizedSearchCV de 30 iteraciones), particiones temporales (prepandemia y estrés, con bootstrap por IES) y arranque en frío estricto agrupando por municipio. Salidas: `reports/resultados_ajuste_temporal.md`, `reports/resultados/ajuste.csv`, `temporal.csv`, `arranque_frio_municipio.csv`, `reports/figuras/ajuste_vs_defecto.png` y `temporal_por_anio.png`.
- **Sesgo medido, sensibilidad e interpretabilidad** (`evaluacion_final.py`, `subgrupos.py`, `sensibilidad.py`, `diagnostico_colapso.py`, `interpretabilidad.py`; `python run_sesgo_sensibilidad.py`; notebook `notebooks/06_sesgo_interpretabilidad.ipynb`): error por subgrupo de los modelos finales frente a su línea base (IC por bootstrap sobre IES), cuatro escenarios de sensibilidad (sin Bogotá, denominador mínimo 200, sin SENA, 2017–2021), diagnóstico reproducible del colapso del ridge ajustado de la Tarea 2, y SHAP, importancia por permutación y estabilidad de coeficientes. Salidas: `reports/resultados_sesgo_sensibilidad.md`, `reports/resultados/{subgrupos,sensibilidad,colapso_ridge,shap_top}.csv` y `reports/figuras/{error_subgrupos,shap_t1,shap_t2,coeficientes_ridge}.png`.
- **Complejidad medida** (`complejidad.py`, `costo_protocolo.py`, `informe_complejidad.py`; `python run_complejidad.py`; notebook `notebooks/07_complejidad.ipynb`): costo de ajuste, predicción y memoria de cada modelo y método con un solo hilo (calentamiento, 5 repeticiones, mediana y RIQ), escalado empírico con submuestras por IES y exponente log-log frente al teórico, y costo del protocolo de evaluación. Salidas: `reports/resultados_complejidad.md`, `reports/resultados/complejidad*.csv` y `reports/figuras/complejidad_{costo_error,escalado}.png`.
- **Sesgo** (`sesgo.py`, `geografia.py`, `mitigacion.py`): análisis de sesgo del dataset y opciones de mitigación, **todas apagadas por defecto** (el dataset y los valores por defecto no cambian): `modelado.pesos_muestrales` (`ninguno` | `matriculados` | `raiz_matriculados` | `inverso_grupo` | `inverso_ies`), `modelado.umbral_denominador` (p. ej. 200), `modelado.indicador_panel_incompleto`, `particion.agrupar_por` (`departamento`) y `particion.estratificar_por` (`[sector, caracter]`); además `particion_dejando_fuera` (prueba en Bogotá) y `metricas_por_subgrupo` (MAE por sector, carácter, región, ciudad y tamaño). Informe en `reports/analisis_sesgo.md` y figuras en `notebooks/02_sesgo.ipynb` (`python notebooks/construir_sesgo.py`).

Filtros del ETL (todos configurables en `config.yaml`): `min_matriculados_t2` (por defecto 100), `excluir_posible_cierre`, `excluir_inconsistencias` (ratio SPADIES/SNIES fuera de 1/3–3) y `excluir_regimen_especial` (lista explícita de 16 escuelas militares y policiales), todos `true`; `es_sena` marca al SENA (9110) y `excluir_sena` (`false`) lo saca. El detalle está en `reports/exclusiones_calidad.csv` y las tasas > 50 % en `reports/tasas_altas.csv`. El dataset no incluye variables de mercado laboral (ver `reports/etl_log.md`).

### Salidas
`data/processed/dataset_<unidad>.csv` · `reports/data_dictionary.md` · `reports/etl_log.md` (la parte posterior a `<!-- pipeline:inicio -->` se regenera en cada corrida; lo anterior es manual) · `notebooks/01_eda.ipynb` · `notebooks/02_sesgo.ipynb` · `reports/analisis_sesgo.md` · `reports/diagrama_pipeline.md` · `reports/resultados_v1.md` · `reports/resultados_v2.md` · `reports/resultados_v3.md` · `reports/resultados_tarea2.md` · `reports/resultados_ajuste_temporal.md` · `reports/resultados_sesgo_sensibilidad.md` · `reports/resultados_complejidad.md` · `reports/sin_historia.csv` · `reports/resultados/` · `reports/figuras/`.
