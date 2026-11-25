# Deserción en educación superior en Colombia: ETL y modelado (IES × año, 2015–2021)

[![CI](https://github.com/miguelmendez04/ProyectoAprendizajeAutomatico/actions/workflows/ci.yml/badge.svg)](https://github.com/miguelmendez04/ProyectoAprendizajeAutomatico/actions/workflows/ci.yml)
[![Licencia](https://img.shields.io/badge/licencia-MIT%20(c%C3%B3digo)%20%7C%20CC%20BY--SA%204.0%20(datos)-blue)](#licencia-y-citación)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.14-blue)](pyproject.toml)
[![Estilo: ruff](https://img.shields.io/badge/estilo-ruff-261230)](pyproject.toml)

> **English summary.** ETL and machine-learning experiments to estimate the dropout rate of Colombian higher-education institutions (IES × year, 2015–2021; 1,928 rows, 295 institutions) from SPADIES (target), SNIES, Saber 11 and TerriData. **Task 1** (with the institution's own history): does a model beat the naive forecast "this year's rate = last year's rate"? **Task 2** (no history, "cold start"): can the municipal and institutional context alone beat the median by character × sector? Evaluation: GroupKFold by institution (5 × 5), 95 % CIs with the Nadeau–Bengio correction, nested hyper-parameter tuning, temporal splits, bias analysis, and measured computational cost. **Main findings:** the best Task 1 model (ridge correcting the naive forecast) improves MAE by 4.4 % over the naive baseline (0.0245 vs 0.0256), small but significant under both tests; in Task 2 no model beats the median-by-group baseline with statistical firmness (best: ridge, +0.9 %, p = 0.60). **Use / reproduce:** `pip install -e ".[dev]"`, then `desercion predecir --tarea 1 ...` (models in `models/`, see the [model card](models/model_card.md)) or `desercion reproducir --etapa todo --rapido` (< 10 min, writes to `reproduccion/`, does not touch published results) or `--etapa todo` (≈ 1.5 h). Code: MIT; the derived dataset is CC BY-SA 4.0 with attribution to MEN (SPADIES/SNIES), ICFES, DNP (TerriData) and DANE ([`data/processed/LICENSE`](data/processed/LICENSE)); raw data keep their sources' terms ([`docs/DATOS.md`](docs/DATOS.md)).

Proyecto del curso Aprendizaje Automático (Maestría IA y Ciencia de Datos, UAO). Entrega: 26 de noviembre de 2026 (tag `v1.0-entrega`).
Especificación: `docs/guia_etl.md` · Bitácora de decisiones y calidad de datos: `reports/01_etl.md`.

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
MAE (menor es mejor) con validación anidada: GroupKFold por IES 5 × 5 = 25 ajustes; IC 95 % con la corrección de Nadeau y Bengio. Fuente: `reports/04_ajuste_temporal.md` (`reports/tables/ajuste_resumen.csv`). «p» es la prueba t corregida (y Wilcoxon) frente a la línea base.

| Tarea | Modelo | MAE [IC 95 %] | Mejora frente a la línea base | p (t corregida; Wilcoxon) |
|---|---|---|---|---|
| 1 | **Ingenua** (tasa de t−1) | 0,0256 [0,0225; 0,0288] | — | — |
| 1 | **Ridge de corrección, C solo historia** (ajustado) | 0,0245 [0,0212; 0,0278] | +4,4 % | 0,011; < 0,001 |
| 2 | **Mediana por carácter × sector** | 0,0400 [0,0345; 0,0455] | — | — |
| 2 | Mejor modelo: ridge, logit, información mutua, sin clusters (ajustado) | 0,0396 [0,0345; 0,0448] | +0,9 % | 0,596; 0,127 |

> **Nota.** Estas cifras vienen de la validación anidada (semillas externas 301–305). La confirmación de la Tarea 2 (semillas 201–205, sin ajuste anidado) da 0,0405 para la mediana por carácter × sector (`reports/03_tarea2.md`), y ambas conclusiones coinciden: el mejor modelo mejora 2,2 % en esa confirmación (p de Wilcoxon 0,019 pero p de la t corregida 0,35), sin firmeza estadística; en la validación anidada, +0,9 % (p t corregida 0,60).

**Lectura.** En la Tarea 1 el ridge de corrección supera a la ingenua con significancia en ambas pruebas, pero la ventaja es pequeña (0,001 de MAE) y en las particiones temporales (prepandemia y 2020–2021) ningún modelo se distingue de la ingenua. En la Tarea 2 el contexto no mejora a la mediana por carácter × sector con firmeza estadística; xgboost y random forest empeoran con hiperparámetros por defecto y quedan «sin diferencia firme» tras el ajuste. Costo: el ridge de la Tarea 1 se ajusta en ~9 ms; random forest, en ~1,3 s sin menos error (`reports/07_complejidad.md`).

## Uso del modelo
Los modelos finales se pueden usar sobre filas nuevas desde la línea de comandos (el detalle, las métricas con IC, los subgrupos peor servidos y las limitaciones están en la **[ficha del modelo](models/model_card.md)**; la arquitectura, en [`docs/ARQUITECTURA.md`](docs/ARQUITECTURA.md)). Quedan guardados en `models/` (joblib + JSON con fecha, commit, SHA-256 del dataset y versiones).
```
desercion entrenar                                    # reentrena con todo el dataset y reescribe models/
desercion predecir --tarea 1 --entrada historia.csv --salida predicciones.csv
desercion predecir --tarea 2 --entrada contexto.csv --salida predicciones.csv [--modelo ridge]
```
- **Tarea 1 (con historia):** exige la columna `tasa_t1` (tasa de deserción del año anterior, entre 0 y 1) y acepta `tasa_t2`, `tasa_media_previa`, `n_anios_previos` y `falta_historia`; sin `tasa_t1` el comando sugiere la Tarea 2.
- **Tarea 2 (sin historia):** exige el contexto de la IES (`sector`, `caracter` y las variables de Saber 11, TerriData y matrícula; las columnas de `data/processed/dataset_ies.csv` sirven de plantilla). Por defecto usa la mediana por carácter × sector (`tarea2_mediana`), porque el ridge no la supera con firmeza y colapsó en la partición prepandemia; `--modelo ridge` lo usa igual, con una advertencia (ver la ficha).
- La entrada se valida antes de predecir y los errores dicen qué falta (código de salida 2).
- **Columnas de la salida:** `modelo`, `tasa_predicha` y las identificadoras que haya en la entrada: `cod_ies`, `anio`, `semestre`, `cod_divipola_mpio`, `caso` e `id`. Para copiar otras (por ejemplo un nombre), use `--conservar nombre_ies,otra_columna`.

Ejemplo con cinco filas de 2021 del propio dataset (`historia.csv` con `cod_ies, anio, tasa_t1, tasa_t2, tasa_media_previa, n_anios_previos, falta_historia`):
```
$ desercion predecir --tarea 1 --entrada historia.csv --salida predicciones.csv
5 predicciones (T1 C solo historia | ridge (ajustado)) escritas en predicciones.csv
```
| cod_ies | tasa_t1 | tasa_predicha | tasa observada |
|---|---|---|---|
| 1101 | 0,0460 | 0,0448 | 0,0259 |
| 1102 | 0,1070 | 0,0925 | 0,0399 |
| 1103 | 0,0460 | 0,0453 | 0,0338 |
| 1104 | 0,0512 | 0,0515 | 0,0342 |
| 1105 | 0,0797 | 0,0725 | 0,0512 |

Son filas con las que se entrenó el modelo (no una evaluación) y muestran el tipo de resultado: la predicción es casi la tasa del año anterior con una pequeña corrección, y el error típico fuera de muestra es de unos 0,025 (ver arriba). Los modelos están guardados con pickle: no cargue archivos de modelo de origen desconocido (`predecir` verifica el SHA-256).

## Demo
Dos entradas de ejemplo en [`examples/`](examples/). Tras `pip install -e .`:
```
desercion predecir --tarea 1 --entrada examples/historia_2022.csv --salida predicciones_t1.csv
desercion predecir --tarea 2 --entrada examples/ies_nuevas.csv   --salida predicciones_t2.csv
```
- `historia_2022.csv` (Tarea 1): la historia de cinco IES para estimar 2022 (`cod_ies`, `tasa_t1` = tasa observada en 2021, `tasa_t2`, `tasa_media_previa`, `n_anios_previos`, `falta_historia`). El repositorio no tiene la tasa de 2022, así que estas predicciones no se pueden evaluar aquí.
- `ies_nuevas.csv` (Tarea 2, sin historia): cuatro casos descritos solo por `sector` y `caracter`, con una columna `caso` que la salida conserva. Usa la mediana por carácter × sector; con estas columnas **no** se puede usar `--modelo ridge`, que además exige el contexto completo (el comando lo dice y sale con código 2).

Salida de la Tarea 2:
```
caso,modelo,tasa_predicha
A,T2 mediana carácter × sector,0.20635759557547279
B,T2 mediana carácter × sector,0.0820695807314897
C,T2 mediana carácter × sector,0.13731280495697534
D,T2 mediana carácter × sector,0.0742733765155289
```

## Cómo reproducir
**Requisitos:** Python 3.14, las versiones fijas de `requirements.txt` y el dataset `data/processed/dataset_ies.csv` (ya está en el repositorio). Los datos crudos **no** son necesarios para el modelado; solo para la etapa `etl` (ver [`docs/DATOS.md`](docs/DATOS.md): URL, pasos de descarga, nombre y carpeta de cada archivo, fecha y checksums).

```
python -m venv .venv
.venv\Scriptsctivate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -e ".[dev]"           # instala el paquete (comando `desercion`) con las versiones fijas y las herramientas de desarrollo
desercion verificar                           # SHA-256 del dataset (y de los crudos, si están)
desercion reproducir --etapa todo --rapido    # verificación en < 10 min; escribe en reproduccion/
desercion reproducir --etapa todo             # corrida completa (≈ 1,5 h)
python -m pytest -q                           # 448 pruebas (≈ 2,5 min); `-m "not integration"` omite las que necesitan los datos publicados
```
Sin `pip install -e` (solo con `requirements.txt`, generado desde `pyproject.toml`): `pip install -r requirements.txt` y anteponer `PYTHONPATH=src` (PowerShell: `$env:PYTHONPATH = "src"`) a `python -m desercion_ies <comando>`, que es equivalente a `desercion <comando>`.

Opciones: `--etapa {etl,eda,tarea1,tarea2,ajuste,sesgo,complejidad,todo}`, `--rapido`, `--reanudar` (reutiliza lo ya calculado donde el script lo permite: ajuste anidado, sesgo y complejidad guardan puntos de control), `--sin-notebooks`, `--sin-verificar`. Al terminar imprime el tiempo y las cifras clave de cada etapa. Cada etapa llama a los scripts existentes (`src/desercion_ies/etl/pipeline.py`, `desercion evaluar v1`, `desercion evaluar v2`, `desercion evaluar tarea1`, `desercion evaluar tarea2`, `desercion evaluar ajuste`, `desercion evaluar sesgo`, `desercion evaluar complejidad`) con sus semillas, sin cambiar su lógica.

| Etapa | Qué corre | Completa (≈) | `--rapido` (≈) |
|---|---|---|---|
| `etl` | `desercion etl --saltar-extract` (necesita los 14 crudos) | 2 min | 2 min |
| `eda` | cuadernos 01 y 02 | 2 min | 2 min |
| `tarea1` | v1 (36 configuraciones), v2 (formulaciones), v3 (protocolo 5 × 5) | 31 min | 0,5 min |
| `tarea2` | cribado + confirmación + clusters | 15 min | 1,5 min |
| `ajuste` | anidado + temporales + arranque en frío | 31 min | 1,5 min |
| `sesgo` | subgrupos, sensibilidad, colapso, SHAP | 2 min | 1 min |
| `complejidad` | medición con un hilo | 3 min | 2 min |

`--etapa todo --rapido` tardó **585–633 s (≈ 10 min)** en dos corridas con el ETL y el EDA incluidos; desde un clon limpio, sin los crudos (`etl` y `eda` se omiten), **381 s (6,3 min)**. Los tiempos son de la máquina original (**Windows 11, Intel Core i5-12500H, 16 hilos, 31,6 GiB**) y **dependen de la máquina**: las etapas pesadas usan 16 procesos en paralelo. La corrida completa regenera además los cuadernos 04–07 (`--sin-notebooks` lo evita).

**Modo `--rapido`.** Solo cambia la *configuración*: menos semillas (se conservan las primeras de las publicadas), grillas más chicas, `n_iter` = 3 en la búsqueda interna, 100 réplicas de bootstrap y 2 repeticiones en la medición. Escribe todo en `reproduccion/` (no se versiona) y no toca `reports/`. Al final **compara contra los CSV publicados**: las cifras por pliegue de las configuraciones comunes coinciden **exactamente** en el entorno original y hasta ~1e-13 con otra biblioteca de álgebra lineal (v1, v3, Tarea 2, ajuste por defecto, arranque en frío), salvo dos diferencias esperadas y documentadas en [`docs/REPRODUCIBILIDAD.md`](docs/REPRODUCIBILIDAD.md) (la v2 con historia y el ajuste con `n_iter` reducido). Los tiempos medidos (complejidad) nunca coinciden: dependen de la máquina.

## Mapa de informes y cuadernos
Los informes (`reports/`) se numeran en orden de lectura; los cuadernos (`notebooks/`), de 01 a 06 sin saltos.
| Informe | Cuaderno | Pregunta que responde |
|---|---|---|
| `reports/01_etl.md` (bitácora del ETL; anexos en `reports/etl/`) | `notebooks/01_eda.ipynb` | ¿Cómo se construyó el dataset, qué calidad tiene y qué se excluyó? (`notebooks/conclusiones_eda.md`) |
| `reports/05_sesgo_dataset.md` | `notebooks/02_sesgo.ipynb` | ¿Hay sesgo de selección, supervivencia, geografía o fuga temporal en el dataset? |
| `reports/02_tarea1.md` (v3, referencia vigente; v1 y v2 en `reports/historico/`) | — | Tarea 1: ¿qué formulaciones y configuraciones funcionan, con qué protocolo? |
| `reports/03_tarea2.md` | `notebooks/03_tarea2.ipynb` | Tarea 2: sin historia, ¿ayudan KernelPCA, el autoencoder o los clusters? |
| `reports/04_ajuste_temporal.md` | `notebooks/04_ajuste_temporal.ipynb` | ¿El ajuste de hiperparámetros, la pandemia o el arranque en frío cambian las conclusiones? |
| `reports/06_sesgo_modelos.md` | `notebooks/05_sesgo_interpretabilidad.ipynb` | ¿Dónde falla más el modelo, qué tan robusto es y qué variables usa? |
| `reports/07_complejidad.md` | `notebooks/06_complejidad.ipynb` | ¿Cuánto cuesta cada modelo y cómo escala con más filas? |

La Tarea 1 y la v1–v2 no tienen cuaderno propio (se documentan en sus informes). Todos los cuadernos se ejecutan de punta a punta sin errores (`jupyter nbconvert --execute`); ver [`docs/REPRODUCIBILIDAD.md`](docs/REPRODUCIBILIDAD.md). Los nombres anteriores de rutas, módulos y comandos están en [`docs/MIGRACION.md`](docs/MIGRACION.md).

## Estructura del repositorio
Estructura tipo Cookiecutter Data Science con *src layout*; `pip install -e .` instala el paquete `desercion_ies` y el comando `desercion`.
```
pyproject.toml           paquete, dependencias con versiones fijas, pytest y ruff   ·   requirements.txt: generado desde pyproject
Makefile                 atajos (install, test, lint, reproduce, notebooks…)         ·   tools/: generador de requirements.txt
configs/config.yaml      toda la configuración (fuentes, filtros, grillas, semillas, salidas)
src/desercion_ies/
  cli.py, __main__.py    comando `desercion` (etl, eda, evaluar, reproducir, verificar)
  paths.py, config.py    rutas del proyecto (raíz detectada sin depender del directorio actual) y lectura de la configuración
  reproduce.py, reproducibilidad.py   reproducción por etapas, checksums, modo rápido y comparación con lo publicado
  inference/             entrenar, predecir y el objeto del modelo guardado (desercion entrenar / predecir)
  etl/                   extract/, transform/, homologation/, integrate/, validate/ y pipeline.py
  features/              particion, objetivo
  models/                metodos (KernelPCA, autoencoder, clusters, selección)
  evaluation/            protocolo, experimentos, formulaciones, tarea2, ajuste, temporal, arranque_frio, evaluacion_final
  analysis/              sesgo, subgrupos, sensibilidad, interpretabilidad, diagnostico_colapso, complejidad, costo_protocolo…
  reporting/             generadores de los informes (informe_*)
  runners/               lógica de cada `desercion evaluar <etapa>`
models/                  modelos finales (joblib + JSON de metadatos) y model_card.md
data/raw/                crudos (no versionados; ver docs/DATOS.md)
data/processed/          dataset_ies.csv (versionado, CC BY-SA 4.0)   ·   data/checksums.json (SHA-256)
reports/                 01_etl.md … 07_complejidad.md, etl/ (anexos del ETL), historico/ (v1 y v2), tables/ (CSV), figures/
notebooks/               01 a 06 (+ construir/: los scripts que los generan y ejecutan)
tests/unit/              pruebas unitarias   ·   tests/integration/: necesitan data/processed, data/interim o los resultados publicados
docs/                    guia_etl.md, DATOS.md, REPRODUCIBILIDAD.md, MIGRACION.md, ARQUITECTURA.md
.github/workflows/ci.yml integración continua (ruff, pytest sin integración, desercion verificar)
.pre-commit-config.yaml  ruff, ruff-format y revisiones de archivos
```

### Desarrollo
`pip install -e ".[dev]"` y `pre-commit install` activan los hooks (ruff, ruff-format, espacios y final de archivo, archivos grandes). `make lint`, `make format` y `make test-unit` (o sus equivalentes de abajo) corren lo mismo que la integración continua de GitHub Actions (Python 3.12 y 3.14, `ruff check`, `pytest -m "not integration"` y `desercion verificar`). Los cuadernos se versionan con sus salidas, por eso no se usa `nbstripout`.

### Equivalentes en PowerShell
`make` no viene con Windows; cada objetivo del `Makefile` equivale a:
| `make …` | PowerShell |
|---|---|
| `install` / `install-dev` | `pip install -e .` / `pip install -e ".[dev]"` |
| `test` / `test-unit` | `python -m pytest -q` / `python -m pytest -q -m "not integration"` |
| `lint` / `format` | `python -m ruff check src tests tools notebooks/construir` / `python -m ruff format src tests tools notebooks/construir` |
| `etl` | `desercion etl --saltar-extract` |
| `reproduce` / `reproduce-fast` | `desercion reproducir --etapa todo` / `desercion reproducir --etapa todo --rapido` |
| `notebooks` | `Get-ChildItem notebooks/construir/construir_*.py \| ForEach-Object { python $_.FullName }` |
| `requirements` | `python tools/generar_requirements.py` |
| `clean` | `Remove-Item -Recurse -Force reproduccion, .pytest_cache, .ruff_cache -ErrorAction SilentlyContinue` |

## Limitaciones principales
- **Tamaño y alcance:** 1.928 filas de 295 IES; la unidad es la institución, no el estudiante. Las conclusiones no se trasladan a programas ni a personas.
- **Ventaja pequeña de los modelos:** en la Tarea 1 es de ~4 %, y ningún modelo supera a la ingenua con IC que excluya 0 en las particiones temporales; en la Tarea 2 no hay mejora firme sobre la mediana por carácter × sector.
- **Disponibilidad de la historia:** la Tarea 1 supone conocer la tasa de t−1; el escenario conservador («historia rezagada») está en `02_tarea1.md`.
- **Sesgo de selección y supervivencia:** solo IES con denominador suficiente y sin posible cierre; el error es mayor en algunos subgrupos (ver `06_sesgo_modelos.md`). El análisis de fuga cubre 32 explicativas, no todos los caminos posibles.
- **Datos que cambian:** las fuentes de datos.gov.co se actualizan; por eso se fijan SHA-256 y fecha de descarga. SPADIES y TerriData se descargan a mano.
- **Complejidad medida en una sola máquina** (CPU híbrida, un hilo): sirve para comparar componentes entre sí, no para predecir tiempos en otra; con 4 puntos los exponentes de escalado son aproximados.
- **Variables de mercado laboral** no disponibles a nivel municipal para la ventana.

## Licencia y citación
- **Código:** MIT ([`LICENSE`](LICENSE)).
- **Dataset derivado** `data/processed/dataset_ies.csv`: obra derivada de fuentes de datos.gov.co, distribuida bajo **CC BY-SA 4.0** con atribución a MEN (SPADIES y SNIES), ICFES, DNP (TerriData) y DANE ([`data/processed/LICENSE`](data/processed/LICENSE)). La obligación de «compartir igual» viene de las cinco fuentes de datos.gov.co, cuya licencia se leyó de la API del portal; las condiciones de SPADIES y TerriData no se pudieron verificar automáticamente ([`docs/DATOS.md`](docs/DATOS.md)).
- **Datos crudos** (`data/raw/`, no versionados): conservan las condiciones de sus fuentes oficiales.
- Para citar: [`CITATION.cff`](CITATION.cff).

---

## Detalles del ETL y del modelado
### Regenerar `data/raw/` (no se versiona; detalle en `docs/DATOS.md`)
- **Automático** (`desercion etl` sin `--saltar-extract`): Saber 11 (`kgxf-xxbe`, ~3,4 M filas, ~25 min), matrícula SNIES y DIVIPOLA. datos.gov.co falla de forma intermitente (500/503); la descarga reintenta sola.
- **Manual**:
  - `data/raw/spadies/spadies_diferenciado.csv` — SPADIES → Analíticas → Deserción Anual → tipo de cálculo IES → IES: Diferenciado (variable objetivo; todas las IES en un archivo) y `spadies_nacional.csv` (total nacional, solo contexto y control). Formato y cálculo de la tasa en `reports/01_etl.md`. Sin el diferenciado el pipeline genera un dataset **preliminar** con `tasa_desercion` nula.
  - `data/raw/terridata/TerriData_Dim{2,4,7,8,14,16,25}.txt.zip` — descargas de TerriData (DNP) por dimensión: población, educación, finanzas públicas, desempeño municipal, pobreza y mercado laboral. Fuente de población y NBI/IPM (ya no se usan archivos del DANE). La lectura tarda ~1 min (la dimensión 2 pesa ~1 GB) y se guarda en `data/interim/terridata_largo.csv`. Si faltan, la fuente se omite y se anota en `reports/01_etl.md`.

### Modelado
Utilidades en `src/desercion_ies/` (subpaquetes `features`, `models`, `evaluation`, `analysis` y `reporting`) (parámetros en `configs/config.yaml`, secciones `modelado` y `particion`):
- **Partición** (`particion.py`): `particion_group_kfold` (por `cod_ies`) y `particion_temporal` (entrena 2015–2019, prueba 2020–2021). `cargar_dataset(cfg, excluir_sena=..., ventana_modelado=...)` aplica las sensibilidades: `modelado.ventana_modelado` (`[2015, 2021]` por defecto; `[2017, 2021]` evita los años sin MDM ni SNIES) y `modelado.excluir_sena` (quita al SENA al cargar).
- **Variables**: `variables_modelado(df)` devuelve las **numéricas y las categóricas por separado** (`sector` y `caracter`, con etiquetas según SNIES) y `construir_preprocesador` arma el `ColumnTransformer` (imputación, estandarización y one-hot). El tamaño por defecto es `log_matriculados_spadies_t2` (denominador de SPADIES t-2, completo 2015–2021; información de t-2, sin fuga); `log_matriculados_ies_t2` (SNIES) es la alternativa. Quedan fuera por defecto `td_poblacion_total_t1` (cambio de base de las proyecciones 2005 → 2018), `ratio_spadies_snies` y la matrícula bruta. Los indicadores `falta_saber11`, `falta_mdm` y `falta_snies` marcan los bloques de años sin dato (MDM y SNIES: 2015–2016; Saber 11: 2019 y 2021) y entran como numéricas.
- **Objetivo** (`objetivo.py`): transformación logit opcional (`modelado.transformacion_objetivo`).
- **Experimentos** (`experimentos.py`, `informe_experimentos.py`; `desercion evaluar v1`): grilla de 36 configuraciones (objetivo ninguna/logit × imputación mediana/KNN × selección ninguna/información mutua/PCA × Ridge/Random Forest/XGBoost, configurable en `modelado.experimentos`) con GroupKFold por IES de 5 pliegues idénticos para todas, más las líneas base mediana e ingenua (tasa del año anterior de la IES). Escribe `reports/tables/experimentos_v1.csv` (por pliegue) y `experimentos_v1_resumen.csv`, `reports/historico/resultados_v1.md` y `reports/figures/mae_configuraciones.png`.
- **Formulaciones con historia** (`formulaciones.py`, `informe_formulaciones.py`; `desercion evaluar v2`): A contexto, B historia + contexto, B solo historia y C corrección sobre la ingenua (+ C solo historia), con los mismos pliegues que la v1; reporta el MAE total y por filas con/sin historia. Las columnas de historia (`tasa_t1`, `tasa_t2`, `tasa_media_previa`, `n_anios_previos`, `falta_historia`, solo años anteriores a t) las agrega el ETL (`incluir_historia`) y no son explicativas por defecto. Salidas: `reports/tables/experimentos_v2.csv`, `reports/historico/resultados_v2.md`, `reports/figures/mae_formulaciones.png`.
- **Protocolo v3** (`protocolo.py`, `informe_v3.py`; `desercion evaluar tarea1`): evaluación rigurosa de un conjunto de 15 configuraciones fijado antes de correr, con GroupKFold por IES repetido 5 veces (25 ajustes), IC 95 % con la corrección de Nadeau y Bengio, prueba de Wilcoxon pareada contra la ingenua y MAE por filas con/sin historia. Incluye la variante «historia rezagada» (sin la tasa de t-1) y su ingenua rezagada. La historia del dataset usa las tasas anuales de SPADIES desde 1999 (`historia_previa_a_ventana`); las filas sin historia (34) se listan en `reports/etl/sin_historia.csv`.
- **Tarea 2: estimación sin historia** (`metodos.py`, `tarea2.py`, `informe_tarea2.py`; `desercion evaluar tarea2`; notebook `notebooks/03_tarea2.ipynb`): predice la tasa solo con el contexto (formulación A) con GroupKFold por IES. Líneas base: mediana y mediana por carácter × sector. Métodos del profesor como transformers de scikit-learn (KernelPCA RBF, autoencoder n→16→8→16→n con parada temprana, clusters K-means con k por silueta que agregan el cluster y las distancias a los centroides). Protocolo en dos etapas: cribado de 120 configuraciones (una semilla) y confirmación 5 × 5 con semillas distintas, IC 95 % y t corregida. Salidas: `reports/03_tarea2.md`, `reports/tables/experimentos_tarea2_*.csv`, `reports/figures/tarea2_metodologias.png` y `clusters.png`.
- **Ajuste, particiones temporales y arranque en frío** (`ajuste.py`, `temporal.py`, `arranque_frio.py`, `informe_ajuste.py`; `desercion evaluar ajuste`; notebook `notebooks/04_ajuste_temporal.ipynb`): ajuste de hiperparámetros con validación anidada (externa GroupKFold por IES 5 × 5, interna RandomizedSearchCV de 30 iteraciones), particiones temporales (prepandemia y estrés, con bootstrap por IES) y arranque en frío estricto agrupando por municipio. Salidas: `reports/04_ajuste_temporal.md`, `reports/tables/ajuste.csv`, `temporal.csv`, `arranque_frio_municipio.csv`, `reports/figures/ajuste_vs_defecto.png` y `temporal_por_anio.png`.
- **Sesgo medido, sensibilidad e interpretabilidad** (`evaluacion_final.py`, `subgrupos.py`, `sensibilidad.py`, `diagnostico_colapso.py`, `interpretabilidad.py`; `desercion evaluar sesgo`; notebook `notebooks/05_sesgo_interpretabilidad.ipynb`): error por subgrupo de los modelos finales frente a su línea base (IC por bootstrap sobre IES), cuatro escenarios de sensibilidad (sin Bogotá, denominador mínimo 200, sin SENA, 2017–2021), diagnóstico reproducible del colapso del ridge ajustado de la Tarea 2, y SHAP, importancia por permutación y estabilidad de coeficientes. Salidas: `reports/06_sesgo_modelos.md`, `reports/tables/{subgrupos,sensibilidad,colapso_ridge,shap_top}.csv` y `reports/figures/{error_subgrupos,shap_t1,shap_t2,coeficientes_ridge}.png`.
- **Complejidad medida** (`complejidad.py`, `costo_protocolo.py`, `informe_complejidad.py`; `desercion evaluar complejidad`; notebook `notebooks/06_complejidad.ipynb`): costo de ajuste, predicción y memoria de cada modelo y método con un solo hilo (calentamiento, 5 repeticiones, mediana y RIQ), escalado empírico con submuestras por IES y exponente log-log frente al teórico, y costo del protocolo de evaluación. Salidas: `reports/07_complejidad.md`, `reports/tables/complejidad*.csv` y `reports/figures/complejidad_{costo_error,escalado}.png`.
- **Sesgo** (`sesgo.py`, `geografia.py`, `mitigacion.py`): análisis de sesgo del dataset y opciones de mitigación, **todas apagadas por defecto** (el dataset y los valores por defecto no cambian): `modelado.pesos_muestrales` (`ninguno` | `matriculados` | `raiz_matriculados` | `inverso_grupo` | `inverso_ies`), `modelado.umbral_denominador` (p. ej. 200), `modelado.indicador_panel_incompleto`, `particion.agrupar_por` (`departamento`) y `particion.estratificar_por` (`[sector, caracter]`); además `particion_dejando_fuera` (prueba en Bogotá) y `metricas_por_subgrupo` (MAE por sector, carácter, región, ciudad y tamaño). Informe en `reports/05_sesgo_dataset.md` y figuras en `notebooks/02_sesgo.ipynb` (`python notebooks/construir/construir_sesgo.py`).

Filtros del ETL (todos configurables en `config.yaml`): `min_matriculados_t2` (por defecto 100), `excluir_posible_cierre`, `excluir_inconsistencias` (ratio SPADIES/SNIES fuera de 1/3–3) y `excluir_regimen_especial` (lista explícita de 16 escuelas militares y policiales), todos `true`; `es_sena` marca al SENA (9110) y `excluir_sena` (`false`) lo saca. El detalle está en `reports/etl/exclusiones_calidad.csv` y las tasas > 50 % en `reports/etl/tasas_altas.csv`. El dataset no incluye variables de mercado laboral (ver `reports/01_etl.md`).

### Salidas
`data/processed/dataset_<unidad>.csv` · `reports/etl/data_dictionary.md` · `reports/01_etl.md` (la parte posterior a `<!-- pipeline:inicio -->` se regenera en cada corrida; lo anterior es manual) · `notebooks/01_eda.ipynb` · `notebooks/02_sesgo.ipynb` · `reports/05_sesgo_dataset.md` · `reports/etl/diagrama_pipeline.md` · `reports/historico/resultados_v1.md` · `reports/historico/resultados_v2.md` · `reports/02_tarea1.md` · `reports/03_tarea2.md` · `reports/04_ajuste_temporal.md` · `reports/06_sesgo_modelos.md` · `reports/07_complejidad.md` · `reports/etl/sin_historia.csv` · `reports/tables/` · `reports/figures/`.
