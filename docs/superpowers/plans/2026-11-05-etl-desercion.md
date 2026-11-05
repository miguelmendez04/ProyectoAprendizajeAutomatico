# Plan de implementación: ETL del dataset de deserción (Colombia)

> **Para agentes:** SUB-HABILIDAD REQUERIDA: usar superpowers:subagent-driven-development (recomendado) o superpowers:executing-plans para implementar este plan tarea por tarea. Los pasos usan casillas (`- [ ]`) para seguimiento.

**Objetivo:** Construir un ETL modular y configurable que integre Saber 11, SNIES, DANE/DIVIPOLA, TerriData y SPADIES (exportado a mano) en un dataset propio de riesgo de deserción por IES, programa o municipio × año.

**Arquitectura:** Un módulo por fuente en `extract/` y `transform/`, un módulo central de homologación DIVIPOLA, una capa `integrate/` que agrega a la unidad de análisis elegida en `config.yaml` y rezaga las variables explicativas (t-1), y una capa `validate/` con pandera. `pipeline.py` orquesta todo y escribe `reports/etl_log.md`.

**Tecnologías:** Python 3.11+, pandas, requests (Socrata), pandera, PyYAML, pytest, openpyxl, matplotlib/geopandas (solo EDA), difflib (biblioteca estándar) para fuzzy matching.

**Especificación:** `guia_etl_desercion.md` (raíz del proyecto)

## Restricciones globales

- Llave canónica de municipio: código DANE de 5 dígitos como `str`, con ceros a la izquierda (`str.zfill(5)`); columna `cod_divipola_mpio`. Llave temporal: `anio`.
- Unidad de análisis definida en `config.yaml: unidad_analisis: ies | programa | municipio`; ninguna fuente se amarra a una sola opción.
- Variables explicativas del año t-1 para predecir el objetivo del año t (sin fuga de datos).
- `data/raw/` nunca se versiona; sí los scripts que lo regeneran. Cada descarga escribe un `metadata.json` (fuente, URL, fecha, filas, hash).
- SPADIES: el código solo lee y valida archivos exportados a mano en `data/raw/spadies/`.
- Todo emparejamiento no exacto de municipios se registra en un reporte; cada join documenta las filas perdidas.
- Comentarios y docstrings en español (la rúbrica evalúa código comentado).
- Prohibidos los datasets de scikit-learn y el dataset UCI de Realinho et al.
- Los IDs de datos.gov.co de la guía (`m2nt-jw2h`, `w7bi-rwg5`, `86ha-bzcy`) no están verificados: se confirman en la Tarea 1 antes de usarlos.
- Los nombres de columnas de Saber 11 cambian entre periodos: mapeo explícito por periodo.
- Ventana temporal común: se fija en la Tarea 1 (ej. 2015–2023) y vive en `config.yaml: anios`.

## Puntos de revisión (casos no cubiertos por la guía)

- Código DIVIPOLA numérico sin ceros (`5001`, `int` o `float` `5001.0`) debe resolver a `"05001"`, no fallar ni duplicar llaves.
- Código con valores especiales (`0`, `99999`, "SIN INFORMACION", NaN) debe quedar como nulo y contarse, no unirse a un municipio.
- Municipio con nombre repetido en distintos departamentos (p. ej. "San Antonio") no debe emparejarse sin departamento.
- Saber 11 de un periodo con una columna faltante o renombrada debe fallar con un error claro que nombre periodo y columna, no producir columnas vacías en silencio.
- Un municipio-año sin estudiantes de Saber 11 debe quedar con nulos (no 0) y aparecer en el conteo de pérdidas del join.
- Archivo SPADIES con tasas en porcentaje (0–100) en lugar de proporción (0–1), o con coma decimal, debe rechazarse o normalizarse de forma explícita.
- Llave duplicada (misma IES-año repetida en un export) debe detener la validación.

---

### Tarea 1: Estructura, configuración y Fase 0

**Archivos:**
- Crear: `.gitignore`, `requirements.txt`, `README.md`, `config/config.yaml`, `src/__init__.py` (y `__init__.py` en `extract/ transform/ homologation/ integrate/ validate/`), `src/config.py`, `tests/test_config.py`, `reports/fase0.md`, `reports/etl_log.md`

**Interfaces:**
- Produce: `load_config(path: str | Path = "config/config.yaml") -> dict` con claves `anios: list[int]`, `unidad_analisis: str`, `rutas: {raw, interim, processed}`, `datasets: {saber11: {periodo: id}, snies_matricula_mpio: id, ...}`, `spadies: {archivo, columnas: {...}}`, `max_nulos_pct: float`. Lanza `ValueError` si `unidad_analisis` no está en `{ies, programa, municipio}`.

- [ ] **Paso 1: Test.** En `tests/test_config.py`: `test_carga_config_valida` (el config del repo carga y `unidad_analisis in {"ies","programa","municipio"}`) y `test_unidad_invalida_lanza_valueerror` (config temporal con `unidad_analisis: "colegio"` → `pytest.raises(ValueError)`).
- [ ] **Paso 2:** Ejecutar `pytest tests/test_config.py -v`. Esperado: FALLA (`ModuleNotFoundError: src.config`).
- [ ] **Paso 3:** Crear la estructura de carpetas de la guía §5, `git init`, `.gitignore` (`data/raw/`, `data/interim/`, `__pycache__/`, `.ipynb_checkpoints/`), `requirements.txt` (pandas, requests, pandera, pyyaml, pytest, openpyxl, matplotlib, geopandas, jupyter), `config.yaml` con valores provisionales, e implementar `load_config` con validación de `unidad_analisis`.
- [ ] **Paso 4:** Ejecutar `pytest tests/test_config.py -v`. Esperado: 2 PASAN.
- [ ] **Paso 5: Fase 0 (manual, registrar en `reports/fase0.md`).** Para cada punto escribir el hallazgo y la fecha: (a) indicadores, filtros y formatos de exportación de la consulta pública de SPADIES, con capturas; (b) si la cuenta institucional UAO da acceso ampliado; (c) IDs, años y columnas verificados en datos.gov.co para Saber 11, matrícula por municipio y catálogo de IES/programas; (d) DIVIPOLA vigente y su histórico de cambios descargados; (e) ventana temporal elegida. Actualizar `config.yaml` con los IDs y años verificados.
- [ ] **Paso 6: Commit** `git add . && git commit -m "chore: estructura, config y fase 0"`

### Tarea 2: Homologación DIVIPOLA

**Archivos:**
- Crear: `src/homologation/divipola.py`, `tests/test_divipola.py`

**Interfaces:**
- Produce:
  - `normalizar_texto(s: str) -> str` (mayúsculas, sin tildes, espacios colapsados y recortados)
  - `normalizar_codigo(codigo) -> str | None` (acepta `str`, `int`, `float`, NaN; devuelve 5 dígitos, o `None` para vacíos, `0`, `99999`, "SIN INFORMACION")
  - `cargar_divipola(path: Path) -> pd.DataFrame` (columnas `cod_divipola_mpio`, `cod_depto`, `municipio_norm`)
  - `aplicar_cambios_codigo(df: pd.DataFrame, col: str, cambios: dict[str, str]) -> pd.DataFrame`
  - `emparejar_por_nombre(df, divipola, col_municipio: str, col_depto: str, umbral: float = 0.9) -> tuple[pd.DataFrame, pd.DataFrame]` (df con `cod_divipola_mpio` y reporte de emparejamientos no exactos con columnas `nombre_original, nombre_emparejado, cod_divipola_mpio, score`)

- [ ] **Paso 1: Tests** en `tests/test_divipola.py` (incluyen los puntos de revisión):
  - `normalizar_codigo(5001) == "05001"`, `normalizar_codigo(5001.0) == "05001"`, `normalizar_codigo("5001") == "05001"`, `normalizar_codigo("11001") == "11001"`
  - `normalizar_codigo(0) is None`, `normalizar_codigo(99999) is None`, `normalizar_codigo("SIN INFORMACION") is None`, `normalizar_codigo(float("nan")) is None`
  - `normalizar_texto("  Bogotá,  D.C. ") == "BOGOTA, D.C."`
  - `aplicar_cambios_codigo` reemplaza un código viejo por el nuevo (municipio nuevo o con código cambiado)
  - `emparejar_por_nombre`: "Medellin" + depto "Antioquia" → `"05001"` y aparece en el reporte con `score < 1`; "SAN ANTONIO" sin depto coincidente no se empareja (queda `None`); un nombre exacto no aparece en el reporte.
- [ ] **Paso 2:** Ejecutar `pytest tests/test_divipola.py -v`. Esperado: FALLA (el módulo no existe).
- [ ] **Paso 3:** Implementar las cinco funciones. Fuzzy con `difflib.get_close_matches` restringido a municipios del mismo departamento (por nombre normalizado); primero exacto, fuzzy solo como último recurso.
- [ ] **Paso 4:** Ejecutar `pytest tests/test_divipola.py -v`. Esperado: todos PASAN.
- [ ] **Paso 5: Commit** `git add src/homologation tests/test_divipola.py && git commit -m "feat: homologación DIVIPOLA"`

### Tarea 3: Extracción genérica (Socrata + metadata)

**Archivos:**
- Crear: `src/extract/common.py`, `tests/test_extract_common.py`

**Interfaces:**
- Produce:
  - `write_metadata(path: Path, fuente: str, url: str, filas: int) -> Path` (escribe `metadata.json` junto al archivo con `fuente, url, fecha_descarga, filas, sha256`)
  - `download_socrata(dataset_id: str, dest: Path, columnas: list[str] | None = None, page_size: int = 50000, base_url: str = "https://www.datos.gov.co/resource") -> Path` (pagina con `$limit/$offset` hasta una página vacía, usa `$select` si hay `columnas`, escribe el CSV y la metadata)

- [ ] **Paso 1: Tests** con `requests` simulado (`monkeypatch` sobre `requests.get`): `test_pagina_hasta_vacio` (2 páginas + 1 vacía → CSV con todas las filas, 3 llamadas con `$offset` 0, page_size, 2·page_size), `test_select_columnas` (parámetro `$select` presente), `test_metadata_json` (existe con las filas correctas y un sha256 de 64 caracteres hex).
- [ ] **Paso 2:** Ejecutar `pytest tests/test_extract_common.py -v`. Esperado: FALLA.
- [ ] **Paso 3:** Implementar ambas funciones; escribir el CSV de forma incremental (sin acumular todo en memoria).
- [ ] **Paso 4:** Ejecutar `pytest tests/test_extract_common.py -v`. Esperado: PASAN.
- [ ] **Paso 5: Commit** `git add src/extract tests && git commit -m "feat: descarga Socrata con metadata"`

### Tarea 4: Saber 11 (un periodo primero, luego todos)

**Archivos:**
- Crear: `src/extract/saber11.py`, `src/transform/saber11.py`, `tests/test_saber11.py`, `tests/fixtures/saber11_2018_2.csv` (≈20 filas reales muestreadas del periodo verificado en la Fase 0)

**Interfaces:**
- Consume: `download_socrata`, `normalizar_codigo`, `load_config`.
- Produce:
  - `extract_saber11(periodo: str, cfg: dict) -> Path`
  - `COLUMN_MAP: dict[str, dict[str, str]]` (periodo → {columna original: canónica})
  - `estandarizar_saber11(df: pd.DataFrame, periodo: str) -> pd.DataFrame` con columnas canónicas `anio:int, cod_divipola_mpio:str, punt_global:float, estrato:Int64 (1–6), colegio_oficial:bool`. Lanza `KeyError` con un mensaje que incluye periodo y columna si falta una columna esperada.

- [ ] **Paso 1: Tests:** `test_estandariza_periodo_2018_2` (fixture → columnas canónicas, `cod_divipola_mpio` de 5 caracteres, `estrato` en 1–6 o nulo, `punt_global` float); `test_columna_faltante_error_claro` (quitar una columna → `KeyError` cuyo mensaje contiene el periodo y el nombre de la columna); `test_codigos_especiales_a_nulo` (fila con `0`/`99999` → `cod_divipola_mpio` nulo).
- [ ] **Paso 2:** Ejecutar `pytest tests/test_saber11.py -v`. Esperado: FALLA.
- [ ] **Paso 3:** Implementar. Completar `COLUMN_MAP` para el periodo de la fixture; leer en chunks (`chunksize`) solo las columnas necesarias. Estratos tipo "Estrato 1" → 1; "Sin Estrato" → nulo.
- [ ] **Paso 4:** Ejecutar `pytest tests/test_saber11.py -v`. Esperado: PASAN. Luego descargar ese único periodo real a `data/raw/saber11/`, confirmar el mapeo a ojo y registrarlo en `etl_log.md`.
- [ ] **Paso 5:** Agregar las entradas de `COLUMN_MAP` de los demás periodos de la ventana (una fixture mínima por cada esquema distinto, con su test de estandarización) y ejecutar `pytest tests/test_saber11.py -v`. Esperado: PASAN.
- [ ] **Paso 6: Commit** `git add src tests && git commit -m "feat: extract/transform Saber 11"`

### Tarea 5: SPADIES (lector de exports manuales)

**Archivos:**
- Crear: `src/extract/spadies.py`, `tests/test_spadies.py`, `tests/fixtures/spadies_ies.csv`

**Interfaces:**
- Consume: `load_config` (`cfg["spadies"]["columnas"]` mapea columnas del export a canónicas), `normalizar_codigo`, `write_metadata`.
- Produce: `leer_spadies(path: Path, cfg: dict) -> pd.DataFrame` con `cod_ies:str, anio:int, tasa_desercion:float` (proporción 0–1; más `cod_snies_programa` / `cod_divipola_mpio` si el export los trae). Lanza `ValueError` ante llave duplicada o tasas fuera de rango.

- [ ] **Paso 1: Tests** (la fixture se arma con el formato real observado en la Fase 0): `test_lee_y_normaliza` (coma decimal "12,5" → 0.125 si el export viene en porcentaje; configurable con `cfg["spadies"]["tasa_en_porcentaje"]`), `test_llave_duplicada_error` (`ValueError`), `test_tasa_fuera_de_rango_error`.
- [ ] **Paso 2:** Ejecutar `pytest tests/test_spadies.py -v`. Esperado: FALLA.
- [ ] **Paso 3:** Implementar `leer_spadies`; escribir el `metadata.json` del archivo leído con `write_metadata`. Si la Fase 0 mostró que la cuenta UAO da más acceso, anotarlo en `etl_log.md` y ajustar las columnas, no la interfaz.
- [ ] **Paso 4:** Ejecutar `pytest tests/test_spadies.py -v`. Esperado: PASAN.
- [ ] **Paso 5: Commit** `git add src tests && git commit -m "feat: lector SPADIES"`

### Tarea 6: SNIES, DANE y TerriData

**Archivos:**
- Crear: `src/extract/snies.py`, `src/extract/dane.py`, `src/transform/snies.py`, `src/transform/dane.py`, `tests/test_snies_dane.py`, fixtures mínimas en `tests/fixtures/`

**Interfaces:**
- Consume: `download_socrata`, `normalizar_codigo`, `emparejar_por_nombre`.
- Produce (todas devuelven DataFrames limpios con llave `cod_divipola_mpio` + `anio`, salvo el catálogo):
  - `transformar_snies_matricula(df: pd.DataFrame) -> pd.DataFrame` (`matriculados:int` por municipio × año)
  - `transformar_snies_catalogo(df: pd.DataFrame) -> pd.DataFrame` (`cod_ies`, `cod_snies_programa`, `cod_divipola_mpio`, `caracter`, `sector`)
  - `transformar_terridata(df: pd.DataFrame) -> pd.DataFrame` (indicadores en columnas, con prefijo `td_`)
  - `transformar_dane(df: pd.DataFrame) -> pd.DataFrame` (`poblacion`, `nbi_pct` por municipio)

- [ ] **Paso 1: Tests:** por función, con una fixture pequeña: llaves de 5 dígitos, `matriculados` no negativo, sin duplicados de (`cod_divipola_mpio`, `anio`); y una fila con solo nombre de municipio que se resuelve vía `emparejar_por_nombre`.
- [ ] **Paso 2:** Ejecutar `pytest tests/test_snies_dane.py -v`. Esperado: FALLA.
- [ ] **Paso 3:** Implementar los extractores (usan `download_socrata` o leen archivos descargados de TerriData/DANE a `data/raw/`, con `metadata.json`) y los transformadores.
- [ ] **Paso 4:** Ejecutar `pytest tests/test_snies_dane.py -v`. Esperado: PASAN.
- [ ] **Paso 5: Commit** `git add src tests && git commit -m "feat: SNIES, DANE y TerriData"`

### Tarea 7: Integración (agregación, rezago, joins con registro de pérdidas)

**Archivos:**
- Crear: `src/integrate/aggregate.py`, `src/integrate/join.py`, `src/integrate/build.py`, `tests/test_integrate.py`

**Interfaces:**
- Consume: salidas de las Tareas 4–6.
- Produce:
  - `agregar_saber11_municipio(df: pd.DataFrame) -> pd.DataFrame`: `cod_divipola_mpio, anio, punt_global_media, punt_global_mediana, pct_estrato_bajo` (estratos 1–2), `pct_colegio_oficial`, `n_estudiantes`
  - `rezagar_features(df: pd.DataFrame, llave: list[str], cols: list[str], rezago: int = 1) -> pd.DataFrame` (la fila de `anio=t` recibe los valores de `t-rezago`; renombra con sufijo `_t1`)
  - `unir_con_perdida(izq: pd.DataFrame, der: pd.DataFrame, llave: list[str], nombre: str, log: list[dict], how: str = "left") -> pd.DataFrame` (agrega al `log` `{join, filas_izq, filas_der, filas_resultado, filas_sin_match}`)
  - `construir_dataset(cfg: dict) -> tuple[pd.DataFrame, list[dict]]` (el DataFrame final y el log de joins; respeta `cfg["unidad_analisis"]`)

- [ ] **Paso 1: Tests:** `test_agregacion_saber11` (valores calculados a mano sobre 4 estudiantes); `test_rezago_sin_fuga` (la fila `anio=2020` contiene los valores de 2019 y la del primer año, 2015, tiene nulos); `test_municipio_sin_saber11_queda_nulo` (nulo, no 0, y `filas_sin_match == 1` en el log); `test_unidad_ies_vs_municipio` (mismas entradas con distinta `unidad_analisis` → llaves `["cod_ies","anio"]` vs `["cod_divipola_mpio","anio"]`, sin duplicados).
- [ ] **Paso 2:** Ejecutar `pytest tests/test_integrate.py -v`. Esperado: FALLA.
- [ ] **Paso 3:** Implementar. En `construir_dataset`: la tasa de SPADIES (año t) es el objetivo; todas las demás variables pasan por `rezagar_features`; para `ies`/`programa` las variables municipales se asignan por el municipio sede del catálogo SNIES.
- [ ] **Paso 4:** Ejecutar `pytest tests/test_integrate.py -v`. Esperado: PASAN.
- [ ] **Paso 5: Commit** `git add src tests && git commit -m "feat: integración con rezago y log de pérdidas"`

### Tarea 8: Validación y diccionario de datos

**Archivos:**
- Crear: `src/validate/schema.py`, `src/validate/dictionary.py`, `tests/test_validate.py`

**Interfaces:**
- Consume: DataFrame de `construir_dataset`, `cfg["max_nulos_pct"]`.
- Produce:
  - `construir_esquema(unidad: str, max_nulos_pct: float) -> pa.DataFrameSchema` (tasa en [0,1]; `punt_global_*` en rango válido de Saber 11 (0–500); `pct_*` en [0,1]; unicidad de la llave según la unidad; nulos ≤ `max_nulos_pct` por columna; `matriculados >= desertores` cuando ambas existan)
  - `validar(df: pd.DataFrame, unidad: str, max_nulos_pct: float) -> pd.DataFrame` (lanza `pandera.errors.SchemaErrors` con `lazy=True`)
  - `generar_diccionario(df: pd.DataFrame, fuentes: dict[str, str], descripciones: dict[str, str], destino: Path) -> Path` (markdown con variable, tipo, fuente, descripción, % nulos, rango)

- [ ] **Paso 1: Tests:** `test_dataset_valido_pasa`; `test_tasa_mayor_a_1_falla`; `test_llave_duplicada_falla`; `test_exceso_de_nulos_falla`; `test_diccionario_contiene_todas_las_columnas` (cada columna aparece una vez en el .md con su % de nulos).
- [ ] **Paso 2:** Ejecutar `pytest tests/test_validate.py -v`. Esperado: FALLA.
- [ ] **Paso 3:** Implementar con pandera (`Check.in_range`, `unique=` en el esquema, `Column(nullable=True)` + check de % de nulos).
- [ ] **Paso 4:** Ejecutar `pytest tests/test_validate.py -v`. Esperado: PASAN.
- [ ] **Paso 5: Commit** `git add src tests && git commit -m "feat: validación pandera y diccionario de datos"`

### Tarea 9: Orquestación y corrida preliminar

**Archivos:**
- Crear: `pipeline.py`, `tests/test_pipeline.py`
- Modificar: `README.md`, `reports/etl_log.md`

**Interfaces:**
- Consume: todas las anteriores.
- Produce: `main(config_path: str = "config/config.yaml", saltar_extract: bool = False) -> Path` (ejecuta extract → transform → integrate → validate, guarda `data/processed/dataset_<unidad>.csv` y `reports/data_dictionary.md`, vuelca el log de joins y los emparejamientos no exactos a `reports/etl_log.md`; devuelve la ruta del dataset). CLI: `python pipeline.py [--config RUTA] [--saltar-extract]`.

- [ ] **Paso 1: Test** `test_pipeline_end_to_end_con_fixtures` (config temporal apuntando a las fixtures, `saltar_extract=True`, para las tres `unidad_analisis`): el CSV existe, pasa `validar`, y `etl_log.md` contiene la tabla de joins.
- [ ] **Paso 2:** Ejecutar `pytest tests/test_pipeline.py -v`. Esperado: FALLA.
- [ ] **Paso 3:** Implementar `main` y el CLI con `argparse`.
- [ ] **Paso 4:** Ejecutar `pytest -v` (toda la suite). Esperado: todo PASA.
- [ ] **Paso 5:** Correr `python pipeline.py` con datos reales (mínimo 2–3 fuentes) y revisar `reports/etl_log.md`: anotar las filas perdidas por join y las decisiones. Completar `README.md` (cómo regenerar `data/raw/`, dónde colocar los exports de SPADIES).
- [ ] **Paso 6: Commit** `git add . && git commit -m "feat: pipeline y corrida preliminar"`

### Tarea 10: EDA comentado y diagrama

**Archivos:**
- Crear: `notebooks/01_eda.ipynb`, `reports/diagrama_pipeline.md` (Mermaid; exportable a imagen para la diapositiva)

**Interfaces:**
- Consume: `data/processed/dataset_<unidad>.csv`, `reports/data_dictionary.md`.

- [ ] **Paso 1:** Notebook con celdas comentadas en español: carga y descripción general; distribución de la variable objetivo; nulos por fuente; relación variables–objetivo (correlación y dispersión); mapa por departamento/municipio (coroplético con `geopandas`); declaración explícita de que el modelo predice riesgo institucional/territorial, no individual.
- [ ] **Paso 2:** Verificar: `jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb` termina sin error.
- [ ] **Paso 3:** Escribir el diagrama Mermaid del pipeline ETL + modelado (fuentes → extract → transform → DIVIPOLA → integrate → validate → dataset → preprocesamiento ×3 → modelos ×3+ → validación espacial/temporal).
- [ ] **Paso 4: Commit** `git add notebooks reports && git commit -m "docs: EDA y diagrama del pipeline"`
