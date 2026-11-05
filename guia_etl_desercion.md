# Guía ETL — Dataset propio de deserción en educación superior (Colombia)

> Documento de arranque para desarrollar el ETL en Claude Code.
> Proyecto: curso **Aprendizaje Automático** (Maestría IA y Ciencia de Datos, UAO), y a la vez avance para la tesis de maestría sobre deserción estudiantil.
> Fecha: noviembre 2026.

---

## 1. Contexto y requisitos

### Lo que exige el curso
- **Entrega 1 (propuesta, presentación de 15 min máx.):** introducción y motivación, revisión de literatura (tabla de artículos), solución propuesta con diagrama y **descripción + visualización del dataset**.
  - Rúbrica: dataset y EDA **40%** (con *código comentado*), literatura **30%**, relevancia/viabilidad 15%, comunicación 15%.
- **Entrega final (paper IEEE en inglés):** proceso de **obtención/generación del dataset**, tipo de problema de ML, **≥3 metodologías de preprocesamiento**, **≥3 modelos** con análisis de complejidad, y todos los archivos para reproducir el experimento.
- **No se permiten** datasets de scikit-learn. El profesor espera un **dataset propio construido con ETL**, no uno ya usado (se descartó el dataset UCI de Realinho et al.).
- Gestión: roles (Data Engineer, Model Developer, PM, Evaluation Lead, Reporting Lead), Trello (20% de la nota) y repositorio en GitHub.

### Estado actual
- Tema: **deserción en educación superior en Colombia** con datos abiertos (ICFES, DANE, MEN/SNIES, SPADIES).
- **El tema de tesis aún está en análisis.** No hay unidad de análisis definida. El ETL debe ser **modular y configurable** para no amarrarse a una sola opción.
- Acceso a SPADIES: **solo consulta pública**. Hay una cuenta institucional UAO, pero no se ha verificado si da acceso.

---

## 2. Hallazgos sobre las fuentes

| Fuente | Qué ofrece | Granularidad | Acceso | Rol en el dataset |
|---|---|---|---|---|
| **SPADIES (MEN)** | Tasa de deserción anual (desertores en t / matriculados en t-2), deserción por cohorte, graduación | Nacional, IES, programa (consulta web); departamento (informes) | Consulta pública; microdatos **no** públicos | **Variable objetivo** |
| **ICFES Saber 11** | Microdatos por estudiante y periodo (2000 → reciente), con puntajes y variables socioeconómicas | Estudiante → agregable a colegio/municipio | datos.gov.co, un dataset por periodo (ej. 2018-2: `m2nt-jw2h`, 2017-1: `w7bi-rwg5`) | Perfil académico/socioeconómico de entrada |
| **ICFES Saber Pro** | Resultados por estudiante, IES y programa | IES / programa | datos.gov.co / DataICFES | Calidad de la IES/programa (opcional) |
| **SNIES (MEN)** | Matrícula ES por departamento (`86ha-bzcy`) y por municipio; catálogo de IES y programas | IES, programa, municipio, departamento | datos.gov.co / portal SNIES | Características institucionales |
| **DANE / DIVIPOLA** | Códigos oficiales de departamentos y municipios, población, NBI | Municipio | DANE / datos.gov.co | **Llave de integración** y contexto |
| **TerriData (DNP)** | Indicadores territoriales (fiscales, sociales, educativos) | Municipio × año | terridata.dnp.gov.co | Contexto socioeconómico |

> ⚠️ Los IDs de datos.gov.co salen de una búsqueda rápida. **Verificarlos** antes de codificarlos. Los nombres de columnas de Saber 11 **cambian entre periodos** (es uno de los retos del ETL).

**Limitación clave para el paper:** no existen datos abiertos de deserción por estudiante ni por municipio en educación superior. El modelo predice **riesgo a nivel institucional/territorial**, no individual. Hay que decirlo explícitamente.

---

## 3. Unidades de análisis candidatas (decisión pendiente)

| Opción | Fila = | Objetivo | Pros | Contras |
|---|---|---|---|---|
| **A. IES × año** | Una institución en un año | TDA de SPADIES por IES | Fácil de obtener; ~300 IES × ~10 años | Pocas filas (~3.000) |
| **B. Programa × año** | Un programa en un año | TDA por programa | Más filas y más variabilidad | Exportar de SPADIES puede ser tedioso |
| **C. Municipio (sede) × año** | Municipio con IES | TDA de las IES agregada al municipio sede | Encaja con enfoque geoespacial/DIVIPOLA y Spatial K-Fold | Agregación con pérdida de información |

**Recomendación de diseño:** construir el ETL sobre la **llave común `cod_divipola_mpio` + `anio`**, más `cod_ies` / `cod_snies_programa` cuando aplique. La agregación final se elige por configuración (`config.yaml: unidad_analisis: ies | programa | municipio`).

---

## 4. Tareas previas (Fase 0, antes de programar)

1. **Explorar la consulta pública de SPADIES**: qué indicadores, filtros (IES, programa, departamento, año) y formatos de exportación (Excel/CSV) permite. Documentar con capturas para el paper.
2. **Probar la cuenta institucional UAO** en SPADIES. Si da acceso a consultas analíticas o exportación masiva, cambia el alcance.
3. Verificar en datos.gov.co los IDs, años disponibles y columnas de: Saber 11, matrícula ES por municipio, catálogo IES/programas.
4. Descargar DIVIPOLA vigente y el histórico de cambios de códigos municipales.
5. Definir la **ventana temporal** común a todas las fuentes (ej. 2015–2023).

---

## 5. Arquitectura del ETL

### Estructura sugerida del repositorio
```
desercion-etl/
├── config/
│   └── config.yaml            # años, unidad de análisis, rutas, IDs de datasets
├── data/
│   ├── raw/                   # descargas originales, inmutables (NO se versionan en git)
│   ├── interim/               # limpias y homologadas por fuente
│   └── processed/             # dataset final integrado
├── src/
│   ├── extract/               # un módulo por fuente (saber11.py, snies.py, spadies.py, dane.py)
│   ├── transform/             # limpieza y estandarización por fuente
│   ├── homologation/          # divipola.py: normalización de códigos/nombres de municipios
│   ├── integrate/             # joins y agregación a la unidad de análisis
│   └── validate/              # reglas de calidad (pandera)
├── notebooks/
│   ├── 01_eda.ipynb           # EDA comentado (soporte del 40% de la rúbrica)
├── reports/
│   ├── data_dictionary.md     # diccionario de datos generado
│   └── etl_log.md             # bitácora de decisiones (insumo para el paper)
├── pipeline.py                # orquesta extract → transform → integrate → validate
├── requirements.txt
└── README.md
```

### Fases

**Extract**
- datos.gov.co usa API Socrata: descargar por paginación (`$limit`/`$offset`) o el CSV completo. Saber 11 tiene archivos grandes (cientos de miles de filas por periodo): leer en *chunks* o seleccionar solo las columnas necesarias.
- SPADIES: los archivos exportados de la consulta pública se guardan manualmente en `data/raw/spadies/`. El script solo los **lee y valida**.
- Guardar con cada descarga un `metadata.json`: fuente, URL, fecha de descarga, filas, hash. Así queda trazabilidad para el paper.

**Transform (por fuente)**
- Estandarizar nombres de columnas entre periodos de Saber 11 (mapeo explícito en un diccionario).
- Tipos de datos, valores faltantes, códigos especiales ("sin información", 0, 99999).
- Normalizar textos (mayúsculas, tildes, espacios) en nombres de municipios e IES.

**Homologación DIVIPOLA** (módulo central, reutilizable en la tesis)
- Llave canónica: código DANE de 5 dígitos con ceros a la izquierda (`str.zfill(5)`).
- Resolver municipios nuevos o cambiados de código con la tabla de cambios.
- Cuando solo hay nombre, emparejar por nombre normalizado + departamento, con *fuzzy matching* como último recurso. **Registrar cada emparejamiento no exacto** en un reporte.

**Integrate**
- Agregar Saber 11 a municipio × año (medias, medianas, % estrato bajo, % colegio oficial, etc.).
- **Alineación temporal para evitar fuga de datos:** las variables explicativas deben ser **anteriores** al objetivo (ej. contexto del año t-1 para predecir la deserción del año t).
- Unir con SNIES, DANE y TerriData por la llave. Documentar cuántas filas se pierden en cada join.

**Validate**
- Reglas con `pandera`: rangos (tasas entre 0 y 1, puntajes en rango válido), unicidad de la llave, % máximo de nulos por columna, consistencia (matriculados ≥ desertores).
- Generar `reports/data_dictionary.md` automáticamente: variable, tipo, fuente, descripción, % nulos, rango.

---

## 6. Puente con el modelado (para el paper)

**Tres metodologías de preprocesamiento a comparar** (no solo aplicar):
1. **Valores faltantes:** eliminación vs. imputación simple (mediana) vs. KNN/iterativa.
2. **Escalado + codificación:** estándar + one-hot vs. robusto + target encoding.
3. **Selección/reducción de variables:** ninguna vs. información mutua vs. PCA.
- Si el objetivo se discretiza en clases de riesgo: balanceo (ninguno / `class_weight` / SMOTE **dentro** del pipeline de CV).

**Modelos sugeridos (≥3, con complejidad contrastante):** regresión lineal/logística (base), Random Forest, XGBoost, y SVM o MLP para contraste de complejidad.

**Validación:** si la unidad es geográfica, **Spatial K-Fold** o GroupKFold por departamento, además del K-Fold estándar (comparar ambos es un hallazgo en sí mismo). Si hay años, validación temporal (entrenar en años previos, probar en posteriores).

**Complejidad:** O() teórico + tiempos medidos de entrenamiento/inferencia vs. tamaño de muestra.

---

## 7. Entregables del ETL para la Entrega 1

- [ ] Dataset integrado en `data/processed/` (aunque sea versión preliminar con 2–3 fuentes)
- [ ] Diccionario de datos
- [ ] Notebook de EDA comentado: distribución del objetivo, nulos por fuente, relación variables–objetivo, mapa por departamento/municipio
- [ ] Diagrama del pipeline ETL + modelado para la diapositiva de "solución propuesta"
- [ ] Bitácora `etl_log.md` con decisiones y pérdidas de datos en cada paso

---

## 8. Instrucciones sugeridas para Claude Code

1. Leer este documento completo antes de empezar.
2. Crear la estructura del repositorio y `config.yaml` con la unidad de análisis como parámetro.
3. Implementar primero **extract + transform de Saber 11 para un solo periodo**. Validar el mapeo de columnas antes de generalizar a todos los años.
4. Implementar `homologation/divipola.py` con pruebas unitarias (casos: ceros a la izquierda, nombres con tildes, municipios nuevos).
5. Integrar SNIES y DANE; dejar SPADIES como lector de archivos exportados manualmente.
6. Agregar validaciones y generación del diccionario de datos.
7. Comentar el código en español (la rúbrica evalúa "código debidamente comentado").
8. No subir `data/raw/` a git; sí subir los scripts que lo regeneran.

## 9. Preguntas abiertas
- ¿Unidad de análisis final? (depende de la decisión de tesis y de lo que SPADIES deje exportar)
- ¿Regresión de la tasa o clasificación en niveles de riesgo?
- ¿La cuenta institucional da acceso ampliado a SPADIES?
- ¿Ventana temporal común entre fuentes?
