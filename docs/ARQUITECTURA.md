# Arquitectura

Paquete `desercion_ies` (src layout) y comando `desercion`. Los diagramas están en [Mermaid](https://mermaid.js.org/) y se ven en GitHub. Los nombres de módulos son los de `src/desercion_ies/`; el traductor desde la estructura anterior está en [`MIGRACION.md`](MIGRACION.md).

## 1. Módulos y dependencias
Las flechas van de quien usa a quien es usado. `cli` solo importa la biblioteca estándar: carga cada etapa al usarla (para fijar los hilos de BLAS antes de importar numpy en `evaluar complejidad`).

```mermaid
flowchart TD
    CLI["cli.py<br/>comando desercion"]:::entrada
    CLI --> ETL
    CLI --> RUN["runners/<br/>v1 v2 tarea1 tarea2<br/>ajuste sesgo complejidad"]
    CLI --> REP["reproduce.py<br/>etapas, --rapido"]
    CLI --> INF["inference/<br/>entrenar, predecir, modelos"]
    CLI --> VER["reproducibilidad.py<br/>checksums, comparación"]

    subgraph ETL["etl/"]
        direction LR
        EXT["extract/"] --> TRA["transform/"] --> INT["integrate/"] --> VAL["validate/"]
        HOM["homologation/<br/>DIVIPOLA"] --> INT
        PIPE["pipeline.py"] -.orquesta.-> EXT
    end

    RUN --> EVAL["evaluation/<br/>protocolo, experimentos,<br/>formulaciones, tarea2,<br/>ajuste, temporal,<br/>arranque_frio, evaluacion_final"]
    RUN --> ANA["analysis/<br/>sesgo, subgrupos, sensibilidad,<br/>interpretabilidad, complejidad…"]
    RUN --> REPO["reporting/<br/>informe_*"]
    EVAL --> FEAT["features/<br/>particion, objetivo"]
    EVAL --> MOD["models/<br/>metodos: KernelPCA,<br/>autoencoder, clusters"]
    ANA --> EVAL
    REPO --> EVAL
    INF --> EVAL
    INF --> FEAT
    FEAT --> CFG["config.py, paths.py<br/>configs/config.yaml"]
    ETL --> CFG

    classDef entrada fill:#e8f1ff,stroke:#2b6cb0;
```

## 2. Flujo de datos
Los archivos crudos no se versionan; el dataset procesado sí. Cada etapa de modelado parte de `data/processed/dataset_ies.csv` y deja tablas, figuras e informes numerados en `reports/`.

```mermaid
flowchart LR
    subgraph Fuentes
        S1[("SPADIES<br/>(manual)")]
        S2[("SNIES, Saber 11,<br/>DIVIPOLA<br/>(datos.gov.co)")]
        S3[("TerriData<br/>(manual)")]
    end
    S1 & S2 & S3 --> RAW[("data/raw/<br/>14 archivos + SHA-256")]
    RAW -->|"desercion etl"| DS[("data/processed/<br/>dataset_ies.csv<br/>1.928 filas × 295 IES")]
    RAW -->|"anexos"| RETL["reports/01_etl.md<br/>reports/etl/"]
    DS -->|"desercion evaluar<br/>v1, v2, tarea1, tarea2,<br/>ajuste, sesgo, complejidad"| TAB[("reports/tables/<br/>CSV y JSON")]
    TAB --> INFS["reports/02…07 .md<br/>reports/figures/"]
    TAB --> NB["notebooks/01…06"]
    TAB -->|"hiperparámetros<br/>de consenso"| ENT["desercion entrenar"]
    DS --> ENT
    ENT --> MD[("models/<br/>.joblib + .json<br/>model_card.md")]
    MD --> PRED["desercion predecir"]
    CSV["CSV de entrada"] --> PRED --> OUT["predicciones.csv"]
    DS -.->|"desercion verificar"| CHK["data/checksums.json"]
```

## 3. Etapas de evaluación y su dependencia
`desercion reproducir` encadena las etapas en este orden; cada una necesita la salida de la anterior (en `--rapido`, si falta, se copia de lo publicado).

```mermaid
flowchart LR
    ETL["etl"] --> EDA["eda"]
    ETL --> T1["tarea1<br/>v1, v2, v3"]
    ETL --> T2["tarea2<br/>cribado + confirmación"]
    T2 -->|"mejor configuración<br/>de cada modelo"| AJ["ajuste<br/>anidado, temporal, frío"]
    AJ -->|"hiperparámetros finales"| SE["sesgo<br/>subgrupos, sensibilidad, SHAP"]
    AJ --> CX["complejidad<br/>un hilo"]
    AJ -->|"ajuste.csv"| TR["entrenar → predecir"]
```

## 4. Camino de una predicción
`predecir` usa el mismo camino que la evaluación (`ajuste.preparar` → `evaluacion_final.construir_final` → recorte a [0, 1]), de modo que una fila predicha por el modelo guardado y por el pipeline de evaluación da el mismo valor (hay una prueba que lo exige).

```mermaid
sequenceDiagram
    participant U as Usuario
    participant P as predecir.py
    participant M as ModeloEntrenado
    participant E as Estimador (pipeline sklearn)
    U->>P: --tarea, --entrada X.csv, --salida Y.csv
    P->>P: comprueba SHA-256 de models/*.joblib
    P->>M: carga joblib + metadatos JSON
    P->>M: predecir(df)
    M->>M: validar(df): columnas, tipos, tasa_t1 (Tarea 1)
    alt entrada mal formada
        M-->>P: EntradaInvalida (qué falta y cómo corregirlo)
        P-->>U: mensaje claro, código de salida 2
    else entrada válida
        M->>E: preparar_x(df) → predict
        E-->>M: tasa estimada
        M-->>P: recorte a [0, 1]
        P-->>U: Y.csv (identificadores, modelo, tasa_predicha)
    end
```

## 5. Decisiones de diseño
- **Una configuración, muchas etapas:** `configs/config.yaml` fija datos, filtros, grillas, semillas y rutas de salida; el modo `--rapido` solo la reescribe (menos semillas y grillas más chicas) y redirige las salidas a `reproduccion/`.
- **Sin cambios de lógica al reorganizar:** la estructura se movió con `git mv` y los resultados se comprobaron byte a byte (ver `REPRODUCIBILIDAD.md`).
- **Raíz del proyecto sin depender del directorio actual:** `paths.py` la detecta desde el propio paquete (o `DESERCION_ROOT`); el comando cambia a esa raíz porque las rutas de la configuración son relativas a ella.
- **Informes generados desde las tablas:** cada `reporting/informe_*` lee los CSV de `reports/tables/` y escribe el informe; por eso `--solo-informe` regenera un informe sin repetir los cálculos.
- **Modelos guardados con integridad:** cada `.joblib` lleva el SHA-256 en su JSON, que `predecir` verifica antes de cargarlo (joblib es pickle).
