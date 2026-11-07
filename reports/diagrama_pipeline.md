# Diagrama del pipeline ETL + modelado

GitHub dibuja el bloque de abajo directamente. Para la diapositiva de "solución propuesta": pegarlo en <https://mermaid.live> y exportar a PNG/SVG.

```mermaid
flowchart LR
    subgraph Fuentes
        S11["Saber 11 · ICFES<br/>datos.gov.co kgxf-xxbe"]
        SN["SNIES · MEN<br/>matrícula por programa y municipio,<br/>catálogo de IES"]
        DV["DIVIPOLA · DANE"]
        TD["TerriData · DNP<br/>zips manuales (Dim 2, 4, 7, 8, 14, 16, 25)"]
        SP["SPADIES · MEN<br/>export manual = variable objetivo<br/>tasa = desertores t / matriculados t-2 semestres"]
    end

    subgraph ETL
        EX["Extract<br/>Socrata paginado + metadata.json"]
        TR["Transform por fuente<br/>tipos, nulos, tasa anual ponderada,<br/>censo 2018 desde el año objetivo 2020"]
        HO["Homologación DIVIPOLA<br/>llave cod_divipola_mpio de 5 dígitos"]
        FQ["Filtros de calidad<br/>denominador mínimo 100 · posible cierre<br/>inconsistencia SPADIES/SNIES · régimen especial<br/>bandera SENA"]
        IN["Integrate<br/>IES × año · contexto por municipio sede<br/>Saber 11 y TerriData t-1 · matrícula SNIES t-2<br/>indicadores falta_*"]
        VA["Validate · pandera<br/>rangos, llave única, nulos"]
    end

    DS[("Dataset integrado<br/>IES × año, 2015–2021<br/>1.928 filas · 295 IES")]

    subgraph Modelado
        PP["Preprocesamiento<br/>imputación · escalado · one-hot<br/>dentro de cada pliegue"]
        MD["Modelos ≥ 3<br/>lineal · Random Forest · XGBoost · SVM/MLP"]
        CV["Validación<br/>GroupKFold por IES · estratificado ·<br/>por departamento · sin Bogotá · temporal"]
        SG["Sesgo<br/>pesos muestrales · métricas por subgrupo ·<br/>sensibilidad al umbral"]
        CX["Complejidad<br/>O(n) teórico + tiempos medidos"]
    end

    S11 & SN & DV & TD & SP --> EX --> TR --> HO --> FQ --> IN --> VA --> DS
    DS --> PP --> MD --> CV --> CX
    CV --> SG
    FQ -.-> RQ["reports/<br/>exclusiones_calidad.csv · tasas_altas.csv"]
    VA -.-> RP["reports/<br/>data_dictionary.md · etl_log.md"]
    SG -.-> RS["reports/analisis_sesgo.md<br/>notebooks/02_sesgo.ipynb"]
```

**Alcance:** el modelo predice riesgo de deserción a nivel institucional (IES × año); no existen datos abiertos de deserción por estudiante. Los pasos de modelado (modelos, complejidad) son el plan del proyecto; en el repositorio están implementados la partición, el preprocesador, el objetivo con logit opcional y las utilidades de sesgo (`src/modelado/`).
