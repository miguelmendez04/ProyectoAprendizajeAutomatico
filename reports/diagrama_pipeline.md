# Diagrama del pipeline ETL + modelado

Para la diapositiva de "solución propuesta": pegar el bloque en <https://mermaid.live> y exportar a PNG/SVG.

```mermaid
flowchart LR
    subgraph Fuentes
        S11[Saber 11 · ICFES<br/>datos.gov.co kgxf-xxbe]
        SN[SNIES · MEN<br/>matrícula y catálogo IES/programas]
        DV[DIVIPOLA · DANE]
        DA[TerriData (DNP)<br/>zips manuales]
        SP[SPADIES · MEN<br/>export manual = variable objetivo]
    end

    subgraph ETL
        EX[Extract<br/>Socrata paginado + metadata.json]
        TR[Transform por fuente<br/>tipos, nulos, códigos especiales]
        HO[Homologación DIVIPOLA<br/>llave cod_divipola_mpio 5 dígitos]
        IN[Integrate<br/>agregación a la unidad + rezago t-1]
        VA[Validate · pandera<br/>rangos, llave única, nulos]
    end

    DS[(Dataset integrado<br/>ies · programa · municipio × año)]

    subgraph Modelado
        PP[Preprocesamiento ×3<br/>faltantes · escalado/codificación · selección]
        MD[Modelos ≥3<br/>lineal/logística · Random Forest · XGBoost · SVM/MLP]
        CV[Validación<br/>K-Fold · Spatial/GroupKFold · temporal]
        CX[Complejidad<br/>O&#40;&#41; teórico + tiempos medidos]
    end

    S11 & SN & DV & DA & SP --> EX --> TR --> HO --> IN --> VA --> DS
    DS --> PP --> MD --> CV --> CX
    VA -.-> RP[reports/<br/>data_dictionary.md · etl_log.md]
```

**Alcance:** el modelo predice riesgo de deserción a nivel institucional/territorial; no existen datos abiertos de deserción por estudiante.
