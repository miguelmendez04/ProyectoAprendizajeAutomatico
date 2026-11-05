# Bitácora ETL

<!-- pipeline:inicio -->

## Última corrida del pipeline

Unidad de análisis: `municipio` · Filas del dataset: 3074

### Pérdidas por join (filas de la base sin pareja)

| join | filas_izq | filas_der | filas_resultado | filas_sin_match |
|---|---|---|---|---|
| saber11 | 3074 | 5889 | 3074 | 693 |
| snies_municipio | 3074 | 9682 | 3074 | 684 |

### Observaciones de calidad

- DANE: no hay archivo manual en data/raw; la fuente no se integró
- TerriData: no hay archivo manual en data/raw; la fuente no se integró
- saber11_municipio: 11 filas con código de municipio que no está en DIVIPOLA vigente
- SPADIES: no existe data\raw\spadies\desercion_ies.csv; dataset preliminar con tasa_desercion nula
