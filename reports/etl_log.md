# Bitácora ETL

Decisiones manuales (insumo para el paper). La sección posterior a la marca la regenera `pipeline.py`.

## Fuentes y ventana
- **IDs de la guía inválidos** (`m2nt-jw2h`, `w7bi-rwg5`, `86ha-bzcy` devuelven 404). Se usan, verificados el 2026-11-05: Saber 11 `kgxf-xxbe` (unificado), matrícula municipal `y9ga-zwzy`, matrícula IES×programa `5wck-szir`, DIVIPOLA `gdxc-w37w`.
- **Ventana 2015–2021**: límite de la matrícula SNIES. Saber 11 se baja desde 2014 por el rezago t-1.
- **SPADIES** es la única fuente del objetivo (no hay deserción abierta por IES/municipio): sin su export el pipeline produce un dataset preliminar con `tasa_desercion` nula.

## Calidad de las fuentes (revisión final)
- `y9ga-zwzy` repite cada municipio-año (solo cambia la tildación del nombre): se elimina el duplicado antes de sumar niveles. Sin esto la matrícula salía ×2 (Bogotá 2020: 1,43 M vs 0,83 M en 2021).
- `5wck-szir` repite filas completas en 2015–2018: `drop_duplicates()` antes de sumar géneros. **Pendiente confirmar con el MEN.**
- Saber 11 unificado: el periodo 20194 trae cada estudiante dos veces (se deduplica por `estu_consecutivo`) y faltan periodos del calendario A (2018-2, 2020-2, 2021-2). Los años con cobertura menor a la mitad de la mediana se **excluyen** (sus variables _t1 quedan nulas) en vez de alimentar rasgos sesgados.
- datos.gov.co responde 500/503/timeouts de forma intermitente: la descarga reintenta con espera exponencial y escribe a `.part` para no dejar CSV truncados.

## Convenciones de construcción
- **Matrícula anual de programa** = máximo semestral (sumar semestres cuenta dos veces a los mismos estudiantes). **Matrícula de IES** = suma de sus programas.
- **Programa en varios municipios**: se conserva el de mayor matrícula.
- **Unidad municipio**: tasa = media simple (no ponderada) de las IES con sede en el municipio. No se conserva el número de IES (se calcula con el objetivo del año t: fuga).
- **Rezago**: todas las variables con año se desplazan t-1 (sufijo `_t1`); sector/carácter (IES) y población/NBI (DANE, estático) no se rezagan. **Riesgo abierto**: si el NBI viene de un censo posterior a la ventana, hay información posterior al objetivo.
- Fuera del alcance de este ETL y por decidir con la ficha metodológica de SPADIES: si el rezago correcto es t-1 (los de Saber 11 de t-1 apenas ingresan a ES en t; podría ser t-2 o t-3) y si la matrícula t-1 se relaciona mecánicamente con la tasa (matriculados en t-2 en el denominador).
- Mapa del EDA: dispersión lon/lat de DIVIPOLA (no hay shapefile ni geopandas).

## Pendientes manuales
Ver `reports/fase0.md`: consulta pública y cuenta UAO en SPADIES, archivos DANE y TerriData, histórico de cambios de código DIVIPOLA (`config.yaml: cambios_codigo`).

<!-- pipeline:inicio -->

## Última corrida del pipeline

Unidad de análisis: `municipio` · Filas del dataset: 3074

### Pérdidas por join (filas de la base sin pareja)

| join | filas_izq | filas_der | filas_resultado | filas_sin_match |
|---|---|---|---|---|
| saber11 | 3074 | 5550 | 3074 | 889 |
| snies_municipio | 3074 | 9682 | 3074 | 684 |

### Observaciones de calidad

- Saber 11: 571268 fila(s) de estudiantes repetidos (mismo estu_consecutivo) eliminada(s)
- Saber 11: cobertura incompleta en los años [2018, 2020, 2021] (falta el calendario A); se excluyen, por lo que las variables Saber 11 de los años [2019, 2021, 2022] quedan nulas
- SNIES matrícula municipal: 9629 fila(s) duplicada(s) (solo difieren en el nombre) eliminada(s)
- SNIES matrícula municipal: 6 fila(s) con código de municipio nulo o especial descartada(s)
- SNIES matrícula por programa: 160808 fila(s) duplicada(s) exactas eliminada(s)
- DANE: no hay archivo manual en data/raw; la fuente no se integró
- TerriData: no hay archivo manual en data/raw; la fuente no se integró
- saber11_municipio: 10 filas con código de municipio que no está en DIVIPOLA vigente
- SPADIES: no existe data\raw\spadies\desercion_ies.csv; dataset preliminar con tasa_desercion nula
