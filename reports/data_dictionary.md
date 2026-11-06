# Diccionario de datos

Filas: 1928 · Columnas: 42

| Variable | Tipo | Fuente | Descripción | % nulos | Rango |
|---|---|---|---|---|---|
| `cod_ies` | str | SNIES (MEN) | Código de la institución de educación superior | 0.0% | 295 valores distintos |
| `anio` | int64 | — | Año del objetivo (t). Las variables explicativas son del año t-1 | 0.0% | 2015 – 2021 |
| `tasa_desercion` | float64 | SPADIES (MEN) | Tasa de deserción anual como proporción 0-1. Variable objetivo | 0.0% | 0.002712 – 0.9942 |
| `posible_cierre` | bool | SPADIES (MEN) | Bandera de diagnóstico: la fila cae en los 2 semestres previos al último reporte de una IES que dejó de reportar. No es explicativa | 0.0% | 1 valores distintos |
| `ratio_spadies_snies` | float64 | SPADIES / SNIES | Denominador de la tasa (matrícula SPADIES t-2) dividido por la matrícula SNIES t-2 de la IES. Diagnóstico, no explicativa | 29.0% | 0.3539 – 2.628 |
| `inconsistencia_fuentes` | bool | SPADIES / SNIES | Bandera: ratio_spadies_snies > 3 o < 1/3, o SNIES sin dato dentro de su cobertura. Diagnóstico | 0.0% | 1 valores distintos |
| `regimen_especial` | bool | SNIES (MEN) | Bandera: escuela militar o policial (lista explícita de 16 códigos SNIES). Diagnóstico | 0.0% | 1 valores distintos |
| `es_sena` | bool | SNIES (MEN) | Bandera: la IES es el SENA (código 9110). Diagnóstico | 0.0% | 2 valores distintos |
| `cod_divipola_mpio` | str | DIVIPOLA (DANE) | Código DANE del municipio, 5 dígitos con ceros a la izquierda | 0.0% | 67 valores distintos |
| `sector` | str | SNIES (MEN) | Sector de la IES (id_sector: oficial/privada) | 0.0% | 2 valores distintos |
| `caracter` | str | SNIES (MEN) | Carácter académico de la IES (id_caracter) | 0.0% | 4 valores distintos |
| `matriculados_ies_t2` | float64 | SNIES (MEN) | Matrícula anual de la IES (suma de sus programas) (año t-2) | 29.0% | 39 – 4.676e+05 |
| `punt_global_media_t1` | float64 | Saber 11 (ICFES) | Media del puntaje global de los estudiantes del municipio (año t-1) | 28.3% | 206.9 – 294.8 |
| `punt_global_mediana_t1` | float64 | Saber 11 (ICFES) | Mediana del puntaje global de los estudiantes del municipio (año t-1) | 28.3% | 201 – 295 |
| `pct_estrato_bajo_t1` | float64 | Saber 11 (ICFES) | Proporción de estudiantes de estratos 1-2 (entre los de estrato conocido) (año t-1) | 28.3% | 0.2439 – 0.9874 |
| `pct_colegio_oficial_t1` | float64 | Saber 11 (ICFES) | Proporción de estudiantes de colegio oficial (año t-1) | 28.3% | 0.3469 – 1 |
| `n_estudiantes_t1` | float64 | Saber 11 (ICFES) | Número de estudiantes que presentaron Saber 11 (año t-1) | 28.3% | 214 – 9.671e+04 |
| `mpio_matriculados_t2` | int64 | SNIES (MEN) | Matrícula de educación superior en el municipio (año t-2) | 0.0% | 56 – 8.045e+05 |
| `mpio_ies_con_oferta_t2` | int64 | SNIES (MEN) | IES con oferta en el municipio (año t-2) | 0.0% | 1 – 126 |
| `td_cob_bruta_media_t1` | float64 | TerriData (DNP) | Cobertura bruta en educación media, % (año t-1) | 0.4% | 48.11 – 192 |
| `td_cob_bruta_total_t1` | float64 | TerriData (DNP) | Cobertura bruta en educación (preescolar a media), % (año t-1) | 0.4% | 73.76 – 142.6 |
| `td_cob_neta_media_t1` | float64 | TerriData (DNP) | Cobertura neta en educación media, % (año t-1) | 0.4% | 20.05 – 75.67 |
| `td_desercion_escolar_t1` | float64 | TerriData (DNP) | Deserción escolar intra-anual (transición a once), % (año t-1) | 14.8% | 0.72 – 8.99 |
| `td_idf_t1` | float64 | TerriData (DNP) | Indicador de desempeño fiscal (DNP), puntos (año t-1) | 0.2% | 49.22 – 92.24 |
| `td_idf_ahorro_t1` | float64 | TerriData (DNP) | Capacidad de ahorro (desempeño fiscal) (año t-1) | 0.2% | 12.07 – 85.28 |
| `td_idf_deuda_t1` | float64 | TerriData (DNP) | Respaldo de la deuda (desempeño fiscal) (año t-1) | 0.2% | 0 – 50.68 |
| `td_idf_inversion_t1` | float64 | TerriData (DNP) | Porcentaje del gasto destinado a inversión (año t-1) | 0.2% | 59.17 – 96.75 |
| `td_idf_recursos_propios_t1` | float64 | TerriData (DNP) | Porcentaje de ingresos corrientes por recursos propios (año t-1) | 0.2% | 53.41 – 99.76 |
| `td_idf_transferencias_t1` | float64 | TerriData (DNP) | Porcentaje de ingresos por transferencias (año t-1) | 0.2% | 11.97 – 91.96 |
| `td_ipm_censo_t1` | float64 | TerriData (DNP) | Índice de pobreza multidimensional (censo), % (año t-1) | 0.2% | 4.5 – 78.22 |
| `td_mdm_t1` | float64 | TerriData (DNP) | Medición de desempeño municipal, puntos (año t-1) | 29.1% | 32.41 – 86.45 |
| `td_mdm_educacion_t1` | float64 | TerriData (DNP) | MDM: resultados en educación, puntos (año t-1) | 29.1% | 34.6 – 82.8 |
| `td_mdm_gestion_t1` | float64 | TerriData (DNP) | MDM: componente de gestión, puntos (año t-1) | 29.1% | 32.12 – 86.18 |
| `td_mdm_resultados_t1` | float64 | TerriData (DNP) | MDM: componente de resultados, puntos (año t-1) | 29.1% | 58.42 – 82.73 |
| `td_miseria_pct_t1` | float64 | TerriData (DNP) | Población en condición de miseria (censo), % (año t-1) | 0.2% | 0.06 – 37.68 |
| `td_nbi_pct_t1` | float64 | TerriData (DNP) | Necesidades básicas insatisfechas (censo), % (año t-1) | 0.2% | 1.59 – 89.47 |
| `td_pct_pob_15_19_t1` | float64 | TerriData (DNP) | Porcentaje de población de 15 a 19 años (año t-1) | 0.2% | 5.62 – 12.82 |
| `td_pct_pob_20_24_t1` | float64 | TerriData (DNP) | Porcentaje de población de 20 a 24 años (año t-1) | 0.2% | 6.723 – 13.3 |
| `td_pct_poblacion_rural_t1` | float64 | TerriData (DNP) | Porcentaje de población rural (año t-1) | 0.2% | 0.06 – 68.27 |
| `td_repitencia_oficial_t1` | float64 | TerriData (DNP) | Repitencia del sector oficial en básica y media, % (año t-1) | 0.4% | 0 – 13.19 |
| `td_transito_superior_t1` | float64 | TerriData (DNP) | Tasa de tránsito inmediato a la educación superior, % (año t-1) | 0.2% | 13.82 – 72.88 |
| `td_poblacion_total_t1` | float64 | TerriData (DNP) | Indicador territorial poblacion_total (año t-1) | 0.2% | 1.859e+04 – 8.081e+06 |
