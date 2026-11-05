# Diccionario de datos

Filas: 7 · Columnas: 14

| Variable | Tipo | Fuente | Descripción | % nulos | Rango |
|---|---|---|---|---|---|
| `cod_ies` | str | SNIES (MEN) | Código de la institución de educación superior | 0.0% | 1 valores distintos |
| `anio` | int64 | — | Año del objetivo (t). Las variables explicativas son del año t-1 | 0.0% | 2015 – 2021 |
| `tasa_desercion` | float64 | SPADIES (MEN) | Tasa de deserción anual como proporción 0-1. Variable objetivo | 0.0% | 0.07912 – 0.09779 |
| `cod_divipola_mpio` | str | DIVIPOLA (DANE) | Código DANE del municipio, 5 dígitos con ceros a la izquierda | 0.0% | 1 valores distintos |
| `sector` | str | SNIES (MEN) | Sector de la IES (id_sector: oficial/privada) | 0.0% | 1 valores distintos |
| `caracter` | str | SNIES (MEN) | Carácter académico de la IES (id_caracter) | 0.0% | 1 valores distintos |
| `matriculados_ies_t1` | float64 | SNIES (MEN) | Matrícula anual de la IES (suma de sus programas) (año t-1) | 14.3% | 9921 – 1.119e+04 |
| `punt_global_media_t1` | float64 | Saber 11 (ICFES) | Media del puntaje global de los estudiantes del municipio (año t-1) | 28.6% | 248.2 – 267.6 |
| `punt_global_mediana_t1` | float64 | Saber 11 (ICFES) | Mediana del puntaje global de los estudiantes del municipio (año t-1) | 28.6% | 246 – 265 |
| `pct_estrato_bajo_t1` | float64 | Saber 11 (ICFES) | Proporción de estudiantes de estratos 1-2 (entre los de estrato conocido) (año t-1) | 28.6% | 0.5269 – 0.6005 |
| `pct_colegio_oficial_t1` | float64 | Saber 11 (ICFES) | Proporción de estudiantes de colegio oficial (año t-1) | 28.6% | 0.4368 – 0.5315 |
| `n_estudiantes_t1` | float64 | Saber 11 (ICFES) | Número de estudiantes que presentaron Saber 11 (año t-1) | 28.6% | 1.872e+04 – 2.472e+04 |
| `mpio_matriculados_t1` | int64 | SNIES (MEN) | Matrícula de educación superior en el municipio (año t-1) | 0.0% | 1.148e+05 – 1.327e+05 |
| `mpio_ies_con_oferta_t1` | int64 | SNIES (MEN) | IES con oferta en el municipio (año t-1) | 0.0% | 47 – 57 |
