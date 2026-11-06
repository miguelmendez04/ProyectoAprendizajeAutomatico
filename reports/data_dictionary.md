# Diccionario de datos

Filas: 2148 · Columnas: 14

| Variable | Tipo | Fuente | Descripción | % nulos | Rango |
|---|---|---|---|---|---|
| `cod_ies` | str | SNIES (MEN) | Código de la institución de educación superior | 0.0% | 321 valores distintos |
| `anio` | int64 | — | Año del objetivo (t). Las variables explicativas son del año t-1 | 0.0% | 2015 – 2021 |
| `tasa_desercion` | float64 | SPADIES (MEN) | Tasa de deserción anual como proporción 0-1. Variable objetivo | 0.0% | 0 – 0.9942 |
| `cod_divipola_mpio` | str | DIVIPOLA (DANE) | Código DANE del municipio, 5 dígitos con ceros a la izquierda | 0.0% | 73 valores distintos |
| `sector` | str | SNIES (MEN) | Sector de la IES (id_sector: oficial/privada) | 0.0% | 2 valores distintos |
| `caracter` | str | SNIES (MEN) | Carácter académico de la IES (id_caracter) | 0.0% | 4 valores distintos |
| `matriculados_ies_t2` | float64 | SNIES (MEN) | Matrícula anual de la IES (suma de sus programas) (año t-2) | 30.0% | 6 – 4.676e+05 |
| `punt_global_media_t1` | float64 | Saber 11 (ICFES) | Media del puntaje global de los estudiantes del municipio (año t-1) | 28.5% | 206.9 – 294.8 |
| `punt_global_mediana_t1` | float64 | Saber 11 (ICFES) | Mediana del puntaje global de los estudiantes del municipio (año t-1) | 28.5% | 201 – 295 |
| `pct_estrato_bajo_t1` | float64 | Saber 11 (ICFES) | Proporción de estudiantes de estratos 1-2 (entre los de estrato conocido) (año t-1) | 28.5% | 0.2439 – 1 |
| `pct_colegio_oficial_t1` | float64 | Saber 11 (ICFES) | Proporción de estudiantes de colegio oficial (año t-1) | 28.5% | 0.3469 – 1 |
| `n_estudiantes_t1` | float64 | Saber 11 (ICFES) | Número de estudiantes que presentaron Saber 11 (año t-1) | 28.5% | 74 – 9.671e+04 |
| `mpio_matriculados_t2` | float64 | SNIES (MEN) | Matrícula de educación superior en el municipio (año t-2) | 0.2% | 54 – 8.045e+05 |
| `mpio_ies_con_oferta_t2` | float64 | SNIES (MEN) | IES con oferta en el municipio (año t-2) | 0.2% | 1 – 126 |
