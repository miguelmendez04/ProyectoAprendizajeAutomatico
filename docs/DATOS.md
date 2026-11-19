# Datos: fuentes, descarga, checksums y licencias

El ETL (`desercion etl`) parte de archivos crudos que **no se versionan** (`data/raw/`, ~316 MB). El resultado, `data/processed/dataset_ies.csv` (1.928 filas × 295 IES, 843 KB), **sí está en el repositorio**, por lo que el modelado (todas las etapas excepto `etl`) se reproduce sin descargar nada.

Los SHA-256 de los crudos y del dataset final están en [`data/checksums.json`](../data/checksums.json). Cada etapa de `desercion reproducir` los verifica al empezar (`--sin-verificar` lo omite); `desercion reproducir --solo-verificar` los muestra todos.

## Fuentes automáticas (datos.gov.co; las descarga `desercion etl` sin `--saltar-extract`)
| Archivo esperado | Conjunto (ID) | URL | Descargado | Licencia* |
|---|---|---|---|---|
| `data/raw/saber11/saber11.csv` | Resultados únicos Saber 11 (`kgxf-xxbe`; ICFES) | https://www.datos.gov.co/resource/kgxf-xxbe.json | 2026-11-05 | CC BY-SA 4.0 |
| `data/raw/snies/matricula_programa.csv` | MEN matrícula estadística ES (`5wck-szir`) | https://www.datos.gov.co/resource/5wck-szir.json | 2026-11-05 | CC BY-SA 4.0 |
| `data/raw/snies/matricula_mpio.csv` | MEN matrícula por municipios ES (`y9ga-zwzy`) | https://www.datos.gov.co/resource/y9ga-zwzy.json | 2026-11-05 | CC BY-SA 4.0 |
| `data/raw/snies/ies_catalogo.csv` | MEN instituciones de educación superior (`n5yy-8nav`) | https://www.datos.gov.co/resource/n5yy-8nav.json | 2026-11-05 | CC BY-SA 4.0 |
| `data/raw/dane/divipola.csv` | DIVIPOLA, códigos de municipios (`gdxc-w37w`; DANE) | https://www.datos.gov.co/resource/gdxc-w37w.json | 2026-11-05 | CC BY-SA 4.0 |

*Licencia leída de la API de metadatos de datos.gov.co (`https://www.datos.gov.co/api/views/<ID>.json`) el 2026-11-18. Saber 11 pesa ~3,4 M de filas y su descarga tarda ~25 min; datos.gov.co falla de forma intermitente y la descarga reintenta sola. Estos conjuntos **se actualizan**: una descarga nueva puede diferir de la usada aquí (por eso se fijan los SHA-256 y la fecha). Cada descarga deja un `*.metadata.json` con fuente, URL, fecha, filas y hash.

## Fuentes manuales
### SPADIES (variable objetivo; requiere inicio de sesión)
1. Ingresar a la consulta pública de SPADIES (Ministerio de Educación): https://spadies.mineducacion.gov.co (crear cuenta/iniciar sesión).
2. *Analíticas → Deserción Anual → tipo de cálculo: IES → IES: Diferenciado* y exportar a CSV (todas las IES en un archivo; separador `;`, BOM UTF-8, columnas `IES;TIPO;1998-1;…;2024-2`, dos filas por IES: `DESERTORES` y `MATRICULADOS`).
3. Guardar como `data/raw/spadies/spadies_diferenciado.csv` (descargado el 2026-11-06).
4. El mismo menú sin diferenciar por IES (total nacional) → `data/raw/spadies/spadies_nacional.csv` (2026-11-05; solo contexto y control de las sumas).

Sin el diferenciado, el pipeline produce un dataset *preliminar* con `tasa_desercion` nula. El formato y el cálculo de la tasa están en `reports/01_etl.md`.

### TerriData (DNP; contexto territorial)
1. Ir a https://terridata.dnp.gov.co → *Descargas* (datos por dimensión, formato TXT).
2. Descargar las dimensiones 2, 4, 7, 8, 14, 16 y 25 y guardarlas **sin renombrar ni descomprimir** en `data/raw/terridata/` (en minúsculas, como en `config.yaml`): `TerriData_Dim2.txt.zip`, `…Dim4…`, `…Dim7…`, `…Dim8…`, `…Dim14…`, `…Dim16…`, `…Dim25…` (descargadas el 2026-11-06).
3. Contenido: 2 población; 4 educación; 7 finanzas públicas; 8 desempeño municipal; 14 pobreza (NBI/IPM); 16 mercado laboral; 25 proyecciones de población con base en el censo 2005.

Las fechas de SPADIES nacional y de TerriData son las de modificación del archivo (no hay `metadata.json`). La URL de cada portal es la página de entrada; los menús pueden cambiar con el tiempo.

## Reproducir el ETL
Con los 14 crudos en su carpeta: `desercion reproducir --etapa etl` (≈ 2 min con los crudos ya descargados; usa `--saltar-extract`). Al terminar compara el SHA-256 del dataset generado con el publicado: **la corrida del ETL desde los crudos fijados produce un dataset byte a byte idéntico** (se comprobó el 2026-11-18). `desercion etl` sin `--saltar-extract` vuelve a descargar las fuentes de datos.gov.co, que pueden haber cambiado.

## Licencia de `dataset_ies.csv`
Está en el repositorio y se distribuye bajo **CC BY-SA 4.0** (ver [`data/processed/LICENSE`](../data/processed/LICENSE)), con atribución a MEN (SPADIES y SNIES), ICFES, DNP (TerriData) y DANE (DIVIPOLA).
- Es una **tabla derivada y agregada** por IES × año (tasas, conteos y variables de contexto municipal); no contiene datos de personas.
- Las cinco fuentes de datos.gov.co son CC BY-SA 4.0 (atribución y compartir igual); de ahí que el derivado use la misma licencia.
- **No pude verificar las condiciones de uso de SPADIES ni de TerriData** (no hay una licencia legible por máquina como en datos.gov.co); sus cifras son estadísticas oficiales públicas y aquí solo aparecen agregadas por IES o municipio. Conviene que el equipo las confirme antes de redistribuir fuera del curso.
- El código es MIT (ver [`LICENSE`](../LICENSE)); los datos crudos (no versionados) conservan las condiciones de sus fuentes.
