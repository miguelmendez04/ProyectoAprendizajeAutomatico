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

## Cambio de unidad de análisis: IES × año (2026-11-05)
- `unidad_analisis: ies`. El contexto territorial (Saber 11, matrícula municipal, DANE/TerriData) se une por el **municipio sede** de la IES (DIVIPOLA) y se rezaga t-1. Configurable a semestre (`spadies.periodicidad: semestre`): la llave pasa a `cod_ies, anio, semestre` y las variables explicativas siguen siendo anuales (año t-1).

## SPADIES: formato real y cálculo de la tasa
- **Formato**: CSV con `;`, columna `TIPO` y una columna por semestre (`1998-1` … `2024-2`), filas `DESERTORES` y `MATRICULADOS`. **Un archivo por IES**, `spadies_<código SNIES>_<nombre>.csv`, en `data/raw/spadies/`; el código sale del nombre. El total nacional va en `spadies_nacional.csv`. Se pasa a formato largo `ies, periodo, desertores, matriculados`.
- **Los 0 iniciales** de desertores (1998-1, 1998-2 en el nacional y en la UAO) se tratan como dato faltante, no como cero.
- **Definición (SPADIES)**: `tasa_desercion_t = desertores_t / matriculados_(t-2)`, con t-2 = **dos semestres antes** (mismo semestre del año anterior). Se interpretó "periodo -2" como dos periodos semestrales; la alternativa (dos años) cambia el denominador y debe confirmarse con la ficha metodológica.
- **Agregación anual**: promedio de las tasas de los dos semestres del año; sin los dos semestres no hay tasa anual (el año queda sin fila). Se eligió promediar y no tomar solo un semestre para no perder información, y porque la tasa es muy distinta entre semestre 1 y 2 en el agregado nacional (p. ej. 2020-1 0,081 vs. 2020-2 0,131).
- **Solapamiento con las explicativas**: el denominador es la matrícula de t-2, y `matriculados_ies_t1` es la de t-1: están correlacionadas por construcción. Sigue abierto si t-1 es el rezago correcto (ver arriba).
- **Total nacional**: solo contexto y control: por periodo, la suma de desertores y de matriculados de las IES cargadas no puede superar el nacional (si lo hace, queda una nota en la corrida).
- **UAO (1830)**, único export desagregado recibido hasta ahora: tasas anuales 2015–2021 entre 0,079 y 0,098 (dataset de 7 filas); el pipeline corre de punta a punta con ese único archivo.
- **Seccionales**: SNIES identifica una IES por código y SPADIES puede reportar sus seccionales agregadas bajo la sede principal. Asignar la IES a un solo municipio sede (el que concentra más programas en la matrícula SNIES) es una **aproximación**: el contexto territorial puede no reflejar a los estudiantes de otras sedes. La corrida lista las IES con programas en más de un municipio. El catálogo oficial (`n5yy-8nav`) marca 61 de 361 instituciones como "Seccional".

## Lista de IES a exportar (`reports/ies_a_descargar.csv`)
- Generada con `python -m src.integrate.ies_descargar` (matrícula 2021 de SNIES). Criterio: todas las IES con sede en el Valle del Cauca (30) + las 3 de mayor matrícula de Bogotá, Medellín, Cali, Barranquilla y Bucaramanga + la mayor de las demás capitales (Cartagena, Cúcuta, Pereira, Manizales, Ibagué, Santa Marta, Villavicencio, Pasto, Tunja, Popayán, Neiva, Armenia, Montería, Valledupar) + el resto por matrícula hasta 60.
- Columnas: código SNIES, nombre oficial (catálogo `n5yy-8nav`), municipio sede y departamento (DIVIPOLA, normalizados), sector (oficial/privada) y matrícula 2021.
- **Matrícula = suma de programas** (máximo semestral por programa), incluye pregrado y posgrado. **El SENA (9110) figura primero por matrícula pero probablemente SPADIES no lo cubre**; si su consulta no lo ofrece, descartarlo y tomar la siguiente IES.

## Pendientes manuales
Descargar de SPADIES los exports de `reports/ies_a_descargar.csv` (uno por IES, nombrados `spadies_<código>_<nombre>.csv`). Ver también `reports/fase0.md`: cuenta UAO en SPADIES, archivos DANE y TerriData, histórico de cambios de código DIVIPOLA (`config.yaml: cambios_codigo`).

<!-- pipeline:inicio -->

## Última corrida del pipeline

Unidad de análisis: `ies` · Filas del dataset: 7

### Pérdidas por join (filas de la base sin pareja)

| join | filas_izq | filas_der | filas_resultado | filas_sin_match |
|---|---|---|---|---|
| catalogo_ies | 7 | 338 | 7 | 0 |
| matricula_ies | 7 | 2220 | 7 | 1 |
| saber11 | 7 | 5550 | 7 | 2 |
| snies_municipio | 7 | 9682 | 7 | 0 |

### Observaciones de calidad

- Saber 11: 571268 fila(s) de estudiantes repetidos (mismo estu_consecutivo) eliminada(s)
- Saber 11: cobertura incompleta en los años [2018, 2020, 2021] (falta el calendario A); se excluyen, por lo que las variables Saber 11 de los años [2019, 2021, 2022] quedan nulas
- SNIES matrícula municipal: 9629 fila(s) duplicada(s) (solo difieren en el nombre) eliminada(s)
- SNIES matrícula municipal: 6 fila(s) con código de municipio nulo o especial descartada(s)
- SNIES matrícula por programa: 160808 fila(s) duplicada(s) exactas eliminada(s)
- DANE: no hay archivo manual en data/raw; la fuente no se integró
- TerriData: no hay archivo manual en data/raw; la fuente no se integró
- saber11_municipio: 10 filas con código de municipio que no está en DIVIPOLA vigente
- SPADIES: 1 IES leídas de data\raw\spadies (periodos 1998-1 a 2024-2)
