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
- **Rezago**: Saber 11 y TerriData se desplazan t-1 (`_t1`); la matrícula SNIES, t-2 (`_t2`, `rezago_matricula` en `config.yaml`, el mismo desfase del denominador de la tasa). Sector y carácter de la IES no se rezagan. NBI e IPM de TerriData son censales (2005 y 2018): para el año t se usa el último censo ya levantado (ver sección de TerriData).
- **Rezago de la tasa (cerrado el 2026-11-06)**: t-2 = dos semestres, confirmado en el glosario y en las preguntas frecuentes de SPADIES (MEN); ver la sección de SPADIES. Sigue abierto, como decisión de modelado, si t-1 es el mejor rezago para Saber 11 (quienes presentaron Saber 11 en t-1 apenas ingresan a la educación superior en t) y la relación mecánica entre `matriculados_ies_t2` y el denominador de la tasa.
- **EDA** (`notebooks/01_eda.ipynb`, generado y ejecutado con `python notebooks/construir_eda.py`): mapa coroplético por departamento con los polígonos del DANE (MGN 2018) de `data/external/departamentos_colombia.geojson`, dibujados con matplotlib (sin geopandas). Fuente del GeoJSON: repositorio `caticoa3/colombia_mapa` en GitHub (copia del MGN 2018); no es una fuente oficial directa.

## Cambio de unidad de análisis: IES × año (2026-11-05)
- `unidad_analisis: ies`. El contexto territorial (Saber 11, matrícula municipal, DANE/TerriData) se une por el **municipio sede** de la IES (DIVIPOLA) y se rezaga t-1. Configurable a semestre (`spadies.periodicidad: semestre`): la llave pasa a `cod_ies, anio, semestre` y las variables explicativas siguen siendo anuales (año t-1).

## SPADIES: fuente principal, formato y cálculo de la tasa (actualizado 2026-11-06)
- **Fuente principal**: `data/raw/spadies/spadies_diferenciado.csv` (SPADIES → Analíticas → Deserción Anual → tipo de cálculo IES → IES: Diferenciado). CSV con `;` y BOM UTF-8, columnas `IES;TIPO;1998-1…2024-2`, dos filas por IES (`DESERTORES`, `MATRICULADOS`): 343 IES, 18.522 filas en formato largo. El código SNIES viene en la columna `IES`. Reemplaza los exports por IES.
- **Archivos ignorados** en `data/raw/spadies/`: los `spadies_<código>_<nombre>.csv` con una fila `TASADESERCION` (tasa ya calculada por SPADIES, no se usa: se recalcula a partir de desertores y matriculados), `spadies_todas_ies_desertores_matriculados.csv` (copia idéntica byte a byte del diferenciado) y el export previo de la UAO (solo sirvió de control). El pipeline lee únicamente `spadies.archivo` y `spadies_nacional.csv`.
- **Prueba con la UAO (1830)**: la serie del diferenciado coincide exactamente (54 periodos, desertores y matriculados) con el export por IES recibido antes.
- **Ceros iniciales** de desertores (1998-1, 1998-2 y similares al inicio de la serie de cada IES) = dato faltante, no cero.
- **Definición (SPADIES) — confirmada**: `tasa_desercion_t = desertores_t / matriculados_(t-2)`, con t-2 = **dos semestres antes** (mismo semestre del año anterior). Confirmado por el usuario el 2026-11-06 en dos fuentes del MEN: el **Glosario de SPADIES** y las **Preguntas frecuentes (FAQ) de SPADIES**. (Se cita por nombre; las URL no quedaron registradas en la sesión: completar con el enlace de consulta al redactar el paper.) La alternativa de dos años queda descartada.
- **Agregación anual ponderada**: tasa anual = (desertores 1.er sem. + desertores 2.º sem.) / (matriculados t-2 del 1.er + del 2.º sem.), es decir, el promedio de las tasas semestrales ponderado por su denominador. Reemplaza el promedio simple anterior, que daba el mismo peso a un semestre con menos base. Sin los dos semestres no hay tasa anual. La periodicidad semestral (`spadies.periodicidad: semestre`) usa las tasas semestrales sin agregar.
- **Variables de matrícula en t-2**: las de SNIES (`matriculados_ies`, `matriculados_programa`, `mpio_matriculados`, `mpio_ies_con_oferta`) se rezagan 2 años (`rezago_matricula: 2` en `config.yaml`, sufijo `_t2`), el mismo desfase del denominador. Saber 11 y TerriData siguen en t-1. Consecuencia: `matriculados_ies_t2` queda nula en 2015–2016 (la matrícula por programa de SNIES empieza en 2015; 30 % de nulos). Como `matriculados_ies_t2` casi coincide con el denominador de la tasa, hay una relación mecánica con el objetivo que conviene tener presente al modelar.
- **Resultado**: dataset IES × año 2015–2021 con 2.148 filas, 321 IES, `tasa_desercion` sin nulos (media 0,129; mediana 0,100; rango 0–0,994). Años con menos IES (304–310) porque faltan tasas donde no hay denominador o los dos semestres.
- **Tasas atípicas**: 3 tasas semestrales > 1 (desertores > matrícula t-2) se descartan con nota; hay tasas anuales cercanas a 0,99 en IES pequeñas o en crecimiento rápido (el denominador es la matrícula de hace un año). Revisar antes de modelar si se winsoriza o se filtra por tamaño.
- **Control contra el total nacional**: la matrícula de las IES suma exactamente el nacional en todos los periodos, pero la suma de desertores lo supera ligeramente en 17 de 54 periodos (p. ej. 2015-1 por 64, 2021-2 por 2.829, 1,4 %; 2003-1 por 4,2 %). Hipótesis, sin confirmar: estudiantes contados como desertores en más de una IES (se retiran de una y de otra) o ajustes del nacional. No invalida el uso, pero queda anotado en cada corrida.
- **Seccionales**: SNIES identifica una IES por código y SPADIES puede reportar sus seccionales agregadas bajo la sede principal. Asignar la IES a un solo municipio sede (el que concentra más programas en la matrícula SNIES) es una **aproximación**: el contexto territorial puede no reflejar a los estudiantes de otras sedes. La corrida lista las IES con programas en más de un municipio (4: 1202, 2813, 3117, 3828). El catálogo oficial (`n5yy-8nav`) marca 61 de 361 instituciones como "Seccional".
- **Cobertura**: 343 IES en SPADIES frente a 338 con programas en SNIES; 321 IES entran al dataset (con tasa y sede conocida).

## TerriData: fuente de población, NBI/IPM y contexto territorial (2026-11-06)
- **Fuente**: zips `data/raw/terridata/TerriData_Dim{2,4,7,8,14,16}.txt.zip` (DNP), formato largo `|` con columnas Código Departamento, Código Entidad, Entidad, Dimensión, Subcategoría, Indicador, Dato Numérico, Dato Cualitativo, Año, Mes, Fuente, Unidad de Medida. Reemplaza los archivos manuales del DANE (se eliminó ese código y su configuración). La dimensión 2 (población) pesa ~1 GB descomprimida: se lee por bloques y el resultado filtrado se guarda en `data/interim/terridata_largo.csv` (se reutiliza mientras no cambien los zips ni el catálogo).
- **Nivel municipal**: entidades de 5 dígitos que no terminan en `000` (departamentos) ni son el total nacional (dpto `01`). 1.102 municipios con datos.
- **Indicadores** (catálogo en `src/transform/terridata.py`, 28 variables antes de depurar): población total (hombres + mujeres), % población rural y % de 15–19 y 20–24 años (Dim 2); coberturas bruta/neta, cobertura en superior, tránsito inmediato a superior, deserción escolar intra-anual y repitencia (Dim 4); indicador de desempeño fiscal y cinco componentes (Dim 7); medición de desempeño municipal (MDM) y sus componentes de gestión, resultados y educación (Dim 8); NBI, miseria e IPM censal (Dim 14); cotizantes, empresas generadoras de empleo formal, ocupación formal y tasa global de participación (Dim 16).
- **Cómo se lleva cada indicador al año t**: *anual* = el dato de ese año (si falta queda nulo, sin arrastrar valores viejos); *censo* (NBI, miseria, IPM: solo existen 2005 y 2018) = último censo ya levantado, así el NBI de 2005 rige hasta 2017 y el de 2018 desde 2018; *población* (proyecciones DANE con base en el censo 2018, solo desde 2018) = el dato del año y, para años anteriores, el de 2018 **completado hacia atrás**. Esto último es una aproximación (la población municipal cambia lento) y es información posterior al año del objetivo para 2015–2017; su costo si falla es sesgo pequeño en `td_poblacion_total_t1` y `td_pct_*` en los primeros años. Luego todo se rezaga t-1 (`td_*_t1`).
- **Variables descartadas por nulos**: las `td_*` con más de 50 % de nulos tras el rezago se eliminan y quedan listadas en cada corrida (`max_nulos_pct`). En esta corrida: cobertura en educación superior (solo existe desde 2018), cotizantes y empresas de empleo formal (terminan en 2016), ocupación formal (casi todo cualitativo) y tasa global de participación (solo 23 ciudades). **Con esto el dataset se queda sin variables de mercado laboral**: ninguna serie municipal cubre la ventana 2015–2021.
- **IES sin match**: 1 de 321 IES (la de código 4106, con sede en 88001, San Andrés) no tiene su municipio en TerriData y queda con las `td_*` nulas. El resto de IES tiene sede en un municipio presente. El log también cuenta las filas sin pareja del join (7 IES-año).
- **Rezago del NBI censal**: el censo 2018 se publica después de 2018; asignarlo al año 2018 (y por tanto como t-1 de 2019) supone cierta anticipación respecto a la fecha real de publicación. Riesgo bajo, anotado.
- **Deserción escolar y otras variables TerriData** son de otro nivel educativo (básica y media); no son el objetivo ni deben confundirse con él.

## Lista de IES a exportar (`reports/ies_a_descargar.csv`)
- Generada con `python -m src.integrate.ies_descargar` (matrícula 2021 de SNIES). Criterio: todas las IES con sede en el Valle del Cauca (30) + las 3 de mayor matrícula de Bogotá, Medellín, Cali, Barranquilla y Bucaramanga + la mayor de las demás capitales (Cartagena, Cúcuta, Pereira, Manizales, Ibagué, Santa Marta, Villavicencio, Pasto, Tunja, Popayán, Neiva, Armenia, Montería, Valledupar) + el resto por matrícula hasta 60.
- Columnas: código SNIES, nombre oficial (catálogo `n5yy-8nav`), municipio sede y departamento (DIVIPOLA, normalizados), sector (oficial/privada) y matrícula 2021.
- **Matrícula = suma de programas** (máximo semestral por programa), incluye pregrado y posgrado. **El SENA (9110) figura primero por matrícula pero probablemente SPADIES no lo cubre**; si su consulta no lo ofrece, descartarlo y tomar la siguiente IES.

## Pendientes manuales
Ya no hace falta descargar IES una a una (el export diferenciado las trae todas; `reports/ies_a_descargar.csv` queda como referencia de cobertura) ni archivos del DANE (TerriData cubre población y NBI). Quedan: histórico de cambios de código DIVIPOLA (`config.yaml: cambios_codigo`, hoy vacío), confirmar con el MEN las filas repetidas de `5wck-szir`, y decidir el tratamiento de tasas atípicas, de la colinealidad territorial y de la ausencia de mercado laboral municipal (ver el EDA).

<!-- pipeline:inicio -->

## Última corrida del pipeline

Unidad de análisis: `ies` · Filas del dataset: 2148

### Pérdidas por join (filas de la base sin pareja)

| join | filas_izq | filas_der | filas_resultado | filas_sin_match |
|---|---|---|---|---|
| catalogo_ies | 2148 | 338 | 2148 | 0 |
| matricula_ies | 2148 | 2220 | 2148 | 644 |
| saber11 | 2148 | 5550 | 2148 | 612 |
| snies_municipio | 2148 | 9682 | 2148 | 4 |
| terridata | 2148 | 7714 | 2148 | 7 |

### Observaciones de calidad

- Saber 11: 571268 fila(s) de estudiantes repetidos (mismo estu_consecutivo) eliminada(s)
- Saber 11: cobertura incompleta en los años [2018, 2020, 2021] (falta el calendario A); se excluyen, por lo que las variables Saber 11 de los años [2019, 2021, 2022] quedan nulas
- SNIES matrícula municipal: 9629 fila(s) duplicada(s) (solo difieren en el nombre) eliminada(s)
- SNIES matrícula municipal: 6 fila(s) con código de municipio nulo o especial descartada(s)
- SNIES matrícula por programa: 160808 fila(s) duplicada(s) exactas eliminada(s)
- saber11_municipio: 10 filas con código de municipio que no está en DIVIPOLA vigente
- SPADIES: 343 IES leídas de spadies_diferenciado.csv (periodos 1998-1 a 2024-2)
- SPADIES: en 1999-2 la suma de desertores de las IES (30900) supera el total nacional (30892)
- SPADIES: en 2001-1 la suma de desertores de las IES (44571) supera el total nacional (44544)
- SPADIES: en 2003-1 la suma de desertores de las IES (62932) supera el total nacional (59421)
- SPADIES: en 2003-2 la suma de desertores de las IES (65399) supera el total nacional (64877)
- SPADIES: en 2007-1 la suma de desertores de las IES (83936) supera el total nacional (83593)
- SPADIES: en 2008-2 la suma de desertores de las IES (108150) supera el total nacional (108121)
- SPADIES: en 2010-1 la suma de desertores de las IES (129794) supera el total nacional (129749)
- SPADIES: en 2012-2 la suma de desertores de las IES (162228) supera el total nacional (162178)
- SPADIES: en 2013-1 la suma de desertores de las IES (165474) supera el total nacional (164705)
- SPADIES: en 2014-1 la suma de desertores de las IES (190147) supera el total nacional (190131)
- SPADIES: en 2014-2 la suma de desertores de las IES (199181) supera el total nacional (199060)
- SPADIES: en 2015-1 la suma de desertores de las IES (204433) supera el total nacional (204369)
- SPADIES: en 2018-2 la suma de desertores de las IES (244796) supera el total nacional (244788)
- SPADIES: en 2019-2 la suma de desertores de las IES (238596) supera el total nacional (238313)
- SPADIES: en 2021-1 la suma de desertores de las IES (171159) supera el total nacional (170671)
- SPADIES: en 2021-2 la suma de desertores de las IES (202449) supera el total nacional (199620)
- SPADIES: en 2022-2 la suma de desertores de las IES (245588) supera el total nacional (245009)
- SPADIES: 3 tasa(s) semestral(es) mayor que 1 (desertores > matrícula t-2); se descartan
- IES con programas en más de un municipio (4: 1202, 2813, 3117, 3828): SPADIES las reporta agregadas y se asignan al municipio con más programas; el contexto territorial es una aproximación
- TerriData: 5 de 28 variable(s) descartada(s) por superar 50% de nulos tras el rezago: td_cob_superior_t1 (57%), td_cotizantes_t1 (57%), td_empresas_empleo_formal_t1 (57%), td_ocupacion_formal_pct_t1 (57%), td_tgp_t1 (65%)
- TerriData: 1 de 321 IES sin municipio sede en TerriData (4106→88001); quedan con sus variables td_* nulas
