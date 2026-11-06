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
- **Definición (SPADIES) — confirmada**: `tasa_desercion_t = desertores_t / matriculados_(t-2)`, con t-2 = **dos semestres antes** (mismo semestre del año anterior). Confirmado por el usuario el 2026-11-06 en dos fuentes del MEN: el **Glosario de SPADIES** y las **Preguntas frecuentes (FAQ) de SPADIES**. Fuentes: Glosario de SPADIES, https://www.mineducacion.gov.co/sistemasinfo/spadies/secciones/Ayuda/Glosario/ ; Preguntas frecuentes de SPADIES, https://www.mineducacion.gov.co/sistemasinfo/spadies/secciones/Ayuda/Preguntas-frecuentes/ . La alternativa de dos años queda descartada.
- **Agregación anual ponderada**: tasa anual = (desertores 1.er sem. + desertores 2.º sem.) / (matriculados t-2 del 1.er + del 2.º sem.), es decir, el promedio de las tasas semestrales ponderado por su denominador. Reemplaza el promedio simple anterior, que daba el mismo peso a un semestre con menos base. Sin los dos semestres no hay tasa anual. La periodicidad semestral (`spadies.periodicidad: semestre`) usa las tasas semestrales sin agregar.
- **Variables de matrícula en t-2**: las de SNIES (`matriculados_ies`, `matriculados_programa`, `mpio_matriculados`, `mpio_ies_con_oferta`) se rezagan 2 años (`rezago_matricula: 2` en `config.yaml`, sufijo `_t2`), el mismo desfase del denominador. Saber 11 y TerriData siguen en t-1. Consecuencia: `matriculados_ies_t2` queda nula en 2015–2016 (la matrícula por programa de SNIES empieza en 2015; 30 % de nulos). Como `matriculados_ies_t2` casi coincide con el denominador de la tasa, hay una relación mecánica con el objetivo que conviene tener presente al modelar.
- **Resultado** (con el denominador mínimo, sin posibles cierres, sin inconsistencias entre fuentes y sin régimen especial; ver abajo): dataset IES × año 2015–2021 con **1.928 filas, 295 IES**, `tasa_desercion` sin nulos (media 0,120; mediana 0,099; p5–p95 de 4,3 % a 26,1 %; rango 0,003–0,994; asimetría 3,8). Sin ninguna exclusión eran 2.148 filas y 321 IES.
- **Tasas atípicas**: 3 tasas semestrales > 1 (desertores > matrícula t-2) se descartan con nota. Aun con el denominador mínimo quedan 20 IES-año con tasa anual > 50 % (matrícula t-2 mediana de 265). Decisión de modelado pendiente: winsorizar o subir `min_matriculados_t2`.
- **Control contra el total nacional**: la matrícula de las IES suma exactamente el nacional en todos los periodos, pero la suma de desertores lo supera ligeramente en 17 de 54 periodos (p. ej. 2015-1 por 64, 2021-2 por 2.829, 1,4 %; 2003-1 por 4,2 %). Hipótesis, sin confirmar: estudiantes contados como desertores en más de una IES (se retiran de una y de otra) o ajustes del nacional. No invalida el uso, pero queda anotado en cada corrida.
- **Seccionales**: SNIES identifica una IES por código y SPADIES puede reportar sus seccionales agregadas bajo la sede principal. Asignar la IES a un solo municipio sede (el que concentra más programas en la matrícula SNIES) es una **aproximación**: el contexto territorial puede no reflejar a los estudiantes de otras sedes. La corrida lista las IES con programas en más de un municipio (4: 1202, 2813, 3117, 3828). El catálogo oficial (`n5yy-8nav`) marca 61 de 361 instituciones como "Seccional".
- **Cobertura**: 343 IES en SPADIES → 295 en el dataset; el detalle de las 48 que no llegan (22 sin tasa en la ventana, 9 por el denominador mínimo, 1 por posible cierre, 13 por régimen especial y 3 por inconsistencia entre fuentes) está en la sección siguiente.

## TerriData: fuente de población, NBI/IPM y contexto territorial (2026-11-06)
- **Fuente**: zips `data/raw/terridata/TerriData_Dim{2,4,7,8,14,16}.txt.zip` (DNP), formato largo `|` con columnas Código Departamento, Código Entidad, Entidad, Dimensión, Subcategoría, Indicador, Dato Numérico, Dato Cualitativo, Año, Mes, Fuente, Unidad de Medida. Reemplaza los archivos manuales del DANE (se eliminó ese código y su configuración). La dimensión 2 (población) pesa ~1 GB descomprimida: se lee por bloques y el resultado filtrado se guarda en `data/interim/terridata_largo.csv` (se reutiliza mientras no cambien los zips ni el catálogo).
- **Nivel municipal**: entidades de 5 dígitos que no terminan en `000` (departamentos) ni son el total nacional (dpto `01`). 1.102 municipios con datos.
- **Indicadores** (catálogo en `src/transform/terridata.py`, 28 variables antes de depurar): población total (hombres + mujeres), % población rural y % de 15–19 y 20–24 años (Dim 2); coberturas bruta/neta, cobertura en superior, tránsito inmediato a superior, deserción escolar intra-anual y repitencia (Dim 4); indicador de desempeño fiscal y cinco componentes (Dim 7); medición de desempeño municipal (MDM) y sus componentes de gestión, resultados y educación (Dim 8); NBI, miseria e IPM censal (Dim 14); cotizantes, empresas generadoras de empleo formal, ocupación formal y tasa global de participación (Dim 16).
- **Cómo se lleva cada indicador al año t**: *anual* = el dato de ese año (si falta queda nulo, sin arrastrar valores viejos); *censo* (NBI, miseria, IPM: solo existen 2005 y 2018) = último censo ya levantado, así el NBI de 2005 rige hasta 2017 y el de 2018 desde 2018; *población* = proyección DANE con base en el censo 2018 (Dim 2, desde 2018) y, para los años anteriores, la proyección con base en el censo 2005 (Dim 25, 1985–2020); ver la sección de población. Luego todo se rezaga t-1 (`td_*_t1`).
- **Variables descartadas por nulos**: las `td_*` con más de 50 % de nulos tras el rezago se eliminan y quedan listadas en cada corrida (`max_nulos_pct`). En esta corrida: cobertura en educación superior (solo existe desde 2018), cotizantes y empresas de empleo formal (terminan en 2016), ocupación formal (casi todo cualitativo) y tasa global de participación (solo 23 ciudades). **Con esto el dataset se queda sin variables de mercado laboral**: ninguna serie municipal cubre la ventana 2015–2021.
- **IES sin match**: 1 de 295 IES (la de código 4106, con sede en 88001, San Andrés) no tiene su municipio en TerriData y queda con las `td_*` nulas. El resto de IES tiene sede en un municipio presente. El log también cuenta las filas sin pareja del join.
- **Rezago del NBI censal**: el censo 2018 se publica después de 2018; asignarlo al año 2018 (y por tanto como t-1 de 2019) supone cierta anticipación respecto a la fecha real de publicación. Riesgo bajo, anotado.
- **Deserción escolar y otras variables TerriData** son de otro nivel educativo (básica y media); no son el objetivo ni deben confundirse con él.

## Por qué 48 de las 343 IES de SPADIES no llegan al dataset (2026-11-06)
El dataset final tiene 295 IES: 343 − 22 (sin tasa anual en 2015–2021) − 9 (denominador mínimo) − 1 (posible cierre) − 13 (régimen especial) − 3 (inconsistencia entre fuentes). Cada corrida del pipeline escribe estas cifras en las observaciones de calidad.
- **22 IES sin ninguna tasa anual en la ventana 2015–2021.** La tasa de un año exige los dos semestres, cada uno con matrícula del semestre t-2 (dos semestres atrás). Motivos:
  - *Dejó de reportar antes de 2015* (10): 1124, 1125, 1731, 2722, 3823, 3829, 3833, 4704, 4705, 4827. Probablemente instituciones cerradas, fusionadas o con otro código (sin verificar); su última tasa anual es anterior a 2015.
  - *Sin ninguna tasa anual* (6): 2906, 3802, 9132, 9918, 9923, 9938. 2906 y 3802 son series cortas e interrumpidas que terminan antes de 2015; 9132, 9918 y 9938 tienen un solo periodo con matrícula y ningún desertor informado; 9923 (desde 2020-2) tiene 8 semestres pero desertores informados solo en algunos, sin dos semestres del mismo año con denominador.
  - *Empezó a reportar tan tarde que su primera tasa anual es posterior a 2021* (6): 9924, 9930, 9932, 9934, 9935, 9936. IES de reciente creación (primer dato en 2020-2 o 2021-1).
  Ninguna se pierde por falta de sede en SNIES o por el join con otras fuentes: el filtro es la propia tasa de SPADIES.
- **9 IES por el denominador mínimo** (`min_matriculados_t2 = 100`): 3724, 3805, 3822, 4708, 4710, 4805, 9117, 9928, 9931 (todos sus años con denominador por debajo del mínimo).
- **1 IES por posible cierre** (`excluir_posible_cierre`): 3116 queda sin filas al excluir sus semestres finales.
- **13 IES por régimen especial** (todas sus filas): 2106, 2901, 2902, 2903, 2905, 3114, 3901, 3902, 9102, 9103, 9104, 9105, 9107.
- **3 IES por inconsistencia entre fuentes** (todas sus filas): 9926, 9927, 9933.
- **Sesgo**: el dataset subrepresenta instituciones muy pequeñas, las creadas después de 2019 y las escuelas militares y policiales; las conclusiones valen para IES civiles con al menos ~100 matriculados de referencia, varios años de historia y matrículas consistentes entre SPADIES y SNIES.

## Denominador mínimo de la tasa (`min_matriculados_t2`, 2026-11-06)
- `config.yaml: min_matriculados_t2: 100` (también es el valor por defecto del código). Se excluyen las IES-año cuyo denominador de SPADIES (matrícula en t-2) es menor: en la tasa anual, el promedio de los dos semestres; en la semestral, cada semestre.
- **Efecto medido** (ver el EDA, sección 2): elimina 97 de 2.148 IES-año (4,5 %); las eliminadas tenían media 27,9 % y mediana 24,8 %. La mediana global pasa de 10,0 % a 9,8 %, la media de 12,9 % a 12,2 % y el percentil 95 de 31,1 % a 27,7 %. Quita 13 de las 33 filas con tasa > 50 %.
- El umbral de 100 es un supuesto (no hay un criterio oficial); con denominadores de 100 a 300 todavía hay tasas muy altas. Es un parámetro de configuración para que el análisis de sensibilidad sea directo.

## Partición para el modelado y limitaciones (2026-11-06)
- Utilidades en `src/modelado/particion.py` (parámetros en `config.yaml: particion`): `particion_group_kfold` (GroupKFold por `cod_ies`, 5 pliegues; una IES nunca queda en entrenamiento y prueba a la vez, porque sus años consecutivos son casi la misma observación) y `particion_temporal` (entrena 2015–2019, prueba 2020–2021; aquí las mismas IES aparecen en ambos lados y se mide la extrapolación en el tiempo). También `cargar_dataset` y `columnas_explicativas`.
- **Limitación: sin variables de mercado laboral.** Ninguna serie municipal de TerriData cubre la ventana (cotizantes y empresas de empleo formal terminan en 2016; ocupación formal es casi toda cualitativa; la tasa global de participación existe solo para 23 ciudades y desde 2018). El modelado no puede usar ese bloque. Tampoco hay cobertura en educación superior municipal con rezago t-1 para 2015–2018.
- **Otras limitaciones** ya documentadas: 28,5 % de nulos en Saber 11 (años con cobertura incompleta), `matriculados_ies_t2` casi coincide con el denominador de la tasa, cambio de base de las proyecciones de población entre 2017 y 2018 (ver abajo) y muchas variables territoriales colineales.

## Población con la base 2005 para 2015–2017 (2026-11-06)
- Se integró `TerriData_Dim25.txt.zip` (proyecciones DANE con base en el censo 2005, 1985–2020, 1.101 municipios). Para cada año se usa la proyección con base 2018 (Dim 2, desde 2018) y, donde no existe, la de base 2005. Solo hay respaldo hacia atrás con el primer dato 2018 para municipios ausentes de la base 2005 (1 de 1.102). Columnas afectadas: `td_poblacion_total_t1`, `td_pct_poblacion_rural_t1`, `td_pct_pob_15_19_t1`, `td_pct_pob_20_24_t1`. Para la base 2005 el % rural se calcula como rural / (rural + urbana) y los porcentajes de edad como (hombres + mujeres del grupo) / población total, porque esos porcentajes no vienen en la Dim 25.
- **Cambio de base 2005 → 2018 (salto de nivel entre 2017 y 2018).** En los años en que ambas bases coinciden (2018–2020), la proyección con base 2018 es en total 3,1 % menor en 2018 (48,2 M frente a 49,7 M), 2,1 % en 2019 y 1,0 % en 2020. A nivel municipal la diferencia es grande: el cociente base 2018 / base 2005 tiene mediana 0,996 en 2018 pero rango intercuartil de 0,86 a 1,13 (mínimo 0,33, máximo 3,3); Bogotá queda 9,7 % abajo (7,39 M frente a 8,18 M), Cali 8,9 % y Medellín 4,0 %. Por eso `td_poblacion_total_t1` tiene un salto entre los años 2018 (t-1 de 2018 es 2017, base 2005) y 2019 (t-1 = 2018, base 2018): un modelo que mezcle los años puede leerlo como una caída poblacional que no ocurrió. Se descartó añadir una bandera de base porque equivaldría a una variable de año; conviene usar variables relativas (por ejemplo, el % rural y los % de edad) o validar por año.

## Inconsistencia entre SPADIES y SNIES (2026-11-06)
- **Variable** `ratio_spadies_snies` = denominador de SPADIES (matrícula t-2; en la tasa anual, el promedio de los dos semestres) / matrícula SNIES t-2 de la IES (suma de sus programas, máximo semestral). **Bandera** `inconsistencia_fuentes`: ratio > 3 o < 1/3 (`umbral_ratio_inconsistencia: 3`), o SNIES sin dato dentro de su cobertura. Los umbrales exactos (3 y 1/3) no marcan.
- **Ausencia estructural de la fuente, no inconsistencia (criterio confirmado el 2026-11-06).** La matrícula por programa de SNIES empieza en 2015, así que para los años objetivo 2015 y 2016 el dato de t-2 (2013 y 2014) no existe por construcción: 583 IES-año (291 en 2015 y 292 en 2016). Esa falta es una limitación de la fuente y no una señal de que los datos de la IES sean incoherentes, por lo que **no se marca** como inconsistencia (queda `ratio_spadies_snies` en NaN). La regla solo marca la falta de SNIES cuando el año t-2 cae dentro de su cobertura (t-2 ≥ 2015): 19 IES-año.
- **Por qué no la lectura literal de "o cuando SNIES no tenga dato".** Se marcarían 620 de 2.042 IES-año (los 583 estructurales + 19 con ausencia real + 18 con ratio fuera de rango) y el dataset perdería 2015 y 2016 completos: quedarían 1.369 filas en vez de 1.928 y la partición de entrenamiento 2015–2019 se reduciría a 2017–2019. Con esa lectura saldrían 11 de las 20 tasas > 50 % en vez de 5. Con la regla adoptada se marcan **37** (10 con ratio alto, 8 con ratio bajo y 19 sin dato de SNIES en años con cobertura).
- **Exclusión** (`excluir_inconsistencias`, por defecto `true`): 37 IES-año de 22 IES (3 IES quedan sin filas: 9926, 9927, 9933). **De las 20 tasas > 50 % elimina 3** (2824-2017, 4826-2018 y 9929-2020).
- **Distribución del ratio** en el dataset final (1.369 IES-año con ambos datos): media 0,90, mediana 0,88, p25–p75 de 0,79 a 0,97, mínimo 0,35 y máximo 2,63. Es menor que 1 de forma sistemática porque SNIES incluye posgrado y SPADIES no (o solo parcialmente); por eso el umbral bajo (1/3) es más holgado que el alto.
- `ratio_spadies_snies` incluye el denominador de la propia tasa: **no es explicativa** (queda en `EXCLUIDAS_POR_DEFECTO`). Las columnas de diagnóstico (`ratio_spadies_snies`, `inconsistencia_fuentes`, `regimen_especial`, `es_sena`) quedan en el dataset; la tabla de filas marcadas está en `reports/exclusiones_calidad.csv`.

## Régimen especial: escuelas militares y policiales (2026-11-06)
- **Lista explícita de 16 códigos SNIES** (`REGIMEN_ESPECIAL` en `src/integrate/banderas.py`): 2106 (Dirección Nacional de Escuelas, Policía), 2901, 2902, 2903 (Escuela de Comunicaciones), 2904, 2905, 2906 (Escuela de Postgrados de la Fuerza Aérea), 3114 (Escuela Naval de Suboficiales ARC Barranquilla), 3901, 3902, 9101, 9102, 9103, 9104, 9105 y 9107. Sustituye la regla anterior por prefijos (29xx, 39xx, 910x) y por nombre (ESCUELA MILITAR, POLICIA, NAVAL, FAC): ya no hay búsqueda por nombre ni prefijos, de modo que un código nuevo no se marca por accidente.
- **Fuera de la lista**: 9108 (Instituto Caro y Cuervo) y 9109 (ITEC), que están dentro del rango 910x pero no son instituciones militares, y 9110 (SENA, tiene su propia bandera). Un test falla si 9108 o 9109 quedan marcadas (`test_9108_y_9109_no_son_militares_y_no_pueden_quedar_marcadas`) y otro verifica que los prefijos 9100, 9106, 2907, 3903, etc. ya no cuentan.
- **Opción** `excluir_regimen_especial`, por defecto `true`; la bandera `regimen_especial` queda siempre en el dataset.
- **Resultado**: 85 IES-año de 13 IES (2106, 2901, 2902, 2903, 2905, 3114, 3901, 3902, 9102, 9103, 9104, 9105, 9107; las demás de la lista, 2904, 2906 y 9101, no tienen filas en el dataset: 2906 no llega a tener tasa anual y 2904 y 9101 no aparecen en SPADIES). Las 13 son oficiales. **No se marcó** la Universidad Militar Nueva Granada (1117), que es una universidad pública y no una escuela de formación. **De las 20 tasas > 50 % elimina 2** (2106-2016 y 3114-2018). El cambio de regla no altera las cifras: las 13 IES son las mismas que con la regla anterior.
- Las dos banderas de calidad (inconsistencia y régimen especial) se solapan en 8 IES-año; su unión es de **114 IES-año**, además de las 97 del mínimo y las 9 del posible cierre.

## SENA (9110) (2026-11-06)
- **Bandera** `es_sena` (código 9110), siempre en el dataset. **Opción** `excluir_sena`, por defecto `false`: el SENA no se excluye salvo que se pida. Con `excluir_sena: true` salen sus 7 IES-año (la tabla `reports/exclusiones_calidad.csv` lo registra).
- **Atípico por tamaño, no por tasa** (EDA, sección "¿Es el SENA atípico?"): su matrícula SNIES t-2 va de 426 mil (2017) a 468 mil (2019), entre 113 y 128 veces la mediana de las demás IES (3,7 mil) y entre 3,8 y 4,7 veces la de la mayor otra IES (91 mil a 123 mil). Su tasa de deserción es corriente: mediana de 12,4 % (rango 8,2 % a 16,7 %) frente a 9,8 % en las demás, en los percentiles 29 a 83 de cada año y dentro del p5–p95 (4,3 % a 26,1 %). Su ratio SPADIES/SNIES es de 0,74 a 0,87. Por tamaño, un modelo sensible a la escala se puede distorsionar con 7 puntos de 400 mil matriculados (`matriculados_ies_t2` es explicativa); por tasa no hay motivo para sacarlo.

## Decisión sobre el SENA y matrícula en logaritmo (2026-11-06)
- **El SENA se mantiene** (`excluir_sena: false`, se deja explícito en `config.yaml`) **porque su tasa no es atípica**: mediana de 12,4 % (rango 8,2 %–16,7 %), en los percentiles 29 a 83 de cada año y dentro del p5–p95 de las demás IES. Solo es atípico por tamaño, y el tamaño se maneja con una transformación logarítmica, no con la exclusión.
- **Variable** `log_matriculados_ies_t2 = log1p(matriculados_ies_t2)` (se crea para cada `matriculados_ies_t<k>` de la unidad IES; con el rezago por defecto, t2). Es **la explicativa por defecto**: `matriculados_ies_t2` (bruta) queda en el dataset pero en `EXCLUIDAS_POR_DEFECTO` (y en `modelado.excluir_variables` de `config.yaml`). `columnas_explicativas(df, excluir=[])` la vuelve a incluir.
- **Efecto medido** (EDA, filas con matrícula t-2: 1.369 IES-año): la asimetría de la matrícula baja de 13,5 a −0,2 y el SENA pasa de estar a 40 desviaciones estándar del resto (sin él, 5,1 de asimetría) a 3,5. Sigue siendo el valor más alto, pero en la escala logarítmica ya no domina. La correlación de Spearman con la tasa no cambia (el logaritmo es monótono): −0,35.
- **Análisis de sensibilidad para el paper**: correr el modelado con `excluir_sena: true`. Dos formas: (1) sin rehacer el dataset, `cargar_dataset(cfg, excluir_sena=True)` o `modelado.excluir_sena: true` en `config.yaml` (quita las filas con `es_sena` al cargar; el argumento manda sobre la configuración); (2) rehaciendo el dataset con `excluir_sena: true` en la raíz de `config.yaml`. Con la primera forma el dataset pasa de 1.928 a 1.921 filas (7 IES-año del SENA); en las filas con matrícula, la tasa mediana (9,7 %) y media (11,7 %) y la correlación con la matrícula (−0,35 con y −0,36 sin el SENA) casi no cambian. El resultado de referencia del paper es con SENA; el análisis sin SENA se reporta como robustez.

## Composición oficial / privada antes y después de las exclusiones (2026-11-06)
Naturaleza según el catálogo de SNIES (sector de la sede de la IES). Etapas: sin filtro (toda IES con tasa anual en 2015–2021), tras el denominador mínimo y los posibles cierres, y dataset final (además sin inconsistencias ni régimen especial; el SENA se queda).

| etapa | IES | oficiales | privadas | % IES oficiales | IES-año | oficiales | privadas | % IES-año oficiales |
|---|---|---|---|---|---|---|---|---|
| sin filtro | 321 | 95 | 226 | 29,6 | 2.148 | 637 | 1.511 | 29,7 |
| tras mínimo y posible cierre | 311 | 94 | 217 | 30,2 | 2.042 | 615 | 1.427 | 30,1 |
| dataset final | 295 | 79 | 216 | 26,8 | 1.928 | 526 | 1.402 | 27,3 |

- Entre la primera y la última etapa salen 26 IES: 16 oficiales y 10 privadas. El mínimo y el cierre sacan casi solo privadas (9 de 10, instituciones pequeñas); las exclusiones por calidad sacan casi solo oficiales (15 de 16: las 13 escuelas de régimen especial y 2 con ratio inconsistente). El peso de las oficiales baja unos 3 puntos porcentuales respecto del punto de partida.

## Variables explicativas por defecto: población absoluta fuera (2026-11-06)
- `td_poblacion_total_t1` **queda fuera de las explicativas por defecto** (`modelado.excluir_variables` en `config.yaml`; `EXCLUIDAS_POR_DEFECTO` en `src/modelado/particion.py`). Motivo: la proyección de población cambia de base (censo 2005 → 2018) entre los años objetivo 2018 y 2019 (3,1 % menos en total y 9,7 % menos en Bogotá, ver la sección de población) y un modelo lee ese salto como una caída poblacional que no ocurrió. La población absoluta además es colineal con `n_estudiantes_t1`, `mpio_matriculados_t2` y `mpio_ies_con_oferta_t2`.
- Se conservan las variables relativas: `td_pct_poblacion_rural_t1`, `td_pct_pob_15_19_t1` y `td_pct_pob_20_24_t1`. Se espera volver a incluir la población absoluta cuando se tengan las retroproyecciones del DANE con base 2018 (hoy se usa la base 2005 para 2015–2017). La columna sigue en el dataset; solo cambia la lista por defecto (`columnas_explicativas(df, excluir=[])` la incluye). La lista por defecto también deja fuera `ratio_spadies_snies` y `matriculados_ies_t2` (en su lugar entra `log_matriculados_ies_t2`, ver arriba).

## Tasas > 50 % y posible cierre (2026-11-06)
- **Lista**: `reports/tasas_altas.csv` (la escribe el pipeline, umbral `umbral_tasa_alta: 0.5`). Columnas: código, nombre oficial, año, tasa, denominador de SPADIES (promedio de los dos semestres de t-2), matrícula t-2 de SNIES, último periodo reportado y `posible_cierre`. Se hace tras el denominador mínimo y antes de excluir los posibles cierres.
- **Resultado** (tras el denominador mínimo, antes de las demás exclusiones): 20 IES-año de 13 IES con tasa > 50 %; **ninguna cae en la ventana de posible cierre**. 10 de las 13 IES siguen reportando en 2024-2; 2824, 3828 y 4826 dejaron de reportar en 2022-2, lejos de sus años con tasa alta. Con las banderas de calidad (ver abajo) salen **5 de las 20** y **quedan 15**: 3 por `inconsistencia_fuentes` (2824-2017, 4826-2018, 9929-2020) y 2 por `regimen_especial` (2106-2016, 3114-2018). Las 15 restantes son de 9 IES; las persistentes son 9903 (2015, 2016, 2018, 2019, 2020, con tasas de 63 % a 99 %), 4826 (2015, 2016) y 9899 (2020, 2021). 8 de las 20 no tienen matrícula t-2 de SNIES (la de programas empieza en 2015).
- **Definición de la bandera** `posible_cierre`: una fila (semestral, o anual si alguno de sus dos semestres cumple) está marcada si el semestre es el último con matrícula de la IES o uno de los 2 semestres previos, **y** la IES dejó de reportar antes del final del archivo (2024-2); una IES que sigue activa nunca se marca. Ruling: se interpretó "los 2 semestres previos al último reporte" incluyendo el propio último reporte (ventana de 3 semestres, constante `VENTANA_CIERRE = 2` en `src/extract/spadies.py`); el costo si se quería la lectura estricta es excluir algunas filas más. La bandera es de diagnóstico: no es explicativa y no entra en `columnas_explicativas`.
- **Exclusión** (`config.yaml: excluir_posible_cierre`, por defecto `true`): la corrida actual excluye 9 IES-año de 6 IES (1708, 2824, 2903, 3116, 3810, 3824), una de las cuales (3116) queda sin filas. La bandera queda siempre en el dataset (con la exclusión activa es siempre `false`). El efecto se cuenta después del denominador mínimo, en el orden: mínimo → posible cierre.

## Transformación logit del objetivo (2026-11-06)
- `config.yaml: modelado.transformacion_objetivo` (`ninguna` por defecto | `logit`) y `epsilon_logit` (0,001). Utilidades en `src/modelado/objetivo.py`: `transformar_objetivo`, `invertir_objetivo` y `objetivo_para_modelado(df, cfg)`. El logit recorta la tasa a [ε, 1 − ε] y devuelve log(p / (1 − p)); las predicciones hay que devolverlas a la escala de tasa con `invertir_objetivo` antes de evaluar. No se escribe en el dataset: se aplica en el modelado.
- **Por qué**: la tasa está acotada en [0, 1], es asimétrica (3,8) y tiene cola derecha; el logit la lleva a la recta real y comprime esa cola. Con el mínimo de 0,003 y el máximo de 0,994 el recorte de ε = 0,001 no toca ningún valor actual. Es una opción, no una recomendación: conviene comparar el desempeño con y sin transformar.

## Lista de IES a exportar (`reports/ies_a_descargar.csv`)
- Generada con `python -m src.integrate.ies_descargar` (matrícula 2021 de SNIES). Criterio: todas las IES con sede en el Valle del Cauca (30) + las 3 de mayor matrícula de Bogotá, Medellín, Cali, Barranquilla y Bucaramanga + la mayor de las demás capitales (Cartagena, Cúcuta, Pereira, Manizales, Ibagué, Santa Marta, Villavicencio, Pasto, Tunja, Popayán, Neiva, Armenia, Montería, Valledupar) + el resto por matrícula hasta 60.
- Columnas: código SNIES, nombre oficial (catálogo `n5yy-8nav`), municipio sede y departamento (DIVIPOLA, normalizados), sector (oficial/privada) y matrícula 2021.
- **Matrícula = suma de programas** (máximo semestral por programa), incluye pregrado y posgrado. El SENA (9110) figura primero por matrícula; SPADIES sí lo cubre (7 IES-año en el dataset), pero es atípico por tamaño (ver su sección).

## Pendientes manuales
Ya no hace falta descargar IES una a una (el export diferenciado las trae todas; `reports/ies_a_descargar.csv` queda como referencia de cobertura) ni archivos del DANE (TerriData cubre población y NBI). Quedan: histórico de cambios de código DIVIPOLA (`config.yaml: cambios_codigo`, hoy vacío), confirmar con el MEN las filas repetidas de `5wck-szir`, y decidir el tratamiento de tasas atípicas, de la colinealidad territorial y de la ausencia de mercado laboral municipal (ver el EDA).

<!-- pipeline:inicio -->

## Última corrida del pipeline

Unidad de análisis: `ies` · Filas del dataset: 1928

### Pérdidas por join (filas de la base sin pareja)

| join | filas_izq | filas_der | filas_resultado | filas_sin_match |
|---|---|---|---|---|
| catalogo_ies | 1928 | 338 | 1928 | 0 |
| matricula_ies | 1928 | 2220 | 1928 | 559 |
| saber11 | 1928 | 5550 | 1928 | 545 |
| snies_municipio | 1928 | 9682 | 1928 | 0 |
| terridata | 1928 | 7714 | 1928 | 3 |

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
- SPADIES: 22 de 343 IES de SPADIES no tienen tasa anual en 2015–2021: dejó de reportar antes de 2015 (10: 1124, 1125, 1731, 2722, 3823, 3829, 3833, 4704, 4705, 4827); sin ninguna tasa anual (menos de dos semestres con denominador) (6: 2906, 3802, 9132, 9918, 9923, 9938); empezó a reportar tan tarde que su primera tasa anual es posterior a 2021 (6: 9924, 9930, 9932, 9934, 9935, 9936)
- SPADIES: min_matriculados_t2=100 excluye 97 IES-año (4.5% de 2148) con denominador menor; 9 IES quedan sin ninguna fila (3724, 3805, 3822, 4708, 4710, 4805, 9117, 9928, 9931)
- SPADIES: posible cierre (2 semestres previos al último reporte de una IES que dejó de reportar): excluye 9 fila(s) de 6 IES (1708, 2824, 2903, 3116, 3810, 3824); 1 IES quedan sin ninguna fila (3116)
- IES con programas en más de un municipio (4: 1202, 2813, 3117, 3828): SPADIES las reporta agregadas y se asignan al municipio con más programas; el contexto territorial es una aproximación
- TerriData: 5 de 28 variable(s) descartada(s) por superar 50% de nulos tras el rezago: td_cob_superior_t1 (58%), td_cotizantes_t1 (57%), td_empresas_empleo_formal_t1 (57%), td_ocupacion_formal_pct_t1 (57%), td_tgp_t1 (65%)
- TerriData: 1 de 295 IES sin municipio sede en TerriData (4106→88001); quedan con sus variables td_* nulas
- Inconsistencia SPADIES-SNIES (ratio > 3 o < 1/3, o SNIES sin dato dentro de su cobertura): 37 de 2042 IES-año marcadas (10 con ratio alto, 8 con ratio bajo; 602 sin matrícula SNIES t-2, de las cuales las de años anteriores a la cobertura de SNIES no se marcan); se excluyen
- Régimen especial (escuelas militares y policiales): 85 IES-año de 13 IES (2106, 2901, 2902, 2903, 2905, 3114, 3901, 3902, 9102, 9103, 9104, 9105, 9107); se excluyen
- SENA (9110): 7 IES-año marcados con es_sena; no se excluyen (excluir_sena = false)
- Exclusiones por calidad (detalle en reports/exclusiones_calidad.csv): 114 IES-año de 2042 (unión de las banderas activas)
- SPADIES: 20 IES-año de 13 IES con tasa > 50% tras el denominador mínimo (lista en reports/tasas_altas.csv); marcadas: posible_cierre 0, inconsistencia_fuentes 3, regimen_especial 2, es_sena 0; con alguna bandera 5; se eliminan según la configuración 5
