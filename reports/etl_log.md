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

## SPADIES: fuente principal, formato y cálculo de la tasa (actualizado 2026-11-06)
- **Fuente principal**: `data/raw/spadies/spadies_diferenciado.csv` (SPADIES → Analíticas → Deserción Anual → tipo de cálculo IES → IES: Diferenciado). CSV con `;` y BOM UTF-8, columnas `IES;TIPO;1998-1…2024-2`, dos filas por IES (`DESERTORES`, `MATRICULADOS`): 343 IES, 18.522 filas en formato largo. El código SNIES viene en la columna `IES`. Reemplaza los exports por IES.
- **Archivos ignorados** en `data/raw/spadies/`: los `spadies_<código>_<nombre>.csv` con una fila `TASADESERCION` (tasa ya calculada por SPADIES, no se usa: se recalcula a partir de desertores y matriculados), `spadies_todas_ies_desertores_matriculados.csv` (copia idéntica byte a byte del diferenciado) y el export previo de la UAO (solo sirvió de control). El pipeline lee únicamente `spadies.archivo` y `spadies_nacional.csv`.
- **Prueba con la UAO (1830)**: la serie del diferenciado coincide exactamente (54 periodos, desertores y matriculados) con el export por IES recibido antes.
- **Ceros iniciales** de desertores (1998-1, 1998-2 y similares al inicio de la serie de cada IES) = dato faltante, no cero.
- **Definición (SPADIES)**: `tasa_desercion_t = desertores_t / matriculados_(t-2)`, con t-2 = **dos semestres antes** (mismo semestre del año anterior). Se interpretó "periodo -2" como dos periodos semestrales; si la ficha de SPADIES dice dos años, el denominador cambia.
- **Agregación anual ponderada**: tasa anual = (desertores 1.er sem. + desertores 2.º sem.) / (matriculados t-2 del 1.er + del 2.º sem.), es decir, el promedio de las tasas semestrales ponderado por su denominador. Reemplaza el promedio simple anterior, que daba el mismo peso a un semestre con menos base. Sin los dos semestres no hay tasa anual. La periodicidad semestral (`spadies.periodicidad: semestre`) usa las tasas semestrales sin agregar.
- **Variables de matrícula en t-2**: las de SNIES (`matriculados_ies`, `matriculados_programa`, `mpio_matriculados`, `mpio_ies_con_oferta`) se rezagan 2 años (`rezago_matricula: 2` en `config.yaml`, sufijo `_t2`), el mismo desfase del denominador. Saber 11 y TerriData siguen en t-1. Consecuencia: `matriculados_ies_t2` queda nula en 2015–2016 (la matrícula por programa de SNIES empieza en 2015; 30 % de nulos). Como `matriculados_ies_t2` casi coincide con el denominador de la tasa, hay una relación mecánica con el objetivo que conviene tener presente al modelar.
- **Resultado**: dataset IES × año 2015–2021 con 2.148 filas, 321 IES, `tasa_desercion` sin nulos (media 0,129; mediana 0,100; rango 0–0,994). Años con menos IES (304–310) porque faltan tasas donde no hay denominador o los dos semestres.
- **Tasas atípicas**: 3 tasas semestrales > 1 (desertores > matrícula t-2) se descartan con nota; hay tasas anuales cercanas a 0,99 en IES pequeñas o en crecimiento rápido (el denominador es la matrícula de hace un año). Revisar antes de modelar si se winsoriza o se filtra por tamaño.
- **Control contra el total nacional**: la matrícula de las IES suma exactamente el nacional en todos los periodos, pero la suma de desertores lo supera ligeramente en 17 de 54 periodos (p. ej. 2015-1 por 64, 2021-2 por 2.829, 1,4 %; 2003-1 por 4,2 %). Hipótesis, sin confirmar: estudiantes contados como desertores en más de una IES (se retiran de una y de otra) o ajustes del nacional. No invalida el uso, pero queda anotado en cada corrida.
- **Seccionales**: SNIES identifica una IES por código y SPADIES puede reportar sus seccionales agregadas bajo la sede principal. Asignar la IES a un solo municipio sede (el que concentra más programas en la matrícula SNIES) es una **aproximación**: el contexto territorial puede no reflejar a los estudiantes de otras sedes. La corrida lista las IES con programas en más de un municipio (4: 1202, 2813, 3117, 3828). El catálogo oficial (`n5yy-8nav`) marca 61 de 361 instituciones como "Seccional".
- **Cobertura**: 343 IES en SPADIES frente a 338 con programas en SNIES; 321 IES entran al dataset (con tasa y sede conocida).

## Lista de IES a exportar (`reports/ies_a_descargar.csv`)
- Generada con `python -m src.integrate.ies_descargar` (matrícula 2021 de SNIES). Criterio: todas las IES con sede en el Valle del Cauca (30) + las 3 de mayor matrícula de Bogotá, Medellín, Cali, Barranquilla y Bucaramanga + la mayor de las demás capitales (Cartagena, Cúcuta, Pereira, Manizales, Ibagué, Santa Marta, Villavicencio, Pasto, Tunja, Popayán, Neiva, Armenia, Montería, Valledupar) + el resto por matrícula hasta 60.
- Columnas: código SNIES, nombre oficial (catálogo `n5yy-8nav`), municipio sede y departamento (DIVIPOLA, normalizados), sector (oficial/privada) y matrícula 2021.
- **Matrícula = suma de programas** (máximo semestral por programa), incluye pregrado y posgrado. **El SENA (9110) figura primero por matrícula pero probablemente SPADIES no lo cubre**; si su consulta no lo ofrece, descartarlo y tomar la siguiente IES.

## Pendientes manuales
Ya no hace falta descargar IES una a una (el export diferenciado las trae todas; `reports/ies_a_descargar.csv` queda como referencia de cobertura). Ver `reports/fase0.md`: cuenta UAO en SPADIES, archivos DANE y TerriData, histórico de cambios de código DIVIPOLA (`config.yaml: cambios_codigo`).

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

### Observaciones de calidad

- Saber 11: 571268 fila(s) de estudiantes repetidos (mismo estu_consecutivo) eliminada(s)
- Saber 11: cobertura incompleta en los años [2018, 2020, 2021] (falta el calendario A); se excluyen, por lo que las variables Saber 11 de los años [2019, 2021, 2022] quedan nulas
- SNIES matrícula municipal: 9629 fila(s) duplicada(s) (solo difieren en el nombre) eliminada(s)
- SNIES matrícula municipal: 6 fila(s) con código de municipio nulo o especial descartada(s)
- SNIES matrícula por programa: 160808 fila(s) duplicada(s) exactas eliminada(s)
- DANE: no hay archivo manual en data/raw; la fuente no se integró
- TerriData: no hay archivo manual en data/raw; la fuente no se integró
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
