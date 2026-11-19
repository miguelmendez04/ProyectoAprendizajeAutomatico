# Ficha de modelo (model card)

Formato de Mitchell et al. (2019), «Model Cards for Model Reporting». Todas las cifras salen de los informes y tablas del repositorio
(`reports/04_ajuste_temporal.md`, `reports/06_sesgo_modelos.md`, `reports/05_sesgo_dataset.md` y `reports/tables/`); aquí no hay mediciones nuevas.

## 1. Detalles del modelo
| | |
|---|---|
| **Qué predice** | La **tasa de deserción anual** de una institución de educación superior (IES) colombiana en un año: proporción entre 0 y 1 (fuente del objetivo: SPADIES). La unidad es la **IES × año**, no la persona. |
| **Modelos incluidos** | `tarea1_ridge`: **Tarea 1** (con historia), ridge que corrige la ingenua (la tasa del año anterior) con la historia de la propia IES (formulación «C solo historia»). `tarea2_ridge`: **Tarea 2** (sin historia), ridge con selección por información mutua (objetivo en logit, imputación por mediana) sobre el contexto de la IES. `tarea2_mediana`: **línea base** de la Tarea 2, mediana de la tasa por carácter × sector. |
| **Hiperparámetros** | Los de consenso del ajuste anidado (25 pliegues externos): `tarea1_ridge` alpha ≈ 157,4; `tarea2_ridge` alpha ≈ 4,04 y k = 15 variables. Están en el JSON de cada modelo. Esa combinación no se validó como tal: salió de los mismos datos con los que se evalúa (resultado algo optimista). |
| **Versión / fecha** | Los archivos `models/*.json` guardan fecha de entrenamiento, commit del repositorio, versiones de las bibliotecas y SHA-256 del dataset y del modelo. |
| **Implementación** | scikit-learn 1.9.0 (pipeline de preprocesamiento + `Ridge`/`RidgeCV`), código en `src/desercion_ies/`. Se guardan con joblib (pickle): **no cargue archivos de modelo de origen desconocido**; `desercion predecir` comprueba el SHA-256 antes de cargar. |
| **Licencia** | Código MIT; el dataset de entrenamiento es CC BY-SA 4.0 (`data/processed/LICENSE`). |
| **Contacto** | Autores del repositorio (ver `README.md`). |

## 2. Uso previsto
- **Previsto:** investigación y docencia; **estimar el nivel de deserción de una IES en un año** como insumo de análisis agregado (por ejemplo, comparar IES, explorar qué contexto se asocia con más deserción, ver cuánto aporta la historia frente a la persistencia).
- **Quién:** analistas y estudiantes de ciencia de datos que conozcan las limitaciones de abajo.

## 3. Uso no previsto (no lo use así)
- **Decisiones sobre personas:** los modelos no ven estudiantes; no sirven para identificar, clasificar ni intervenir a un estudiante.
- **Decisiones sancionatorias o de financiación** de una IES basadas solo en esta predicción: el error es comparable al de repetir la tasa del año anterior y hay subgrupos con error alto (sección 7).
- **Afirmaciones causales:** las variables de contexto se asocian con la deserción; el modelo **no es causal**.
- **Fuera del dominio de entrenamiento:** años distintos de 2015–2021 sin revalidar; IES con **menos de 100 matriculados** en el denominador de la tasa (se excluyeron al construir el dataset: con denominadores chicos la tasa es muy inestable); otros niveles educativos no se evaluaron.

## 4. Factores
- **Relevantes:** carácter académico (universidad, institución universitaria, tecnológica, técnica profesional), sector (oficial/privada), tamaño de la IES, región (Bogotá, Medellín, Cali, resto) y si la IES tiene historia (primer año en SPADIES).
- **Entorno de evaluación:** GroupKFold por IES; particiones temporales (prepandemia y 2020–2021); pliegues por municipio sede (arranque en frío).

## 5. Métricas
MAE (error absoluto medio de la tasa, en puntos de proporción: 0,01 = 1 punto porcentual) con IC 95 % de Nadeau y Bengio; comparación pareada frente a la línea base con la prueba de Wilcoxon y la t corregida (la conclusión «supera» exige que **ambas** sean significativas).

## 6. Datos
- **Entrenamiento:** `data/processed/dataset_ies.csv`, 1.928 filas de 295 IES, 2015–2021 (SHA-256 en `data/checksums.json`). Variables: historia de la IES (`tasa_t1`, `tasa_t2`, `tasa_media_previa`, `n_anios_previos`, `falta_historia`), contexto municipal de la sede (Saber 11 y TerriData en t−1) y matrícula en t−2. Todo con rezago: ninguna variable usa información posterior al año de la fila (verificado: 0 discrepancias en 55.813 valores).
- **Evaluación:** los mismos datos, con validación fuera de muestra por IES (ninguna IES está a la vez en entrenamiento y prueba). Las cifras de abajo **no** son de un conjunto de prueba aparte; los modelos guardados se entrenan con todo el dataset.
- **Pre-procesamiento:** imputación por mediana, estandarización y one-hot ajustados solo con el entrenamiento de cada pliegue.

## 7. Análisis cuantitativo
**MAE con validación anidada** (GroupKFold por IES 5 × 5, semillas 301–305; `reports/04_ajuste_temporal.md`):
| Tarea | Modelo | MAE [IC 95 %] | Línea base | Mejora | p (t corregida; Wilcoxon) |
|---|---|---|---|---|---|
| 1 | `tarea1_ridge` | 0,0245 [0,0212; 0,0278] | ingenua: 0,0256 [0,0225; 0,0288] | +4,4 % | 0,011; < 0,001 |
| 2 | `tarea2_ridge` | 0,0396 [0,0345; 0,0448] | `tarea2_mediana`: 0,0400 [0,0345; 0,0455] | +0,9 % | 0,596; 0,127 |

La confirmación de la Tarea 2 con otras semillas (201–205, sin ajuste anidado) da 0,0405 para la mediana por carácter × sector y 0,0396 para el ridge (+2,2 %, p de la t corregida 0,35): la misma conclusión.

**Robustez:**
- **Tiempo (Tarea 1):** en prepandemia (entrena 2015–2017, prueba 2018–2019) el modelo tiene MAE 0,0214 frente a 0,0226 de la ingenua, y en 2020–2021, 0,0239 frente a 0,0237; en ninguna partición el IC de la diferencia excluye 0. **Tarea 1 no se distingue de repetir la tasa del año anterior en años futuros ni en pandemia.**
- **Tiempo (Tarea 2):** el ridge ajustado entrenado con 2015–2017 **colapsa** al predecir 2018–2019 (MAE 0,0642 frente a 0,0355 de la mediana por carácter × sector; diagnóstico en `reports/06_sesgo_modelos.md`); en 2020–2021 queda en 0,0391 frente a 0,0404. Es una inestabilidad del modelo ajustado con pocos años de entrenamiento.
- **Arranque en frío por municipio (Tarea 2):** ridge 0,0376 frente a 0,0373 de la línea base (−0,6 %, p = 0,92).

**Subgrupos peor servidos** (MAE; `reports/tables/subgrupos.csv`, IC por bootstrap sobre IES, subgrupos con al menos 5 IES):
| Modelo | Subgrupo | Filas / IES | MAE [IC 95 %] | Línea base |
|---|---|---|---|---|
| `tarea1_ridge` | IES sin historia (primer año) | 34 / 29 | 0,114 [0,079; 0,154] | 0,127 |
| `tarea1_ridge` | carácter técnica profesional | 132 / 24 | 0,055 [0,039; 0,074] | 0,060 |
| `tarea1_ridge` | carácter tecnológica | 218 / 37 | 0,054 [0,039; 0,077] | 0,057 |
| `tarea2_ridge` | IES sin historia (primer año) | 34 / 29 | 0,107 [0,070; 0,153] | 0,111 |
| `tarea2_ridge` | carácter tecnológica | 218 / 37 | 0,078 [0,054; 0,117] | 0,075 |
| `tarea2_ridge` | carácter técnica profesional | 132 / 24 | 0,078 [0,061; 0,097] | 0,091 |
| `tarea2_ridge` | tamaño bajo | 643 / 124 | 0,061 [0,050; 0,077] | 0,063 |
| `tarea2_ridge` | Bogotá | 597 / 93 | 0,053 [0,041; 0,068] | 0,053 |

La fila «sin historia» de la Tarea 1 es de la evaluación (que usa la mediana del entrenamiento cuando falta `tasa_t1`): `desercion predecir --tarea 1` rechaza esas filas y pide usar la Tarea 2.

Ningún subgrupo es significativamente peor que su línea base para estos dos modelos (en las tablas completas el único caso con IC que excluye 0 es otro modelo, xgboost con clusters, en Bogotá). Las IES técnicas profesionales, tecnológicas y de tamaño bajo tienen tasas más altas y más variables, y el error es mayor ahí en cualquier modelo.

## 8. Consideraciones éticas y sesgos
- **Selección y supervivencia:** los filtros del dataset sacan 220 de 2.148 IES-año (10,2 %) y 26 de 321 IES (denominador < 100, posible cierre, inconsistencias entre fuentes y régimen especial). Lo que sale tiene tasa mediana de 14,8 % frente a 9,9 % de lo que se queda: **el modelo no ve las IES más frágiles** (`reports/05_sesgo_dataset.md`).
- **Geografía y tamaño:** el contexto es del municipio sede (no de la IES); pocas ciudades concentran gran parte de las filas (Bogotá: 93 IES) y las IES pequeñas aportan pocos datos.
- **Rezagos y fuga:** hay verificación automática de que ninguna explicativa usa un año posterior al permitido; quedan limitaciones que no cubre (atributos institucionales tomados de 2015–2021, selección por posible cierre con información posterior, censo 2018 usado solo desde 2020).
- **Riesgo de uso:** una predicción baja o alta sobre una IES puede estigmatizarla; la incertidumbre es grande (IC de la tabla anterior).

## 9. Limitaciones y recomendaciones
- **Persistencia:** la Tarea 1 es, en esencia, la tasa del año anterior con una pequeña corrección. Su ventaja (0,001 de MAE) es real pero chica; el contexto casi no aporta sobre la historia.
- **Población:** solo IES con al menos 100 matriculados en el denominador de la tasa; no extrapole a instituciones más pequeñas.
- **No es causal:** asociaciones, no efectos.
- **Tarea 2:** sin firmeza estadística sobre la mediana por carácter × sector. Si necesita una estimación sin historia, `tarea2_mediana` es la alternativa razonable (≈ 700 veces más barata de ajustar, ver `reports/07_complejidad.md`) y no tiene el colapso temporal del ridge.
- **Datos:** las fuentes de datos.gov.co se actualizan; una descarga nueva puede cambiar el dataset. Con una sola ventana 2015–2021, los años de pandemia están en el entrenamiento.
- **Entrada de `predecir`:** la Tarea 1 exige `tasa_t1` (proporción entre 0 y 1); si faltan `tasa_t2`, `tasa_media_previa` o `n_anios_previos`, el pipeline las imputa con la mediana del entrenamiento y la predicción es menos precisa. Con `tasa_t1` vacía use la Tarea 2.

## 10. Cómo usar y reentrenar
```
desercion predecir --tarea 1 --entrada historia.csv --salida predicciones.csv
desercion predecir --tarea 2 --entrada contexto.csv --salida predicciones.csv [--modelo mediana]
desercion entrenar            # reentrena los tres modelos con data/processed/dataset_ies.csv y reescribe models/
```
`desercion entrenar` usa los hiperparámetros de consenso de `reports/tables/ajuste.csv`. Para regenerar el dataset y los hiperparámetros desde cero:
`desercion etl --saltar-extract` (necesita los datos crudos, ver `docs/DATOS.md`), `desercion evaluar ajuste` y luego `desercion entrenar`. Después, `desercion verificar` y las pruebas (`python -m pytest -q`).
