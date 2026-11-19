# Resultados v1: primera tabla de experimentos

Generado por `python run_experimentos.py` (motor en `src/modelado/experimentos.py`). Todas las cifras salen de la corrida; no se editan a mano.

## Diseño

- **Datos:** `data/processed/dataset_ies.csv`, 1.928 filas y 295 IES (ventana 2015–2021); 33 variables numéricas y 2 categóricas (sector y carácter), sin las excluidas por defecto ni `NUNCA_EXPLICATIVAS`.
- **Validación:** GroupKFold por `cod_ies`, 5 pliegues, **los mismos para las 36 configuraciones y las dos líneas base** (la huella de las IES de prueba de cada pliegue está en `hash_prueba`). Preprocesamiento, selección y modelo se ajustan solo con el entrenamiento de cada pliegue.
- **Grilla (36 configuraciones, hiperparámetros por defecto):** objetivo (ninguna | logit) × imputación (mediana | KNN, k = 5) × selección (ninguna | información mutua con k = 15 | PCA con 90 % de varianza) × modelo (RidgeCV con alfas logspace(-3, 3, 13) | Random Forest de 300 árboles | XGBoost de 300 árboles, tasa de aprendizaje 0.05, profundidad 4).
- **Métricas:** MAE, RMSE y R² por pliegue, siempre en escala de tasa (0–1), aunque se entrene en logit; en la tabla, media ± desviación entre pliegues. Las predicciones no se recortan a [0, 1].
- **Líneas base:** `base_mediana` (mediana de la tasa del entrenamiento) y `base_ingenua` (tasa del año anterior de la misma IES; si no hay fila t-1, la mediana del entrenamiento).
- **Semilla:** 42. scikit-learn 1.9.0, xgboost 3.4.1. Tiempo total de la corrida: 337 s (5,6 min).

## Tabla de resultados (ordenada por MAE; líneas base arriba)

La mejora es el cambio porcentual del MAE medio frente a la línea base ingenua (positiva = menor error que la ingenua). «Pliegues mejores» compara cada pliegue con el de la ingenua.

| Configuración (modelo \| objetivo \| imputación \| selección) | MAE | RMSE | R² | Mejora vs. ingenua | Pliegues mejores que la ingenua | Ajuste (s) |
|---|---|---|---|---|---|---|
| **base_ingenua** | 0,0294 ± 0,0046 | 0,0614 ± 0,0131 | 0,454 ± 0,080 | — | — | 0,00 |
| **base_mediana** | 0,0501 ± 0,0085 | 0,0870 ± 0,0244 | -0,065 ± 0,035 | -70,2 % | 0 de 5 | 0,00 |
| ridge | logit | mediana | info_mutua | 0,0392 ± 0,0050 | 0,0690 ± 0,0198 | 0,332 ± 0,065 | -33,4 % | 0 de 5 | 0,38 |
| ridge | logit | knn | info_mutua | 0,0393 ± 0,0050 | 0,0691 ± 0,0197 | 0,331 ± 0,065 | -33,4 % | 0 de 5 | 0,55 |
| ridge | logit | mediana | ninguna | 0,0399 ± 0,0048 | 0,0694 ± 0,0198 | 0,324 ± 0,075 | -35,5 % | 0 de 5 | 0,04 |
| ridge | logit | knn | ninguna | 0,0399 ± 0,0048 | 0,0696 ± 0,0194 | 0,318 ± 0,071 | -35,8 % | 0 de 5 | 0,19 |
| ridge | ninguna | mediana | info_mutua | 0,0412 ± 0,0039 | 0,0694 ± 0,0180 | 0,317 ± 0,038 | -40,0 % | 0 de 5 | 0,42 |
| ridge | ninguna | knn | info_mutua | 0,0413 ± 0,0041 | 0,0694 ± 0,0181 | 0,318 ± 0,036 | -40,4 % | 0 de 5 | 0,58 |
| ridge | ninguna | knn | ninguna | 0,0421 ± 0,0035 | 0,0699 ± 0,0176 | 0,305 ± 0,046 | -43,1 % | 0 de 5 | 0,19 |
| ridge | ninguna | mediana | ninguna | 0,0421 ± 0,0032 | 0,0698 ± 0,0176 | 0,308 ± 0,043 | -43,2 % | 0 de 5 | 0,04 |
| xgboost | logit | knn | ninguna | 0,0447 ± 0,0038 | 0,0821 ± 0,0145 | -0,066 ± 0,506 | -52,0 % | 0 de 5 | 0,70 |
| xgboost | logit | mediana | ninguna | 0,0449 ± 0,0049 | 0,0827 ± 0,0156 | -0,066 ± 0,473 | -52,8 % | 0 de 5 | 0,50 |
| xgboost | logit | mediana | info_mutua | 0,0451 ± 0,0045 | 0,0827 ± 0,0145 | -0,083 ± 0,521 | -53,3 % | 0 de 5 | 0,64 |
| ridge | logit | mediana | pca | 0,0452 ± 0,0049 | 0,0785 ± 0,0202 | 0,120 ± 0,085 | -53,7 % | 0 de 5 | 0,04 |
| ridge | logit | knn | pca | 0,0453 ± 0,0049 | 0,0787 ± 0,0202 | 0,115 ± 0,087 | -54,1 % | 0 de 5 | 0,18 |
| xgboost | logit | knn | info_mutua | 0,0456 ± 0,0049 | 0,0845 ± 0,0149 | -0,130 ± 0,542 | -55,1 % | 0 de 5 | 0,78 |
| xgboost | ninguna | mediana | info_mutua | 0,0457 ± 0,0055 | 0,0787 ± 0,0136 | 0,055 ± 0,333 | -55,2 % | 0 de 5 | 0,63 |
| xgboost | ninguna | knn | info_mutua | 0,0462 ± 0,0056 | 0,0796 ± 0,0147 | 0,035 ± 0,336 | -57,2 % | 0 de 5 | 0,78 |
| xgboost | ninguna | mediana | ninguna | 0,0464 ± 0,0048 | 0,0798 ± 0,0142 | 0,031 ± 0,334 | -57,9 % | 0 de 5 | 0,58 |
| random_forest | logit | mediana | ninguna | 0,0465 ± 0,0074 | 0,0815 ± 0,0220 | 0,019 ± 0,398 | -58,0 % | 0 de 5 | 5,62 |
| xgboost | ninguna | knn | ninguna | 0,0466 ± 0,0043 | 0,0806 ± 0,0126 | -0,009 ± 0,418 | -58,5 % | 0 de 5 | 0,70 |
| random_forest | logit | knn | ninguna | 0,0467 ± 0,0075 | 0,0818 ± 0,0222 | 0,008 ± 0,414 | -58,6 % | 0 de 5 | 6,76 |
| random_forest | logit | mediana | info_mutua | 0,0471 ± 0,0081 | 0,0808 ± 0,0238 | 0,041 ± 0,425 | -60,0 % | 0 de 5 | 2,95 |
| random_forest | logit | knn | info_mutua | 0,0474 ± 0,0078 | 0,0814 ± 0,0237 | 0,026 ± 0,422 | -61,2 % | 0 de 5 | 3,30 |
| ridge | ninguna | mediana | pca | 0,0478 ± 0,0039 | 0,0781 ± 0,0183 | 0,117 ± 0,137 | -62,6 % | 0 de 5 | 0,03 |
| ridge | ninguna | knn | pca | 0,0480 ± 0,0039 | 0,0783 ± 0,0183 | 0,114 ± 0,137 | -63,0 % | 0 de 5 | 0,17 |
| random_forest | ninguna | mediana | ninguna | 0,0482 ± 0,0075 | 0,0809 ± 0,0198 | 0,040 ± 0,301 | -63,9 % | 0 de 5 | 6,02 |
| random_forest | ninguna | knn | ninguna | 0,0485 ± 0,0076 | 0,0813 ± 0,0199 | 0,029 ± 0,316 | -64,9 % | 0 de 5 | 6,98 |
| random_forest | ninguna | mediana | info_mutua | 0,0486 ± 0,0071 | 0,0807 ± 0,0193 | 0,039 ± 0,319 | -65,1 % | 0 de 5 | 3,06 |
| random_forest | ninguna | knn | info_mutua | 0,0490 ± 0,0067 | 0,0817 ± 0,0192 | 0,015 ± 0,318 | -66,7 % | 0 de 5 | 3,39 |
| xgboost | logit | mediana | pca | 0,0491 ± 0,0055 | 0,0868 ± 0,0188 | -0,125 ± 0,352 | -67,1 % | 0 de 5 | 0,39 |
| xgboost | logit | knn | pca | 0,0499 ± 0,0037 | 0,0864 ± 0,0157 | -0,122 ± 0,345 | -69,7 % | 0 de 5 | 0,54 |
| xgboost | ninguna | mediana | pca | 0,0509 ± 0,0050 | 0,0854 ± 0,0174 | -0,091 ± 0,331 | -73,0 % | 0 de 5 | 0,43 |
| random_forest | logit | mediana | pca | 0,0512 ± 0,0055 | 0,0882 ± 0,0179 | -0,169 ± 0,374 | -74,1 % | 0 de 5 | 4,31 |
| random_forest | logit | knn | pca | 0,0515 ± 0,0047 | 0,0883 ± 0,0167 | -0,173 ± 0,362 | -75,2 % | 0 de 5 | 4,15 |
| xgboost | ninguna | knn | pca | 0,0517 ± 0,0034 | 0,0859 ± 0,0164 | -0,095 ± 0,268 | -75,7 % | 0 de 5 | 0,53 |
| random_forest | ninguna | mediana | pca | 0,0528 ± 0,0040 | 0,0865 ± 0,0166 | -0,123 ± 0,338 | -79,6 % | 0 de 5 | 4,50 |
| random_forest | ninguna | knn | pca | 0,0531 ± 0,0031 | 0,0873 ± 0,0154 | -0,145 ± 0,338 | -80,4 % | 0 de 5 | 4,40 |

![MAE por configuración](figures/mae_configuraciones.png)

## Análisis

- **Mejor configuración:** `ridge | logit | mediana | info_mutua`, con MAE 0,0392 ± 0,0050 (-33,4 % frente a la ingenua; mejor que ella en 0 de 5 pliegues).
- **Frente a la línea base ingenua:** ninguna de las 36 configuraciones tiene MAE medio menor que el de la ingenua (0,0294); la mejor llega a 0,0392.
- **Frente a la mediana:** 30 de 36 configuraciones tienen MAE medio menor que el de la mediana (0,0501); la mejora de la mejor configuración sobre la mediana es +21,6 %.
- **Las dos líneas base:** la ingenua (0,0294) mejora a la mediana (0,0501) en +41,2 %: la tasa de una IES persiste de un año al siguiente.
- **Cobertura de la ingenua:** usó la tasa t-1 de la misma IES en 1623 filas de prueba y la mediana del entrenamiento en 305 (de 1928); la mediana aplica en 2015 y cuando la IES no tiene fila del año anterior en el dataset.
- **Modelo:** por MAE medio de sus configuraciones, el orden es ridge (0,0426), xgboost (0,0472), random_forest (0,0492).
- **Objetivo:** logit (0,0455), ninguna (0,0472); la diferencia entre el mejor y el peor nivel es 0,0018 de MAE.
- **Imputación:** mediana (0,0462), knn (0,0465); la diferencia entre el mejor y el peor nivel es 0,0003 de MAE.
- **Selección / reducción:** info_mutua (0,0446), ninguna (0,0447), pca (0,0497); la diferencia entre el mejor y el peor nivel es 0,0051 de MAE.
- **Mejor combinación por modelo:** `ridge | logit | mediana | info_mutua` (0,0392); `xgboost | logit | knn | ninguna` (0,0447); `random_forest | logit | mediana | ninguna` (0,0465).

### Efecto de cada decisión (promedio sobre las demás)

| Modelo | MAE medio de sus configuraciones | Mejor MAE | Ajuste medio (s) | Configuraciones |
|---|---|---|---|---|
| ridge | 0,0426 | 0,0392 | 0,23 | 12 |
| xgboost | 0,0472 | 0,0447 | 0,60 | 12 |
| random_forest | 0,0492 | 0,0465 | 4,62 | 12 |

| Objetivo | MAE medio de sus configuraciones | Mejor MAE | Ajuste medio (s) | Configuraciones |
|---|---|---|---|---|
| logit | 0,0455 | 0,0392 | 1,78 | 18 |
| ninguna | 0,0472 | 0,0412 | 1,86 | 18 |

| Imputación | MAE medio de sus configuraciones | Mejor MAE | Ajuste medio (s) | Configuraciones |
|---|---|---|---|---|
| mediana | 0,0462 | 0,0392 | 1,70 | 18 |
| knn | 0,0465 | 0,0393 | 1,94 | 18 |

| Selección / reducción | MAE medio de sus configuraciones | Mejor MAE | Ajuste medio (s) | Configuraciones |
|---|---|---|---|---|
| info_mutua | 0,0446 | 0,0392 | 1,45 | 12 |
| ninguna | 0,0447 | 0,0399 | 2,36 | 12 |
| pca | 0,0497 | 0,0452 | 1,64 | 12 |

## Limitaciones de esta primera tabla

- Hiperparámetros por defecto: es una comparación de configuraciones, no el modelo final. El ajuste viene después.
- `RidgeCV` elige su alfa con validación cruzada interna sin agrupar por IES (dentro del entrenamiento de cada pliegue), así que puede elegir un alfa algo pequeño; no hay fuga hacia el pliegue de prueba.
- La línea base ingenua usa la tasa observada del año anterior de la IES, que es un dato disponible al predecir el año t; los modelos de la grilla no la reciben como variable. Es la comparación que dice cuánto aporta el contexto territorial frente a la persistencia de cada IES.
- Con 5 pliegues, las diferencias menores que la desviación entre pliegues no son concluyentes.
- Una sola partición agrupada (GroupKFold determinista): no se repite con otras semillas de pliegues.
- El sesgo de selección, supervivencia y geografía descrito en `reports/05_sesgo_dataset.md` aplica igual a estos resultados.
