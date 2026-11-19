# Verificación de reproducibilidad

Fecha: 2026-11-18. Entorno: Windows 11, Python 3.14.6, Intel Core i5-12500H (16 hilos), versiones de `requirements.txt`.

## Qué se verificó
1. **ETL desde los crudos.** `python reproducir.py --etapa etl --rapido` (14 archivos crudos con SHA-256 verificado) genera un `dataset_ies.csv` **byte a byte idéntico** al publicado (mismo SHA-256, 1.928 filas, 295 IES). Tarda ~104 s con los crudos ya descargados.
2. **Modelado desde un clon limpio.** `git clone` del commit a una carpeta nueva (sin `data/raw/`, solo el dataset versionado) y `python reproducir.py --etapa todo --rapido`: **381 s** (6,3 min). `etl` y `eda` se omiten con aviso, porque necesitan los crudos. En la máquina original, con crudos, la misma orden tardó 585 s (9,8 min).
3. **Cuadernos.** Los seis (01, 02, 04–07) se ejecutan de punta a punta sin errores con `jupyter nbconvert --execute` (01 y 02 requieren los crudos y el ETL; 04–07 también se ejecutaron en el clon limpio).
4. **Pruebas.** `python -m pytest -q`: 399 pasan.

## Comparación del modo `--rapido` con los CSV publicados
Igualdad **exacta** (diferencia absoluta 0) de MAE, RMSE, R² y tamaños, fila a fila, para las claves (configuración, semilla, pliegue…) que existen en ambos. El modo rápido usa un subconjunto de semillas y configuraciones, por eso solo hay filas comunes con una parte de lo publicado.

| Archivo | Filas comunes | Idénticas | Resultado |
|---|---|---|---|
| `experimentos_v1` | 30 | 30 | coincide |
| `experimentos_v2` (A contexto) | 10 | 10 | coincide |
| `experimentos_v3` | 70 | 70 | coincide |
| `experimentos_tarea2_cribado` | 160 | 160 | coincide |
| `experimentos_tarea2_confirmacion` | 40 | 40 | coincide |
| `ajuste` (hiperparámetros por defecto) | 35 | 35 | coincide |
| `arranque_frio_municipio` | 40 | 40 | coincide |
| `sensibilidad_pliegues` (líneas base y modelos por defecto) | 125 | 125 | coincide |
| `experimentos_v2` (C solo historia) | 10 | 0 | **difiere (esperado)** |
| `ajuste` (hiperparámetros ajustados) | 35 | 3 | **difiere (esperado)** |
| `sensibilidad_pliegues` (modelos «ajustado») | 100 | 0 | **difiere (esperado)** |

## Diferencias y su causa
- **v2, formulaciones con historia (C solo historia).** El `experimentos_v2.csv` publicado se calculó con la historia que empieza en 2015 (commit 422e3a1); desde la v3 la historia usa también las tasas de SPADIES de 1999 a 2014 y las columnas del dataset cambiaron. Volver a correr `run_formulaciones.py` da otras cifras (diferencia máxima 0,263 de MAE por pliegue). Estaba documentado en la nota de `reports/resultados_v2.md`; la referencia vigente es la v3. La formulación A (sin historia) sí coincide exactamente.
- **Hiperparámetros «ajustados».** El modo rápido hace la búsqueda interna con `n_iter` = 3 en lugar de 30 y con una semilla externa: otro conjunto de hiperparámetros, otro error. Los modelos con hiperparámetros por defecto y las líneas base coinciden exactamente. Los modelos «ajustado» de la sensibilidad usan los hiperparámetros de ese ajuste rápido.
- **Tiempos y complejidad.** Los tiempos medidos (`t_ajuste_s`, `complejidad*.csv`) dependen de la máquina y de la carga: nunca coinciden. En el modo rápido el componente más lento medía ~3,4 s frente a ~2,4 s publicados (autoencoder), por medirse con 2 repeticiones y sin las condiciones controladas de la corrida oficial.
- **Paralelismo.** Las etapas de Tarea 2, ajuste, sesgo corren con 16 procesos (`n_jobs: -1`), pero cada modelo usa un hilo y las particiones y semillas están fijas: los resultados por pliegue no cambian con el paralelismo (comprobado por la igualdad exacta de arriba).

## Qué no se verificó
- La corrida **completa** (`--etapa todo` sin `--rapido`, ≈ 1,5 h) no se repitió desde el clon: sus resultados son los publicados en `reports/`. La verificación rápida cubre un subconjunto exacto de ellos.
- Otras plataformas (Linux/macOS) y otras versiones de librerías: los resultados se generaron con las versiones de `requirements.txt` en Windows 11; cambios de versión de scikit-learn o xgboost pueden alterar cifras en el último decimal o más.
- Una nueva descarga de datos.gov.co puede traer datos distintos (las fuentes se actualizan): por eso se fijan los SHA-256.
