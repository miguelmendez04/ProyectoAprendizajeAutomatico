# Migración a la estructura de paquete (antes → ahora)

El repositorio pasó de scripts en la raíz y un paquete `src/` suelto a un paquete instalable (`desercion_ies`, *src layout*, estilo Cookiecutter Data Science). **No cambió la lógica, las semillas ni los resultados**: los CSV de `reports/tables/` son los mismos de `reports/resultados/` (comprobado en [`REPRODUCIBILIDAD.md`](REPRODUCIBILIDAD.md)) y el SHA-256 de `dataset_ies.csv` no cambió. Los movimientos se hicieron con `git mv`, así que `git log --follow <archivo nuevo>` conserva el historial.

Los textos históricos (`reports/historico/`, bitácora `reports/01_etl.md`) conservan los nombres anteriores de scripts y módulos; esta tabla es su traductor.

## Comandos
| Antes | Ahora |
|---|---|
| `python pipeline.py [--saltar-extract]` | `desercion etl [--saltar-extract]` |
| `python run_experimentos.py` | `desercion evaluar v1` |
| `python run_formulaciones.py` | `desercion evaluar v2` |
| `python run_protocolo_v3.py` | `desercion evaluar tarea1` |
| `python run_tarea2.py [--solo-informe]` | `desercion evaluar tarea2 [--solo-informe]` |
| `python run_ajuste_temporal.py [--reanudar …]` | `desercion evaluar ajuste [--reanudar …]` |
| `python run_sesgo_sensibilidad.py [--reanudar …]` | `desercion evaluar sesgo [--reanudar …]` |
| `python run_complejidad.py [--reanudar …]` | `desercion evaluar complejidad [--reanudar …]` |
| `python reproducir.py [--etapa …] [--rapido] [--reanudar]` | `desercion reproducir [--etapa …] [--rapido] [--reanudar]` |
| `python reproducir.py --solo-verificar` | `desercion verificar` (o `desercion reproducir --solo-verificar`) |
| `python notebooks/construir_*.py` | `python notebooks/construir/construir_*.py` (o `make notebooks`) |
| `pytest` (con `pytest.ini`) | `python -m pytest -q` (configuración en `pyproject.toml`) |

`python -m desercion_ies <comando>` equivale a `desercion <comando>`. Opciones globales nuevas: `-v/--verbose` y `-q/--quiet` (los mensajes van por `logging`; el contenido de los informes no cambió).

## Módulos (`src/…` → `src/desercion_ies/…`)
| Antes | Ahora |
|---|---|
| `src/config.py` | `desercion_ies/config.py` |
| `src/reproducibilidad.py` | `desercion_ies/reproducibilidad.py` |
| (nuevo) | `desercion_ies/paths.py`, `cli.py`, `__main__.py`, `reproduce.py` |
| `pipeline.py` | `desercion_ies/etl/pipeline.py` |
| `src/{extract,transform,homologation,integrate,validate}/` | `desercion_ies/etl/{extract,transform,homologation,integrate,validate}/` |
| `src/modelado/{particion,objetivo}.py` | `desercion_ies/features/` |
| `src/modelado/metodos.py` | `desercion_ies/models/metodos.py` |
| `src/modelado/{protocolo,experimentos,formulaciones,tarea2,ajuste,temporal,arranque_frio,evaluacion_final}.py` | `desercion_ies/evaluation/` |
| `src/modelado/{sesgo,geografia,mitigacion,subgrupos,sensibilidad,interpretabilidad,diagnostico_colapso,complejidad,costo_protocolo}.py` | `desercion_ies/analysis/` |
| `src/modelado/informe_*.py` | `desercion_ies/reporting/informe_*.py` |
| `run_*.py` (lógica de cada etapa) | `desercion_ies/runners/{v1,v2,tarea1,tarea2,ajuste,sesgo,complejidad}.py` (funciones `configurar` y `ejecutar`) |

Imports: `from src.modelado import tarea2` → `from desercion_ies.evaluation import tarea2`; `from src.config import load_config` → `from desercion_ies.config import load_config`.

## Archivos y carpetas
| Antes | Ahora |
|---|---|
| `config/config.yaml` | `configs/config.yaml` |
| `pytest.ini` | `[tool.pytest.ini_options]` en `pyproject.toml` |
| `requirements.txt` (a mano) | generado desde `pyproject.toml` (`make requirements`) |
| `guia_etl_desercion.md` | `docs/guia_etl.md` |
| `tests/test_*.py` | `tests/unit/test_*.py`; las que necesitan `data/processed`, `data/interim` o los resultados publicados: `tests/integration/test_integracion_*.py` (`@pytest.mark.integration`) |

## Informes, tablas y figuras
| Antes | Ahora |
|---|---|
| `reports/etl_log.md` (bitácora) | `reports/01_etl.md` |
| `reports/resultados_v3.md` | `reports/02_tarea1.md` |
| `reports/resultados_tarea2.md` | `reports/03_tarea2.md` |
| `reports/resultados_ajuste_temporal.md` | `reports/04_ajuste_temporal.md` |
| `reports/analisis_sesgo.md` | `reports/05_sesgo_dataset.md` |
| `reports/resultados_sesgo_sensibilidad.md` | `reports/06_sesgo_modelos.md` |
| `reports/resultados_complejidad.md` | `reports/07_complejidad.md` |
| `reports/resultados_v1.md`, `resultados_v2.md` | `reports/historico/resultados_v1.md`, `resultados_v2.md` |
| `reports/{data_dictionary,diagrama_pipeline,fase0}.md` | `reports/etl/…` |
| `reports/{exclusiones_calidad,tasas_altas,sin_historia}.csv` | `reports/etl/…` |
| `reports/resultados/*.csv`, `*.json` | `reports/tables/…` (mismos nombres) |
| `reports/figuras/*.png` | `reports/figures/…` (mismos nombres) |

## Cuadernos
| Antes | Ahora |
|---|---|
| `01_eda.ipynb` | `01_eda.ipynb` |
| `02_sesgo.ipynb` | `02_sesgo.ipynb` |
| `04_tarea2.ipynb` | `03_tarea2.ipynb` |
| `05_ajuste_temporal.ipynb` | `04_ajuste_temporal.ipynb` |
| `06_sesgo_interpretabilidad.ipynb` | `05_sesgo_interpretabilidad.ipynb` |
| `07_complejidad.ipynb` | `06_complejidad.ipynb` |
| `notebooks/construir_*.py` | `notebooks/construir/construir_*.py` |

## Decisiones de ubicación que difieren de lo esperado
- **`reports/01_etl.md` es la bitácora del ETL** (`etl_log.md`); sus anexos generados (`data_dictionary.md`, `exclusiones_calidad.csv`, `tasas_altas.csv`, `sin_historia.csv`) y los manuales (`diagrama_pipeline.md`, `fase0.md`) están en `reports/etl/`. El `pipeline` escribe ahora en esas rutas.
- **Constructores de cuadernos** en `notebooks/construir/` (en lugar de una función del paquete): son scripts que generan y ejecutan el cuaderno, y dependen de la carpeta `notebooks/`.
- **El comando cambia al directorio raíz del proyecto** al arrancar: las rutas de `configs/config.yaml` son relativas a esa raíz. `paths.py` detecta la raíz desde la ubicación del paquete (o `DESERCION_ROOT`), no desde el directorio actual; las rutas pasadas con `--config` o `--salida` se interpretan respecto de donde se invocó el comando.
- **Una prueba y sus datos:** `tests/unit/fixtures/` (las muestras de Saber 11 y SPADIES) viven junto a las pruebas unitarias.
