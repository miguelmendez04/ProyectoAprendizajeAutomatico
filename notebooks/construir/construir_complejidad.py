"""Genera notebooks/06_complejidad.ipynb y lo ejecuta de punta a punta.

Uso, desde la raíz del proyecto:  python notebooks/construir/construir_complejidad.py
Requiere haber corrido antes `desercion evaluar complejidad` (el cuaderno lee sus CSV y figuras). Las cifras oficiales vienen de esa corrida controlada; el cuaderno solo
repite una demostración corta de la metodología. La sección final de conclusiones se toma de `reports/07_complejidad.md`.
"""

from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

AQUI = Path(__file__).resolve().parents[1]  # carpeta notebooks/
DESTINO = AQUI / "06_complejidad.ipynb"
INFORME = AQUI.parent / "reports" / "07_complejidad.md"


def md(texto: str):
    return nbf.v4.new_markdown_cell(texto.strip("\n"))


def code(texto: str):
    return nbf.v4.new_code_cell(texto.strip("\n"))


def seccion(titulo: str) -> str:
    texto = INFORME.read_text(encoding="utf-8")
    i = texto.index(f"## {titulo}")
    j = texto.find("\n## ", i + 3)
    return texto[i : j if j != -1 else None].replace(f"## {titulo}", "", 1).strip()


celdas = [
    md("""
# Complejidad computacional: teórica y medida

Este cuaderno recorre la sección de complejidad del paper:

1. **Entorno** de la medición (CPU, RAM, versiones) y cómo se controla (un solo hilo, calentamiento, 5 repeticiones, mediana y RIQ).
2. **Complejidad teórica** (notación O) de cada modelo y método.
3. **Costo medido** de ajuste, predicción y memoria sobre las 1.928 filas.
4. **Escalado empírico** con submuestras por IES y el exponente log-log frente al teórico.
5. **Costo del protocolo** de evaluación y **costo frente a error** por tarea.

Las cifras oficiales las produce `desercion evaluar complejidad` (≈ 10 min, con un hilo y sin otra carga) y se leen de `reports/tables/`. Aquí solo se repite una **demostración corta**
de cómo se mide, porque dentro de un cuaderno el kernel y otras celdas pueden alterar los tiempos.
"""),
    code("""
# Configuración: rutas relativas a la raíz del proyecto, librerías y avisos
import os, sys, json, warnings
from pathlib import Path

ROOT = Path.cwd()
while not (ROOT / "configs" / "config.yaml").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display

from desercion_ies.config import load_config
from desercion_ies.analysis import complejidad as cx
from desercion_ies.features.particion import cargar_dataset

pd.set_option("display.width", 230); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 70)
pd.set_option("display.float_format", lambda x: f"{x:.4g}")
cfg = load_config()
R = Path("reports/tables")
df = cargar_dataset(cfg)
print(f"{len(df)} filas, {df.cod_ies.nunique()} IES")
"""),
    md("## 1. Entorno de la medición"),
    code("""
entorno = json.loads((R / "complejidad_entorno.json").read_text(encoding="utf-8"))
display(pd.Series(entorno, name="valor").to_frame())
"""),
    md("""
## 2. Complejidad teórica

n filas, p variables, T árboles, d profundidad, m variables por división, k componentes, E épocas, h neuronas, I iteraciones de Lloyd, c clusters, K_c candidatos de k.
El **exponente teórico en n** (con las demás cantidades fijas) es el que se compara con el medido más abajo.
"""),
    code("""
teo = pd.DataFrame(cx.teoria()).T
teo["exp"] = teo["exp"].map(lambda e: f"{e[0]:g}" if e[0] == e[1] else f"{e[0]:g}–{e[1]:g}")
display(teo.rename(columns={"exp": "exponente teórico en n"}))
"""),
    md("""
## 3. Cómo se mide (demostración corta)

`cx.medir(ajustar, predecir, n_filas, repeticiones, calentamiento)` limita BLAS y OpenMP a un hilo, hace un **calentamiento** y luego las repeticiones; devuelve la mediana y el rango
intercuartílico. La memoria se mide en una ejecución aparte con `tracemalloc` (que enlentece, por eso no se mezcla con los tiempos) y, de forma complementaria, con el RSS muestreado.
"""),
    code("""
pc = {**cx.parametros_complejidad(cfg), "repeticiones": 3, "calentamiento": 1}
rapidos = [c for c in cx.componentes(cfg) if c["nombre"] in ("T1 naive (tasa de t-1)", "T1 C solo historia | ridge", "PCA (90 % varianza)", "K-means (k=6, n_init=10)")]
demo = pd.DataFrame([f for c in rapidos for f in cx.medir_componente(c, df, pc, fracciones=[1.0])])
display(demo[["componente", "n_filas", "ajuste_mediana_s", "ajuste_riq_s", "pred_1000_mediana_ms", "memoria_pico_mib"]])
print("(demostración con 3 repeticiones dentro del cuaderno; las cifras oficiales son las de abajo)")
"""),
    md("""
## 4. Costo medido (corrida oficial)

Un hilo, calentamiento + 5 repeticiones, sobre las 1.928 filas. Ajuste en milisegundos (mediana y RIQ), predicción por 1.000 filas y memoria.
"""),
    code("""
c = pd.read_csv(R / "complejidad.csv")
c["ajuste_ms"] = c.ajuste_mediana_s * 1000; c["riq_ms"] = c.ajuste_riq_s * 1000
display(c[["componente", "tipo", "ajuste_ms", "riq_ms", "pred_1000_mediana_ms", "pred_1000_riq_ms", "memoria_pico_mib", "memoria_rss_mib"]].sort_values("ajuste_ms").reset_index(drop=True))
"""),
    code("""
display(Image("reports/figures/complejidad_costo_error.png"))
"""),
    md("""
## 5. Escalado empírico

Submuestras **por IES** (IES completas, anidadas) del 25, 50, 75 y 100 %. El exponente empírico es la pendiente del tiempo de ajuste contra el número de filas en escala log-log;
**con 4 puntos es aproximado** (se muestra su error estándar). Se compara con el teórico.
"""),
    code("""
esc = pd.read_csv(R / "complejidad_escalado.csv")
print("Filas por fracción de IES:", esc.groupby("fraccion_ies").n_filas.first().to_dict())
exp = pd.read_csv(R / "complejidad_exponentes.csv")
display(exp[["componente", "n_min", "n_max", "t_min_s", "t_max_s", "exponente_empirico", "error_estandar", "r2", "exponente_teorico_min", "exponente_teorico_max", "comparacion"]])
display(Image("reports/figures/complejidad_escalado.png"))
"""),
    code("""
# Extrapolación a 10 y 100 veces más filas con el exponente empírico (orden de magnitud; no es una medición)
n0 = int(c.n_filas.max())
ext = exp[["componente", "exponente_empirico", "t_max_s"]].copy()
for f in (10, 100):
    ext[f"t_{f}x_s"] = ext.t_max_s * f ** ext.exponente_empirico
print(f"Con {n0} filas hoy; la matriz n×n de KernelPCA ocuparía {8 * (10 * n0) ** 2 / 2**30:.1f} GiB con 10× filas y {8 * (100 * n0) ** 2 / 2**30:.0f} GiB con 100×")
display(ext.sort_values("exponente_empirico", ascending=False).reset_index(drop=True))
"""),
    md("""
## 6. Costo del protocolo de evaluación

Número de ajustes de cada etapa (contados en los CSV de resultados) y tiempo de pared de las corridas anteriores.
"""),
    code("""
p = pd.read_csv(R / "complejidad_protocolo.csv")
display(p[["etapa", "configuraciones", "ajustes", "tiempo_pared_s", "paralelismo", "suma_t_ajuste_s"]])
print("Total de ajustes:", int(p.ajustes.sum()), "| tiempo de pared acumulado:", round(p.tiempo_pared_s.sum() / 60, 1), "min")
"""),
    md(
        "## 7. Conclusiones\n\n(Tomadas de `reports/07_complejidad.md`; todas las cifras salen de la corrida.)\n\n__CONCLUSION__"
    ),
]

nb = nbf.v4.new_notebook()
conclusion = (
    seccion("Conclusión") if INFORME.exists() else "Ejecutar antes `desercion evaluar complejidad`."
)
for c in celdas:
    if c.cell_type == "markdown" and "__CONCLUSION__" in c.source:
        c.source = c.source.replace("__CONCLUSION__", conclusion)
nb.cells = celdas
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
ExecutePreprocessor(
    timeout=1200, kernel_name="python3", resources={"metadata": {"path": str(AQUI)}}
).preprocess(nb)
nbf.write(nb, DESTINO)
print("escrito", DESTINO)
