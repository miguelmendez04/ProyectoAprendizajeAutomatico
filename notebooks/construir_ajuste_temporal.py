"""Genera notebooks/05_ajuste_temporal.ipynb y lo ejecuta de punta a punta.

Uso, desde la raíz del proyecto:  python notebooks/construir_ajuste_temporal.py
Requiere haber corrido antes `python run_ajuste_temporal.py` (el cuaderno lee sus CSV y figuras; no repite la validación anidada completa).
La sección final de conclusiones se toma de `reports/resultados_ajuste_temporal.md`.
"""
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

AQUI = Path(__file__).resolve().parent
DESTINO = AQUI / "05_ajuste_temporal.ipynb"
INFORME = AQUI.parent / "reports" / "resultados_ajuste_temporal.md"


def md(texto: str):
    return nbf.v4.new_markdown_cell(texto.strip("\n"))


def code(texto: str):
    return nbf.v4.new_code_cell(texto.strip("\n"))


def seccion(titulo: str) -> str:
    texto = INFORME.read_text(encoding="utf-8")
    i = texto.index(f"## {titulo}")
    j = texto.find("\n## ", i + 3)
    return texto[i:j if j != -1 else None].replace(f"## {titulo}", "", 1).strip()


celdas = [
    md("""
# Ajuste de hiperparámetros, particiones temporales y arranque en frío estricto

Antes de cerrar la conclusión de que ningún modelo supera con firmeza a su línea base hay que descartar dos objeciones:

1. **¿Rinden poco por usar hiperparámetros por defecto?** → ajuste con **validación anidada** (sin fuga).
2. **¿La validación por IES representa el uso real?** → **particiones temporales** (años futuros y pandemia) y **arranque en frío estricto**
   (pliegues por municipio, para que ninguna ciudad de prueba esté en el entrenamiento).

Los cálculos pesados los hace `python run_ajuste_temporal.py` y se guardan en `reports/resultados/`. Aquí se leen y se muestran; lo único que se
calcula en el cuaderno es una **ilustración** de la validación anidada sobre un pliegue.
"""),
    code("""
# Configuración: rutas relativas a la raíz del proyecto, librerías y avisos
import os, sys, json, warnings
from pathlib import Path

ROOT = Path.cwd()
while not (ROOT / "config" / "config.yaml").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display

from src.config import load_config
from src.modelado import ajuste as aj, arranque_frio as af, tarea2 as t2, temporal as tp
from src.modelado.experimentos import parametros_experimentos
from src.modelado.particion import cargar_dataset
from src.modelado.protocolo import particiones_repetidas

pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 70)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")
cfg = load_config()
p = aj.parametros_ajuste(cfg)
R = Path("reports/resultados")
print("Semillas externas:", p["semillas_externas"], "| búsqueda interna:", p["n_iter"], "iteraciones, GroupKFold de", p["n_splits_interno"], "pliegues")
"""),
    md("""
## 1. Validación anidada: cómo funciona (ilustrado en un pliegue)

- **Externa:** GroupKFold por IES. El pliegue de prueba externo **no se toca** hasta la medición final.
- **Interna:** sobre el entrenamiento externo, una búsqueda aleatoria de 30 iteraciones evalúa cada combinación con GroupKFold por IES de 3 pliegues.
- El mejor modelo se reajusta con todo el entrenamiento externo y se mide en la prueba externa. Se compara con el de hiperparámetros por defecto en el mismo pliegue.

Ilustración con la configuración más rápida de la Tarea 1 (C solo historia, xgboost):
"""),
    code("""
# Un pliegue externo (semilla 301) y la configuración T1 C solo historia | xgboost
df = cargar_dataset(cfg)
y, grupos = df["tasa_desercion"].astype(float), df["cod_ies"].astype(str).to_numpy()
spec = next(s for s in aj.configuraciones_ajuste(cfg) if s["nombre"] == "T1 C solo historia | xgboost")
X, numericas, categoricas = aj.preparar(spec, cfg, df)
entrena, prueba = particiones_repetidas(df, p["n_splits_externo"], [p["semillas_externas"][0]])[0][0]
print(f"Entrenamiento externo: {len(entrena)} filas, {len(set(grupos[entrena]))} IES | prueba externa: {len(prueba)} filas, {len(set(grupos[prueba]))} IES")
print("IES compartidas entre entrenamiento y prueba externos:", len(set(grupos[entrena]) & set(grupos[prueba])))

# La búsqueda recibe SOLO el entrenamiento externo (y sus IES)
p_exp, p_t2 = parametros_experimentos(cfg), t2.parametros_tarea2(cfg)
est, dist = aj.estimador_ajustable(aj.construir_estimador(spec, numericas, categoricas, p_exp, p_t2), spec, p["k_info_mutua"])
busqueda, t_busqueda = aj.buscar(est, dist, X.iloc[entrena], y.iloc[entrena].to_numpy(), grupos[entrena], p)
print(f"Búsqueda interna: {t_busqueda:.1f} s; filas vistas por la búsqueda: {len(busqueda.cv_results_['params'])} combinaciones x {p['n_splits_interno']} pliegues internos")
"""),
    code("""
# Las 5 mejores combinaciones de la búsqueda interna (MAE interno) y el resultado en la prueba externa
cv = pd.DataFrame(busqueda.cv_results_)
cv["mae_interno"] = -cv["mean_test_score"]
cols = [c for c in cv.columns if c.startswith("param_")] + ["mae_interno"]
display(cv.sort_values("mae_interno")[cols].head(5).rename(columns=lambda c: c.replace("param_regressor__modelo__", "").replace("param_modelo__", "")))
d, a = aj.ajustar_y_evaluar(spec, X, y, grupos, entrena, prueba, numericas, categoricas, p_exp, p_t2, p)
real = y.iloc[prueba].to_numpy()
print(f"MAE en la prueba externa: por defecto {np.mean(np.abs(d['pred'] - real)):.4f} | ajustado {np.mean(np.abs(a['pred'] - real)):.4f}")
print("Hiperparámetros elegidos:", a["hiperparametros"])
"""),
    md("""
## 2. Por defecto frente a ajustado (25 ajustes externos por configuración)

Tabla completa en el informe. «Mejora por el ajuste» = MAE por defecto − MAE ajustado. La línea base es la ingenua (Tarea 1) o la mediana por
carácter × sector (Tarea 2). IC 95 % y «t corregida» de Nadeau y Bengio.
"""),
    code("""
r = pd.read_csv(R / "ajuste_resumen.csv")
cols = ["configuracion", "variante", "mae_media", "mae_ic_inf", "mae_ic_sup", "mejora_base_pct", "p_wilcoxon_base", "p_t_base", "gana_base"]
for tarea, titulo in (("T1", "Tarea 1 (línea base: ingenua)"), ("T2", "Tarea 2 (línea base: mediana por carácter × sector)")):
    print(titulo)
    display(r[(r.tarea == tarea)][cols].reset_index(drop=True))
display(Image("reports/figuras/ajuste_vs_defecto.png"))
"""),
    code("""
# Hiperparámetros más elegidos en los pliegues externos y tiempos de ajuste/predicción/búsqueda (para el análisis de complejidad)
display(pd.read_csv(R / "ajuste_hiperparametros.csv"))
display(r[r.variante != "linea_base"][["configuracion", "variante", "t_ajuste_media_s", "t_prediccion_media_s", "t_busqueda_media_s"]].reset_index(drop=True))
"""),
    md("""
## 3. Particiones temporales

- **Prepandemia:** entrena 2015–2017, prueba 2018–2019. **Estrés:** entrena 2015–2019, prueba 2020–2021.
- En cada partición se re-ajusta con la búsqueda interna **solo con los años de entrenamiento**.
- **Tarea 1 es un pronóstico a un paso:** para predecir t se usa la tasa observada de t-1, aunque t-1 esté en el período de prueba (en la práctica está disponible).
- IC 95 % por bootstrap sobre IES (1.000 réplicas, semilla fija); las diferencias con la línea base son pareadas.
"""),
    code("""
tm = pd.read_csv(R / "temporal.csv")
cols = ["particion", "tarea", "configuracion", "variante", "mae", "mae_ic_inf", "mae_ic_sup", "rmse", "r2", "dif_ref", "dif_ref_ic_inf", "dif_ref_ic_sup", "prop_gana_ref"]
display(tm[tm.variante.isin(["ajustado", "linea_base"])][cols].reset_index(drop=True))
display(Image("reports/figuras/temporal_por_anio.png"))
"""),
    code("""
# Degradación en 2020–2021: razón entre el MAE del estrés y el de la prepandemia (los entrenamientos son distintos)
g = tm[tm.variante.isin(["ajustado", "linea_base"])]
a, b = g[g.particion == "prepandemia"].set_index("configuracion"), g[g.particion == "estres"].set_index("configuracion")
degr = pd.DataFrame({"tarea": a.tarea, "mae_prepandemia": a.mae, "mae_estres": b.mae, "razon": b.mae / a.mae}).sort_values(["tarea", "razon"], ascending=[True, False])
display(degr)
"""),
    code("""
# Error por sector, carácter y año de prueba (MAE) de los modelos ajustados y las líneas base
gr = pd.read_csv(R / "temporal_grupos.csv")
gr = gr[gr.variante.isin(["ajustado", "linea_base"])]
for (part, tarea), x in gr.groupby(["particion", "tarea"]):
    print(f"--- {part} · {tarea}")
    display(x.pivot_table(index="configuracion", columns=["variable", "nivel"], values="mae"))
"""),
    md("""
## 4. Arranque en frío estricto (Tarea 2): pliegues por municipio

Se repite la confirmación 5 × 5 con GroupKFold por municipio sede: **ninguna ciudad de prueba aparece en el entrenamiento**. Bogotá concentra mucha
de las filas, por lo que los pliegues quedan desbalanceados.
"""),
    code("""
tam = pd.read_csv(R / "arranque_frio_pliegues.csv")
fig, ax = plt.subplots(figsize=(10, 3.5))
colores = np.where(tam.contiene_bogota, "#e45756", "#4c78a8")
ax.bar(range(len(tam)), tam.n_prueba, color=colores)
ax.set_xticks(range(len(tam)), [f"{r}-{k}" for r, k in zip(tam.repeticion, tam.pliegue)], fontsize=7)
ax.set_xlabel("repetición-pliegue"); ax.set_ylabel("filas de prueba"); ax.set_title("Tamaño de los pliegues de prueba por municipio (rojo = contiene a Bogotá)")
plt.show()
display(tam.groupby("contiene_bogota").n_prueba.describe())
"""),
    code("""
# Resultados por municipio frente a la confirmación por IES de la Tarea 2
frio = pd.read_csv(R / "arranque_frio_municipio_resumen.csv")
ies = pd.read_csv(p["confirmacion_tarea2"]).set_index("configuracion")
tabla = frio.assign(mae_por_ies=frio.configuracion.map(ies.mae_media), mejora_por_ies_pct=frio.configuracion.map(ies.mejora_grupo_pct))
display(tabla[["configuracion", "mae_media", "mae_ic_inf", "mae_ic_sup", "mae_por_ies", "mejora_grupo_pct", "mejora_por_ies_pct", "p_wilcoxon_grupo", "p_t_grupo"]].sort_values("mae_media").reset_index(drop=True))
"""),
    md("## 5. Conclusión\n\n(Tomada de `reports/resultados_ajuste_temporal.md`; todas las cifras salen de la corrida.)\n\n__CONCLUSION__"),
]

nb = nbf.v4.new_notebook()
conclusion = seccion("Conclusión") if INFORME.exists() else "Ejecutar antes `python run_ajuste_temporal.py`."
for c in celdas:
    if c.cell_type == "markdown" and "__CONCLUSION__" in c.source:
        c.source = c.source.replace("__CONCLUSION__", conclusion)
nb.cells = celdas
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
ExecutePreprocessor(timeout=1200, kernel_name="python3", resources={"metadata": {"path": str(AQUI)}}).preprocess(nb)
nbf.write(nb, DESTINO)
print("escrito", DESTINO)
