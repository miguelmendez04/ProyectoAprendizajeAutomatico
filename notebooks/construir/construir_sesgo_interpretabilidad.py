"""Genera notebooks/05_sesgo_interpretabilidad.ipynb y lo ejecuta de punta a punta.

Uso, desde la raíz del proyecto:  python notebooks/construir/construir_sesgo_interpretabilidad.py
Requiere haber corrido antes `desercion evaluar sesgo` (el cuaderno lee sus CSV y figuras; recalcula solo lo corto: una verificación del error por subgrupo
y el diagnóstico del colapso). La sección final de conclusiones se toma de `reports/06_sesgo_modelos.md`.
"""
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

AQUI = Path(__file__).resolve().parents[1]  # carpeta notebooks/
DESTINO = AQUI / "05_sesgo_interpretabilidad.ipynb"
INFORME = AQUI.parent / "reports" / "06_sesgo_modelos.md"


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
# Sesgo medido en los modelos finales e interpretabilidad

En `reports/05_sesgo_dataset.md` se declararon sesgos (selección, geografía, carácter…). Aquí se **miden en los modelos finales** y se interpreta lo que aprenden:

1. **Error por subgrupo** (sector, carácter, región, SENA, tamaño, historia) frente a la línea base de cada tarea, con IC 95 % por bootstrap sobre IES.
2. **Sensibilidad**: ¿cambian las conclusiones sin Bogotá, con denominador mínimo 200, sin SENA, o solo con 2017–2021 (SNIES observado)?
3. **Diagnóstico reproducible del colapso** del ridge ajustado de la Tarea 2 en la prepandemia.
4. **Interpretabilidad**: SHAP, importancia por permutación y estabilidad de los coeficientes del ridge.

Los cálculos pesados los hace `desercion evaluar sesgo` (≈2 min) y se guardan en `reports/tables/`; aquí se leen y se muestran.
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
from desercion_ies.evaluation import evaluacion_final as ef
from desercion_ies.analysis import subgrupos as sg
from desercion_ies.analysis import diagnostico_colapso as dc
from desercion_ies.features.particion import cargar_dataset

pd.set_option("display.width", 230); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 60)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")
cfg = load_config()
p = cfg["modelado"]["sesgo_sensibilidad"]
R = Path("reports/tables")
df = cargar_dataset(cfg)
print(f"{len(df)} filas, {df.cod_ies.nunique()} IES")
"""),
    md("""
## 1. Modelos finales e hiperparámetros

- **Tarea 1:** naive (tasa de t-1), C solo historia con ridge (ajustado) y B historia + contexto con xgboost (ajustado).
- **Tarea 2:** mediana por carácter × sector, ridge con información mutua (por defecto y ajustado) y xgboost con clusters (ajustado).

En el ajuste anidado cada pliegue eligió sus hiperparámetros; los **finales son el consenso** de los 25 pliegues (moda; mediana si son continuos).
Esa combinación no se validó como tal y salió de los mismos datos, así que los resultados son algo optimistas (igual en todos los escenarios).
"""),
    code("""
hp = pd.read_csv(R / "hiperparametros_finales.csv")
display(hp)
"""),
    md("""
## 2. Error por subgrupo

GroupKFold por IES 5 × 5 (semillas 301–305). El error de cada fila es el promedio de sus 5 predicciones fuera de muestra. El IC sale de remuestrear IES completas
(1.000 réplicas); la diferencia es **línea base − modelo**: negativa = el modelo es peor.
"""),
    code("""
sub = pd.read_csv(R / "subgrupos.csv")
peores = sub[sub.clasificacion.str.startswith("peor")]
print("Subgrupos donde el modelo es peor que su línea base (con IC que excluye 0 → «peor que la línea base»; si no, «sin firmeza»):")
display(peores[["modelo", "variable", "nivel", "n_filas", "n_ies", "mae_modelo", "mae_base", "dif", "dif_ic_inf", "dif_ic_sup", "clasificacion"]].sort_values("dif").head(15).reset_index(drop=True))
display(Image("reports/figures/error_subgrupos.png"))
"""),
    code("""
# Verificación: el punto estimado de un subgrupo es el promedio, sobre las filas del subgrupo, del error medio de las 5 repeticiones
oof = np.load(R / "_parcial_sesgo" / "oof_base.npz") if (R / "_parcial_sesgo" / "oof_base.npz").exists() else None
if oof is not None:
    y = df.tasa_desercion.to_numpy()
    s_, _ = sg.asignar_subgrupos(df, p["ciudades"])
    m = (s_["region"] == "Bogotá").to_numpy()
    for modelo in ("T2 xgboost con clusters (ajustado)", "T2 base_mediana_grupo"):
        print(f"{modelo:40s} MAE en Bogotá = {np.abs(oof[modelo] - y).mean(axis=0)[m].mean():.4f}")
    fila = sub[(sub.modelo == "T2 xgboost con clusters (ajustado)") & (sub.variable == "region") & (sub.nivel == "Bogotá")].iloc[0]
    print("En subgrupos.csv: modelo", round(fila.mae_modelo, 4), "| línea base", round(fila.mae_base, 4), "| diferencia", round(fila.dif, 4), f"[{fila.dif_ic_inf:.4f}; {fila.dif_ic_sup:.4f}]")
else:
    print("Los puntos de control no están (data/ no se versiona): se omite la verificación.")
"""),
    md("""
## 3. Sensibilidad

Se re-entrena todo con el mismo protocolo en el dataset ya construido (sin re-correr el ETL). Los hiperparámetros finales **no se re-ajustan** en cada escenario.
«Cambia la conclusión» = cambia la clase frente a la línea base (`supera`, `empeora` o `sin diferencia firme`, con significancia en Wilcoxon y t corregida).
"""),
    code("""
sens = pd.read_csv(R / "sensibilidad.csv")
cols = ["escenario", "n_filas", "n_ies", "configuracion", "mae", "mae_ic_inf", "mae_ic_sup", "mejora_base_pct", "p_wilcoxon", "p_t_corregida", "clase", "cambia_conclusion"]
for tarea in ("T1", "T2"):
    print("Tarea", tarea[1])
    display(sens[sens.tarea == tarea][cols].reset_index(drop=True))
"""),
    md("""
## 4. Diagnóstico del colapso del ridge ajustado de la Tarea 2 (prepandemia)

Se entrena con 2015–2017 y se prueba en 2018–2019 con los hiperparámetros que la búsqueda interna eligió en esa partición. Se prueba si el colapso desaparece
(i) quitando `mpio_matriculados_t2`, (ii) con el alfa por defecto, (iii) recortando las variables escaladas a [-3, 3]; y (iv, adicional) entrenando solo con 2017.
Es solo diagnóstico: no cambia los modelos finales.
"""),
    code("""
resumen, z, pe = dc.ejecutar_diagnostico(cfg, df)   # se recalcula aquí (≈2 s)
display(resumen[["variante", "alpha", "k", "n_entrenamiento", "selecciona_variable", "coef_variable", "mae", "mae_2018", "mae_2019", "mae_bogota", "mae_resto", "peores_en_bogota"]])
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.barh(resumen.variante[::-1], resumen.mae[::-1], color=["#9ecae1" if v != "referencia" else "#e6550d" for v in resumen.variante[::-1]])
ax.set_xlabel("MAE en 2018–2019"); ax.set_title("El colapso (naranja) y las variantes del diagnóstico"); plt.show()
"""),
    code("""
# z de las variables respecto del entrenamiento y las filas de mayor error (municipio incluido)
print("Variables con mayor |z| en la prueba, y la variable sospechosa:")
display(pd.concat([z.head(6), z[z.variable == "mpio_matriculados_t2"]]).drop_duplicates("variable")[["variable", "seleccionada", "coef", "z_max_abs", "z_max_abs_bogota", "pct_filas_z_sobre_umbral", "pct_bogota_fuera_del_rango"]])
print("Las 8 filas de mayor error:")
display(pe.head(8)[["rango", "cod_ies", "municipio", "anio", "caracter", "y", "pred", "error", "z_variable_sospechosa", "aporte_variable_sospechosa"]])
print("Municipios de las", len(pe), "filas de mayor error:", pe.municipio.value_counts().to_dict())
"""),
    md("""
## 5. Interpretabilidad

- **SHAP** (`TreeExplainer`) en los dos xgboost finales, ajustados con todo el dataset. xgboost se entrena sobre el logit de la tasa: SHAP positivo = sube la deserción predicha.
- **Importancia por permutación** en validación (GroupKFold por IES): aumento del MAE al permutar cada variable original.
- **Coeficientes del ridge** y su estabilidad en los 25 pliegues de entrenamiento.
"""),
    code("""
top = pd.read_csv(R / "shap_top.csv")
for modelo, g in top.groupby("modelo", sort=False):
    print(modelo)
    display(g[["rank", "variable", "mean_abs_shap", "spearman"]].reset_index(drop=True))
display(Image("reports/figures/shap_t1.png"))
display(Image("reports/figures/shap_t2.png"))
"""),
    code("""
# ¿Coinciden SHAP y permutación? Orden por variable original (los clusters, derivados de todas las variables, quedan aparte)
gr = pd.read_csv(R / "shap_grupos.csv")
from scipy.stats import spearmanr
fig, ejes = plt.subplots(1, 2, figsize=(11, 4))
for ax, (modelo, g) in zip(ejes, gr[gr.variable != "clusters (agregados)"].dropna(subset=["rank_permutacion"]).groupby("modelo", sort=False)):
    rho = spearmanr(g.mean_abs_shap, g.aumento_mae_medio).statistic
    ax.scatter(g.mean_abs_shap, g.aumento_mae_medio, s=18)
    for _, r in g.sort_values("mean_abs_shap", ascending=False).head(4).iterrows():
        ax.annotate(r.variable, (r.mean_abs_shap, r.aumento_mae_medio), fontsize=7)
    ax.set_xlabel("SHAP absoluto medio"); ax.set_ylabel("aumento del MAE al permutar"); ax.set_title(f"{modelo.split(' (')[0]}\\nSpearman = {rho:.2f}", fontsize=9)
plt.tight_layout(); plt.show()
"""),
    code("""
# Coeficientes del ridge de la Tarea 2 (información mutua): estabilidad entre los 25 pliegues
coef = pd.read_csv(R / "coeficientes_ridge.csv")
display(coef[coef.modelo.str.startswith("T2")].head(12))
display(Image("reports/figures/coeficientes_ridge.png"))
"""),
    md("## 6. Conclusiones\n\n(Tomadas de `reports/06_sesgo_modelos.md`; todas las cifras salen de la corrida. La tabla «en lenguaje simple» con la dirección de cada variable y la coincidencia entre métodos está en ese informe.)\n\n__CONCLUSION__"),
]

nb = nbf.v4.new_notebook()
conclusion = seccion("Conclusión") if INFORME.exists() else "Ejecutar antes `desercion evaluar sesgo`."
for c in celdas:
    if c.cell_type == "markdown" and "__CONCLUSION__" in c.source:
        c.source = c.source.replace("__CONCLUSION__", conclusion)
nb.cells = celdas
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
ExecutePreprocessor(timeout=1200, kernel_name="python3", resources={"metadata": {"path": str(AQUI)}}).preprocess(nb)
nbf.write(nb, DESTINO)
print("escrito", DESTINO)
