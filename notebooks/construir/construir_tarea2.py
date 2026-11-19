"""Genera notebooks/03_tarea2.ipynb (Tarea 2: estimación sin historia) y lo ejecuta de punta a punta.

Uso, desde la raíz del proyecto:  python notebooks/construir/construir_tarea2.py
Requiere haber corrido antes `desercion evaluar tarea2` (el notebook lee sus CSV y figuras; no repite el cribado). La sección final
de conclusiones se toma de `reports/03_tarea2.md`.
"""
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

AQUI = Path(__file__).resolve().parents[1]  # carpeta notebooks/
DESTINO = AQUI / "03_tarea2.ipynb"
INFORME = AQUI.parent / "reports" / "03_tarea2.md"


def md(texto: str):
    return nbf.v4.new_markdown_cell(texto.strip("\n"))


def code(texto: str):
    return nbf.v4.new_code_cell(texto.strip("\n"))


def seccion(titulo: str) -> str:
    """Texto de la sección `## titulo` del informe (hasta la siguiente `## `)."""
    texto = INFORME.read_text(encoding="utf-8")
    i = texto.index(f"## {titulo}")
    j = texto.find("\n## ", i + 3)
    return texto[i:j if j != -1 else None].replace(f"## {titulo}", "", 1).strip()


celdas = [
    md("""
# Tarea 2 — Estimación sin historia («arranque en frío»)

Predecir la tasa de deserción de una IES **solo con su contexto institucional y territorial** (formulación A: sin ninguna variable
de historia). La validación agrupa por IES, de modo que cada IES de prueba es nueva para el modelo, como una institución sin historia.
Este cuaderno recorre: las líneas base, los métodos que recomendó el profesor (KernelPCA, autoencoder, clusters), el cribado de la
grilla, la confirmación con el protocolo repetido, y la descripción de los clusters.

**Cómo leerlo.** Los resultados pesados (cribado de 120 configuraciones y confirmación 5 × 5) los produce `desercion evaluar tarea2`;
aquí se leen de `reports/tables/` y se muestran. Lo que sí se calcula en el cuaderno es corto: la ilustración de cada método sobre un
pliegue y los clusters.
"""),
    code("""
# Configuración: rutas relativas a la raíz del proyecto, librerías y avisos
import os, sys, warnings
from pathlib import Path

ROOT = Path.cwd()
while not (ROOT / "configs" / "config.yaml").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")  # avisos de KMeans/MKL en Windows y de convergencia: no cambian los resultados

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, Markdown, display
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold

from desercion_ies.config import load_config
from desercion_ies.evaluation import tarea2 as t2
from desercion_ies.evaluation.experimentos import parametros_experimentos
from desercion_ies.models.metodos import AgregarClusters, AutoencoderPequeno, KernelPCAComoPCA
from desercion_ies.features.particion import COLUMNAS_HISTORIA, COLUMNAS_HISTORIA_REZAGADA, NUNCA_EXPLICATIVAS, construir_preprocesador
from desercion_ies.reporting.informe_tarea2 import describir_clusters, REFERENCIAS

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")
cfg = load_config()
p, p_exp = t2.parametros_tarea2(cfg), parametros_experimentos(cfg)
print("Semilla del cribado:", p["semilla_cribado"], "| semillas de la confirmación:", p["semillas_confirmacion"])
"""),
    md("""
## 1. Datos y variables de la tarea

La formulación A usa solo variables de contexto (Saber 11, SNIES, TerriData, tamaño de la IES y los indicadores `falta_*` de los
bloques de años). **Ninguna** variable de historia (`tasa_t1`, `tasa_t2`, medias previas, etc.) ni de `NUNCA_EXPLICATIVAS`. El código
lo comprueba:
"""),
    code("""
# Dataset sin cambios; X = variables explicativas de la tarea 2
df, X, y, numericas, categoricas = t2.preparar_datos(cfg)
prohibidas = set(COLUMNAS_HISTORIA) | set(COLUMNAS_HISTORIA_REZAGADA) | set(NUNCA_EXPLICATIVAS)
assert not prohibidas & set(X.columns), "una columna de historia entró a la tarea 2"
print(f"{len(df)} filas, {df.cod_ies.nunique()} IES; {len(numericas)} numéricas + {len(categoricas)} categóricas {categoricas}")
print("Columnas de historia o NUNCA_EXPLICATIVAS en X:", sorted(prohibidas & set(X.columns)))
print("Tasa de deserción: mediana", round(y.median(), 4), "| media", round(y.mean(), 4))
"""),
    md("""
## 2. Líneas base (sin historia)

- **Mediana** del entrenamiento del pliegue.
- **Mediana por carácter × sector**, calculada en el entrenamiento; si el grupo no existe, la mediana global.

Así se ven las medianas por grupo con todos los datos (en la evaluación se recalculan en cada pliegue con solo su entrenamiento):
"""),
    code("""
# Mediana de la tasa y número de IES-año por carácter × sector (descriptivo; la evaluación las calcula solo con el entrenamiento)
etiquetas = {"caracter": {"1": "técnica profesional", "2": "tecnológica", "3": "inst. universitaria", "4": "universidad"}, "sector": {"1": "oficial", "2": "privada"}}
tabla = (df.assign(caracter=df.caracter.astype(str).map(etiquetas["caracter"]), sector=df.sector.astype(str).map(etiquetas["sector"]))
         .groupby(["caracter", "sector"]).tasa_desercion.agg(filas="size", mediana="median", media="mean"))
display(tabla)
"""),
    md("""
## 3. Los métodos del profesor, ilustrados sobre un pliegue

Todos son *transformers* de scikit-learn que se ajustan **solo con el entrenamiento** del pliegue. Tomamos el primer pliegue del
GroupKFold por IES y mostramos qué hace cada uno (dimensiones, k elegido, parada temprana).
"""),
    code("""
# Primer pliegue del GroupKFold por IES (semilla del cribado); preprocesamos con mediana y escalado SOLO con el entrenamiento
entrena, prueba = next(iter(GroupKFold(n_splits=5, shuffle=True, random_state=p["semilla_cribado"]).split(df, groups=df.cod_ies)))
pre = construir_preprocesador(numericas, categoricas, "mediana", p_exp["k_vecinos"]).fit(X.iloc[entrena])
Ztr, Zte = pre.transform(X.iloc[entrena]), pre.transform(X.iloc[prueba])
print(f"Entrenamiento: {Ztr.shape}  |  prueba: {Zte.shape}")

# PCA al 90 % de la varianza (lineal) y KernelPCA RBF con el MISMO número de componentes (no lineal)
pca = PCA(n_components=p_exp["varianza_pca"], svd_solver="full", random_state=0).fit(Ztr)
kpca = KernelPCAComoPCA(varianza=p_exp["varianza_pca"], gamma=p["kernel_pca"]["gamma"], random_state=p_exp["semilla"]).fit(Ztr)
print(f"PCA: {pca.n_components_} componentes retienen {pca.explained_variance_ratio_.sum():.1%} de la varianza")
print(f"KernelPCA (RBF, gamma = 1/n_variables): {kpca.n_componentes_} componentes; salida de prueba {kpca.transform(Zte).shape}")
"""),
    code("""
# Autoencoder pequeño n -> 16 -> 8 -> 16 -> n con parada temprana (validación interna) y semilla fija; devuelve el cuello de botella (8)
ae = AutoencoderPequeno(**p["autoencoder"], random_state=p_exp["semilla"]).fit(Ztr)
print("Arquitectura (pesos):", [w.shape for w in ae.mlp_.coefs_])
print(f"Iteraciones: {ae.n_iter_} de {ae.max_iter} máximas | parada temprana: {ae.paro_temprano_}")
print("Cuello de botella (entrenamiento, prueba):", ae.transform(Ztr).shape, ae.transform(Zte).shape)
# error de reconstrucción en el entrenamiento y en la prueba (en unidades estandarizadas)
for nombre, Z in (("entrenamiento", Ztr), ("prueba", Zte)):
    print(f"Error cuadrático medio de reconstrucción ({nombre}): {np.mean((ae.mlp_.predict(Z) - Z) ** 2):.3f}")
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(ae.mlp_.loss_curve_, label="pérdida de entrenamiento")
ax.plot(-np.array(ae.mlp_.validation_scores_), label="validación interna (−R²)")
ax.set_xlabel("iteración"); ax.legend(); ax.set_title("Autoencoder: parada temprana"); plt.show()
"""),
    code("""
# Clusters: K-means sobre las variables territoriales e institucionales imputadas y escaladas (sin los indicadores falta_*),
# k por silueta entre 3 y 8. Agrega el cluster (one-hot) y las distancias a los centroides como variables NUEVAS.
ag = AgregarClusters(numericas, categoricas, **p["clusters"], random_state=p_exp["semilla"]).fit(X.iloc[entrena])
print("Silueta por k:", {k: round(v, 3) for k, v in ag.siluetas_.items()})
print(f"k elegido = {ag.k_}; columnas agregadas = {ag.transform(X.iloc[prueba]).shape[1]} (= 2k: one-hot + distancias)")
"""),
    md("""
## 4. Cribado (GroupKFold por IES, 5 pliegues, una semilla)

La grilla completa — objetivo × imputación × reducción × clusters × modelo — se evaluó con `desercion evaluar tarea2`. Las diez mejores:
"""),
    code("""
# Resultados del cribado (una fila por configuración y pliegue) y su resumen por configuración
cribado = pd.read_csv("reports/tables/experimentos_tarea2_cribado.csv")
r = t2.resumen_cribado(cribado)
ref = r.set_index("configuracion").loc[t2.BASE_GRUPO, "mae_media"]
top = r.assign(mejora_vs_mediana_por_grupo_pct=100 * (ref - r.mae_media) / ref)
print(f"{(r.tipo == 'modelo').sum()} configuraciones evaluadas")
display(pd.concat([top[top.tipo == "linea_base"], top[top.tipo == "modelo"].head(10)])[
    ["configuracion", "mae_media", "mae_desv", "rmse_media", "r2_media", "mejora_vs_mediana_por_grupo_pct"]])
"""),
    md("""
### Efecto de cada metodología y de cada modelo

Primero el MAE medio de las configuraciones de cada nivel (promediando las demás decisiones) y después el contraste pareado: la
diferencia de MAE entre un nivel y la referencia en cada par de configuraciones que solo difieren en esa decisión (negativa = reduce el error).
"""),
    code("""
niveles = pd.concat([t2.efecto_por_nivel(r, f) for f in t2.FACTORES]).reset_index(drop=True)
display(niveles[["factor", "nivel", "n_configuraciones", "mae_medio", "mae_mediano", "mae_minimo"]])
contrastes = pd.concat([t2.efecto_contraste(r, f, REFERENCIAS[f]) for f in t2.FACTORES]).reset_index(drop=True)
display(contrastes)
"""),
    code("""
# Figura del cribado (la genera desercion evaluar tarea2)
display(Image("reports/figures/tarea2_metodologias.png"))
"""),
    md("""
## 5. Confirmación (GroupKFold repetido 5 × 5, semillas distintas a las del cribado)

Con el protocolo de la v3: 25 ajustes por configuración, IC 95 % con la corrección de Nadeau y Bengio, y prueba pareada frente a la
mediana por carácter × sector (Wilcoxon y t corregida). Se confirman las dos líneas base, la mejor configuración de cada modelo y la mejor
de cada método de reducción **según el cribado**.
"""),
    code("""
cr = pd.read_csv("reports/tables/experimentos_tarea2_confirmacion_resumen.csv")
cols = ["configuracion", "mae_media", "mae_ic_inf", "mae_ic_sup", "r2_media", "mejora_grupo_pct", "p_wilcoxon_grupo", "p_t_grupo", "gana_grupo"]
display(cr.sort_values("mae_media")[cols])
# IC 95 % de cada configuración (barras) con las dos líneas base como referencia vertical
g = cr[cr.tipo == "modelo"].sort_values("mae_media", ascending=False)
fig, ax = plt.subplots(figsize=(9, 0.5 * len(g) + 1.8))
ax.errorbar(g.mae_media, range(len(g)), xerr=[g.mae_media - g.mae_ic_inf, g.mae_ic_sup - g.mae_media], fmt="o", color="#4c78a8", ecolor="#333", capsize=3)
for nombre, color, estilo in ((t2.BASE_GRUPO, "black", "-"), (t2.BASE_MEDIANA, "#555", "--")):
    ax.axvline(cr.set_index("configuracion").loc[nombre, "mae_media"], color=color, ls=estilo, label=nombre)
ax.set_yticks(range(len(g)), g.configuracion, fontsize=7); ax.set_xlabel("MAE (media e IC 95 %)"); ax.legend(); plt.show()
"""),
    md("""
## 6. Clusters: qué son

K-means ajustado con **todas** las filas (solo para describir; en la evaluación se ajusta dentro de cada pliegue). La proyección
en 2D con PCA es solo para visualizar. Los nombres cortos son una interpretación manual a partir de los municipios sede dominantes.
"""),
    code("""
ag_total, df_cl, etiquetas = t2.ajustar_clusters(cfg, df)
desc, medias = describir_clusters(df_cl, X, etiquetas, numericas)
siluetas = {k: round(v, 3) for k, v in ag_total.siluetas_.items()}
print(f"k = {ag_total.k_} | silueta por k: {siluetas}")
display(desc[["cluster", "nombre", "n_filas", "n_ies", "municipios", "pct_oficial", "pct_universidad", "matricula_mediana", "tasa_mediana",
              "punt_global_media_t1", "pct_estrato_bajo_t1", "td_nbi_pct_t1", "td_ipm_censo_t1", "td_cob_bruta_media_t1", "td_mdm_t1"]])
display(Image("reports/figures/clusters.png"))
"""),
    code("""
# Cómo cambia la composición de cada cluster con los años (si crece o se achica, el cluster recoge también cambios de las fuentes)
display(pd.crosstab(df_cl.assign(cluster=etiquetas).cluster.map(dict(zip(desc.cluster, desc.nombre))), df_cl.anio))
"""),
    md("## 7. Conclusiones\n\n(Tomadas de `reports/03_tarea2.md`; todas las cifras salen de la corrida.)\n\n" + "__CONCLUSION__"),
]

nb = nbf.v4.new_notebook()
conclusion = seccion("Conclusión") if INFORME.exists() else "Ejecutar antes `desercion evaluar tarea2`."
for c in celdas:
    if c.cell_type == "markdown" and "__CONCLUSION__" in c.source:
        c.source = c.source.replace("__CONCLUSION__", conclusion)
nb.cells = celdas
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
ExecutePreprocessor(timeout=900, kernel_name="python3", resources={"metadata": {"path": str(AQUI)}}).preprocess(nb)
nbf.write(nb, DESTINO)
print("escrito", DESTINO)
