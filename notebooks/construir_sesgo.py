"""Genera notebooks/02_sesgo.ipynb (análisis de sesgo del dataset IES x año) y lo ejecuta de punta a punta.

Uso, desde la raíz del proyecto:  python notebooks/construir_sesgo.py
La subsección final se toma de `reports/analisis_sesgo.md` (sección "Conclusiones"), que se escribe a mano tras leer los resultados.
"""
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

AQUI = Path(__file__).resolve().parent
DESTINO = AQUI / "02_sesgo.ipynb"
INFORME = AQUI.parent / "reports" / "analisis_sesgo.md"


def md(texto: str):
    return nbf.v4.new_markdown_cell(texto.strip("\n"))


def code(texto: str):
    return nbf.v4.new_code_cell(texto.strip("\n"))


celdas = [
    md("""
# Análisis de sesgo — Deserción en educación superior (Colombia), IES × año

Cuantifica los sesgos del dataset `data/processed/dataset_ies.csv` (1.928 IES-año, 295 IES, 2015–2021) y prueba las
técnicas de mitigación que ofrece el repositorio como **opciones** (`src/modelado/`). Nada de lo que sigue cambia el
dataset ni los valores por defecto del pipeline; las cifras salen de los datos del repositorio.

Siete bloques: **a)** selección y cobertura, **b)** supervivencia, **c)** concentración geográfica, **d)** desbalance por
sector y carácter, **e)** faltantes no aleatorios, **f)** medición, **g)** fuga temporal. Al final, la sensibilidad al
umbral de denominador y una comparación de esquemas de validación y de pesos con un modelo lineal de referencia.

**Alcance.** Los contrastes de hipótesis son orientativos: las filas de una misma IES no son independientes.
"""),
    code("""
# Configuración: rutas relativas a la raíz del proyecto y librerías
import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # la configuración usa rutas relativas a la raíz del proyecto

from src.config import load_config
from src.modelado import sesgo as S
from src.modelado.geografia import agregar_geografia, denominador
from src.modelado.mitigacion import ESQUEMAS_PESOS, agregar_panel_incompleto, metricas_por_subgrupo, pesos_muestrales
from src.modelado.particion import (
    ETIQUETAS_CARACTER, ETIQUETAS_SECTOR, cargar_dataset, construir_preprocesador, particion_dejando_fuera,
    particion_group_kfold, particion_group_kfold_estratificado, particion_por_departamento, particion_temporal,
    preparar_x, variables_modelado_desde_config,
)
from src.transform.terridata import transformar_terridata

# Misma paleta que el EDA: azul = oficial, naranja = privada; gris para el resto
AZUL, NARANJA, GRIS, OSCURO = "#2a78d6", "#eb6834", "#8a8984", "#0d366b"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
})
pd.set_option("display.width", 150)
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda v: f"{v:.3f}")
pct = lambda v, _=None: f"{v:.0%}"
"""),
    code("""
cfg = load_config(ROOT / "config" / "config.yaml")
assert cfg["unidad_analisis"] == "ies"
df = cargar_dataset(cfg)  # valores por defecto: ventana 2015-2021, SENA incluido, sin opciones de sesgo
ANIOS = (int(df["anio"].min()), int(df["anio"].max()))
cod = {"cod_divipola_mpio": str, "cod_ies": str, "sector": str, "caracter": str}
interim = ROOT / cfg["rutas"]["interim"]
exclusiones = pd.read_csv(ROOT / "reports" / "exclusiones_calidad.csv", dtype={"cod_ies": str}, encoding="utf-8-sig")
catalogo = pd.read_csv(interim / "snies_catalogo.csv", dtype={**cod, "cod_divipola_mpio_programa": str})

# Universo: toda IES-año con tasa anual en 2015-2021 antes de cualquier filtro, con el filtro que la saca (si alguno)
universo = S.universo_spadies(cfg)
universo = S.clasificar_exclusion(universo, exclusiones, df, cfg.get("min_matriculados_t2", 100))
universo = S.adjuntar_atributos(universo, catalogo)
universo["sector_et"] = universo["sector"].map(ETIQUETAS_SECTOR)
universo["caracter_et"] = universo["caracter"].map(ETIQUETAS_CARACTER)
dfg = agregar_geografia(df)
dfg["sector_et"] = dfg["sector"].map(ETIQUETAS_SECTOR)
dfg["caracter_et"] = dfg["caracter"].map(ETIQUETAS_CARACTER)

print(f"Dataset: {len(df):,} IES-año · {df['cod_ies'].nunique()} IES · {ANIOS[0]}–{ANIOS[1]}")
print(f"Universo antes de filtros: {len(universo):,} IES-año · {universo['cod_ies'].nunique()} IES")
display(universo["estado"].value_counts().rename("IES-año").to_frame())
"""),
    md("""
## a) Sesgo de selección y cobertura

Los filtros de calidad (denominador mínimo, posible cierre, inconsistencia entre fuentes, régimen especial) cambian quién
está en el dataset. Se compara la composición en tres etapas y la tasa de lo que se sacó con la de lo que se quedó.
"""),
    code("""
etapas = ["sin filtro", "tras mínimo y cierre", "dataset final"]
comp = {}
for col, valor, nombre in (("sector", "1", "IES oficiales"), ("caracter", "4", "universidades"), ("ciudad_principal", "Bogotá", "sede en Bogotá"),
                           ("tramo_tamano", "menos de 200", "denominador < 200")):
    t = S.cobertura_por_etapa(universo, col)
    comp[nombre] = t[t[col] == valor].set_index("etapa").reindex(etapas)[["pct_ies", "pct_filas"]]

fig, ejes = plt.subplots(1, 4, figsize=(13, 3.4), sharey=False)
for ax, (nombre, t) in zip(ejes, comp.items()):
    ax.plot(etapas, t["pct_ies"], marker="o", color=AZUL, label="% de IES")
    ax.plot(etapas, t["pct_filas"], marker="s", color=NARANJA, label="% de IES-año")
    for x, (a, b) in enumerate(zip(t["pct_ies"], t["pct_filas"])):
        arriba_ies = a >= b  # la etiqueta del valor más alto va arriba y la del más bajo, abajo, para que no se pisen
        ax.annotate(f"{a:.1%}", (x, a), xytext=(0, 7 if arriba_ies else -13), textcoords="offset points", ha="center", fontsize=8, color=AZUL)
        ax.annotate(f"{b:.1%}", (x, b), xytext=(0, -13 if arriba_ies else 7), textcoords="offset points", ha="center", fontsize=8, color=NARANJA)
    ax.set_title(nombre)
    ax.set_xticks(range(3), ["sin filtro", "mínimo +\\ncierre", "final"], fontsize=8)
    ax.yaxis.set_major_formatter(pct)
    ax.margins(y=0.25)
ejes[0].legend(fontsize=8, loc="best")
fig.suptitle("Composición del dataset antes y después de los filtros", fontweight="bold", y=1.03)
plt.tight_layout()
plt.show()
"""),
    code("""
# Distribución de la tasa: IES-año que salen (por motivo) frente a las que se quedan
resumen = S.resumen_tasa(universo, "estado").set_index("estado").reindex(S.ETAPAS)
cont = {e: S.contraste(universo.loc[universo["estado"] == e, "tasa_desercion"], universo.loc[universo["estado"] == "incluida", "tasa_desercion"])
        for e in S.ETAPAS[1:]}
resumen["P(tasa excluida > tasa incluida)"] = [np.nan] + [cont[e]["prob_superioridad"] for e in S.ETAPAS[1:]]
resumen["p_valor"] = [np.nan] + [cont[e]["p_valor"] for e in S.ETAPAS[1:]]
display(resumen)

universo["sale"] = np.where(universo["estado"] == "incluida", "incluida", "excluida")
display(S.resumen_tasa(universo.assign(grupo=universo["sale"] + " · " + universo["sector_et"].fillna("sin catálogo")), "grupo"))
print(f"Media de la tasa: universo {universo['tasa_desercion'].mean():.3f} · dataset {df['tasa_desercion'].mean():.3f} · "
      f"excluidas {universo.loc[universo['sale'] == 'excluida', 'tasa_desercion'].mean():.3f}")

fig, ax = plt.subplots(figsize=(8.5, 3.6))
orden = S.ETAPAS
datos = [universo.loc[universo["estado"] == e, "tasa_desercion"] for e in orden]
bp = ax.boxplot(datos, orientation="horizontal", patch_artist=True, widths=0.55, showfliers=True, flierprops=dict(markersize=3, markerfacecolor=GRIS, markeredgecolor="none"))
for caja, e in zip(bp["boxes"], orden):
    caja.set(facecolor=AZUL if e == "incluida" else NARANJA, alpha=0.75, edgecolor="#0b0b0b")
for m in bp["medians"]:
    m.set(color="#0b0b0b")
ax.set_yticks(range(1, len(orden) + 1), [f"{e}\\n(n={len(d)})" for e, d in zip(orden, datos)], fontsize=8)
ax.invert_yaxis()
ax.xaxis.set_major_formatter(pct)
ax.set(title="Tasa de deserción por destino de la IES-año", xlabel="tasa anual de deserción")
plt.tight_layout()
plt.show()
"""),
    code("""
# Cobertura por departamento sede: IES antes de los filtros, IES en el dataset final y lo que se pierde
from src.homologation.divipola import cargar_divipola
divipola = cargar_divipola(ROOT / cfg["rutas"]["raw"] / "dane" / "divipola.csv")
nombres_depto = divipola.drop_duplicates("cod_depto").set_index("cod_depto")["depto_norm"].str.title()
ies_u = universo.drop_duplicates("cod_ies").assign(depto=lambda d: d["cod_depto"].map(nombres_depto))
ies_f = universo[universo["estado"] == "incluida"].drop_duplicates("cod_ies")
dep = pd.DataFrame({"ies_sin_filtro": ies_u.groupby("depto").size(), "ies_final": ies_f.assign(depto=ies_f["cod_depto"].map(nombres_depto)).groupby("depto").size()}).fillna(0).astype(int)
dep["pct_ies_sin_filtro"] = dep["ies_sin_filtro"] / dep["ies_sin_filtro"].sum()
dep["pct_ies_final"] = dep["ies_final"] / dep["ies_final"].sum()
dep["cambio_pp"] = (dep["pct_ies_final"] - dep["pct_ies_sin_filtro"]) * 100
dep = dep.sort_values("ies_sin_filtro", ascending=False)
display(dep.head(10))
print(f"{len(dep)} departamentos con IES antes de los filtros; {int((dep['ies_final'] > 0).sum())} después; "
      f"{int((dep['ies_final'] < 3).sum())} tienen menos de 3 IES en el dataset final.")
perd = dep.assign(perdidas=dep["ies_sin_filtro"] - dep["ies_final"]).sort_values("perdidas", ascending=False)
display(perd.head(6)[["ies_sin_filtro", "ies_final", "perdidas", "cambio_pp"]])
"""),
    md("""
## b) Sesgo de supervivencia y panel incompleto

Una IES que deja de reportar (o aparece tarde, o tiene huecos) tiene menos años. Se cuenta cuántas IES no cubren los 7 años
y cómo se compara su tasa con la de las IES de panel completo.
"""),
    code("""
pan = S.patron_panel(df, ANIOS)
panel_universo = S.patron_panel(universo, ANIOS)
print(f"Panel completo ({ANIOS[1] - ANIOS[0] + 1} años): {int(pan['panel_completo'].sum())} de {len(pan)} IES en el dataset "
      f"({pan['panel_completo'].mean():.1%}); antes de los filtros eran {int(panel_universo['panel_completo'].sum())} de {len(panel_universo)} "
      f"({panel_universo['panel_completo'].mean():.1%}).")
display(pan["patron"].value_counts().rename("IES").to_frame())
display(S.comparar_panel(df, ANIOS))
mas = S.comparar_panel(df, ANIOS).iloc[:2].set_index("grupo")
print("Menos de 7 años -> tasa mediana", f"{mas.loc['panel incompleto', 'mediana']:.3f}", "frente a", f"{mas.loc['panel completo', 'mediana']:.3f}", "en panel completo")

fig, ejes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1, 1]})
n = pan["n"].value_counts().reindex(range(1, 8), fill_value=0)
barras = ejes[0].bar(n.index, n.values, color=[AZUL if i == 7 else NARANJA for i in n.index], width=0.7)
ejes[0].bar_label(barras, fontsize=8, padding=2)
ejes[0].set(title="IES según los años con tasa (2015–2021)", xlabel="años con tasa", ylabel="IES")
tmp = df.merge(pan[["cod_ies", "panel_completo"]], on="cod_ies")
cajas = [tmp.loc[tmp["panel_completo"], "tasa_desercion"], tmp.loc[~tmp["panel_completo"], "tasa_desercion"]]
bp = ejes[1].boxplot(cajas, patch_artist=True, widths=0.5, tick_labels=[f"completo\\n(n={len(cajas[0])})", f"incompleto\\n(n={len(cajas[1])})"],
                     flierprops=dict(markersize=3, markerfacecolor=GRIS, markeredgecolor="none"))
for caja, c in zip(bp["boxes"], (AZUL, NARANJA)):
    caja.set(facecolor=c, alpha=0.75, edgecolor="#0b0b0b")
for m in bp["medians"]:
    m.set(color="#0b0b0b")
ejes[1].yaxis.set_major_formatter(pct)
ejes[1].set(title="Tasa según el panel de la IES", ylabel="tasa anual de deserción")
plt.tight_layout()
plt.show()
"""),
    md("""
## c) Concentración geográfica

Cada IES se asigna al municipio donde tiene más programas. Bogotá, Medellín y Cali concentran la oferta y, con ella, las
filas. Además, todas las IES de un municipio comparten sus variables territoriales: tres municipios aportan solo 21 valores
distintos por variable (3 municipios × 7 años).
"""),
    code("""
conc = S.concentracion_ciudades(df)
display(conc)
print(f"Bogotá + Medellín + Cali: {conc.iloc[:3]['pct_ies'].sum():.1%} de las IES y {conc.iloc[:3]['pct_filas'].sum():.1%} de las IES-año.")
reg = S.distribucion(dfg, "region").sort_values("filas", ascending=False)
display(reg)

fig, ax = plt.subplots(figsize=(8, 3.4))
x = np.arange(len(conc))
ax.bar(x - 0.2, conc["pct_ies"], 0.4, color=AZUL, label="% de IES")
ax.bar(x + 0.2, conc["pct_filas"], 0.4, color=NARANJA, label="% de IES-año")
for i, (a, b) in enumerate(zip(conc["pct_ies"], conc["pct_filas"])):
    ax.annotate(f"{a:.1%}", (i - 0.2, a), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    ax.annotate(f"{b:.1%}", (i + 0.2, b), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
ax.set_xticks(x, conc["ciudad_principal"])
ax.yaxis.set_major_formatter(pct)
ax.set(title="Dónde está el dataset (municipio sede de la IES)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
"""),
    code("""
# Variación de las variables territoriales dentro de las tres ciudades frente al resto del país
territoriales = [c for c in df.columns if c.startswith("td_") and c != "td_poblacion_total_t1"] + [
    "punt_global_media_t1", "pct_estrato_bajo_t1", "n_estudiantes_t1", "mpio_matriculados_t2", "mpio_ies_con_oferta_t2"]
disp = S.dispersion_territorial(df, territoriales)
print(f"Municipios distintos: {disp.attrs['municipios_ciudades']} en las tres ciudades y {disp.attrs['municipios_resto']} en el resto del país.")
display(disp.sort_values("razon_desv").reset_index(drop=True))
print("razón < 1: la variable varía menos dentro de las tres ciudades que en el resto; > 1: varía más (variables de tamaño y de escala).")

# Correlación de la tasa con las variables territoriales, con y sin las tres ciudades
ciudad = dfg["ciudad_principal"] != "Resto del país"
filas = []
for c in territoriales:
    filas.append({"variable": c, "rho_todo": dfg[[c, "tasa_desercion"]].corr("spearman").iloc[0, 1],
                  "rho_sin_ciudades": dfg.loc[~ciudad, [c, "tasa_desercion"]].corr("spearman").iloc[0, 1],
                  "rho_solo_ciudades": dfg.loc[ciudad, [c, "tasa_desercion"]].corr("spearman").iloc[0, 1]})
tc = pd.DataFrame(filas).set_index("variable")
tc["cambio"] = tc["rho_sin_ciudades"] - tc["rho_todo"]
display(tc.reindex(tc["cambio"].abs().sort_values(ascending=False).index).head(10))
"""),
    md("""
## d) Desbalance por carácter y sector
"""),
    code("""
cruce = pd.crosstab(dfg["sector_et"], dfg["caracter_et"], margins=True)
ies_cruce = pd.crosstab(dfg.drop_duplicates("cod_ies")["sector_et"], dfg.drop_duplicates("cod_ies")["caracter_et"], margins=True)
print("IES-año por sector y carácter:"); display(cruce)
print("IES por sector y carácter:"); display(ies_cruce)
display(S.resumen_tasa(dfg, ["sector_et", "caracter_et"]) if False else dfg.groupby(["sector_et", "caracter_et"])["tasa_desercion"].agg(
    filas="size", ies=lambda s: dfg.loc[s.index, "cod_ies"].nunique(), mediana="median", media="mean", desv="std").reset_index())

fig, ax = plt.subplots(figsize=(9, 3.8))
orden_car = list(ETIQUETAS_CARACTER.values())
for k, (sec, color) in enumerate((("oficial", AZUL), ("privada", NARANJA))):
    datos = [dfg.loc[(dfg["sector_et"] == sec) & (dfg["caracter_et"] == c), "tasa_desercion"] for c in orden_car]
    pos = np.arange(len(orden_car)) + (k - 0.5) * 0.36
    bp = ax.boxplot(datos, positions=pos, widths=0.3, patch_artist=True, showfliers=False)
    for caja in bp["boxes"]:
        caja.set(facecolor=color, alpha=0.75, edgecolor="#0b0b0b")
    for m in bp["medians"]:
        m.set(color="#0b0b0b")
    for p, d in zip(pos, datos):
        ax.annotate(f"n={len(d)}", (p, 0), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)
    ax.plot([], [], color=color, lw=6, alpha=0.75, label=sec)
ax.set_xticks(range(len(orden_car)), orden_car, fontsize=8)
ax.yaxis.set_major_formatter(pct)
ax.set(title="Tasa de deserción por carácter y sector (sin valores extremos; eje cortado en 45 %)", ylabel="tasa anual", ylim=(0, 0.45))
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
"""),
    md("""
## e) Faltantes no aleatorios

Los nulos de las explicativas vienen por **bloques de años** (MDM y SNIES en 2015–2016, Saber 11 en 2019 y 2021). Se mide
cuánto cambia ese patrón por subgrupo y si, fuera de los bloques, la falta depende del sector, el carácter o el territorio.
"""),
    code("""
cols_nulos = ["punt_global_media_t1", "td_mdm_t1", "matriculados_ies_t2", "td_cob_bruta_media_t1", "td_idf_t1",
              "td_ipm_censo_t1", "td_nbi_pct_t1", "td_repitencia_oficial_t1", "td_transito_superior_t1", "td_desercion_escolar_t1"]
por_anio = dfg.groupby("anio")[cols_nulos].apply(lambda g: g.isna().mean())
fig, ax = plt.subplots(figsize=(9, 3.6))
im = ax.imshow(por_anio.T.values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(por_anio)), por_anio.index)
ax.set_yticks(range(len(cols_nulos)), cols_nulos, fontsize=8)
ax.grid(False)
for i in range(por_anio.shape[1]):
    for j in range(por_anio.shape[0]):
        v = por_anio.iloc[j, i]
        ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7, color="white" if v > 0.5 else "#0b0b0b")
ax.set_title("% de nulos por año (los bloques de años explican casi todo)")
plt.tight_layout()
plt.show()

filas = []
for grupo in ("sector", "caracter", "region", "ciudad_principal"):
    for c in cols_nulos:
        r = S.contraste_nulos(dfg, c, grupo)
        pg = S.nulos_por_grupo(dfg[dfg.groupby("anio")[c].transform(lambda s: s.notna().any())], [c], grupo)[c]
        filas.append({"grupo": grupo, "variable": c, "filas_con_año_observable": r["filas"], "pct_nulos_fuera_de_bloques": r["pct_nulos"],
                      "pct_nulos_max_subgrupo": pg.max(), "subgrupo_max": pg.idxmax(), "p_valor": r["p_valor"]})
nul = pd.DataFrame(filas)
print("Falta fuera de los bloques de años (solo años en que la variable existe):")
display(nul[nul["pct_nulos_fuera_de_bloques"] > 0].sort_values("pct_nulos_max_subgrupo", ascending=False).head(12))
print("Variables sin ningún nulo fuera de los bloques:", sorted(set(nul[nul["pct_nulos_fuera_de_bloques"] == 0]["variable"])))
bloques = dfg.groupby(["sector_et"])[["falta_saber11", "falta_mdm", "falta_snies"]].mean()
display(bloques)
"""),
    md("""
## f) Sesgo de medición

La tasa es `desertores / matriculados de hace dos semestres`: con denominadores chicos es muy inestable, y la matrícula de
SPADIES no siempre coincide con la de SNIES (`ratio_spadies_snies`).
"""),
    code("""
disp_t = S.dispersion_por_tamano(df)
display(disp_t)
rho = df[["tasa_desercion", "log_matriculados_spadies_t2"]].corr("spearman").iloc[0, 1]
print(f"Spearman entre la tasa y el tamaño (denominador): {rho:.3f}")

fig, ejes = plt.subplots(1, 2, figsize=(12, 3.9), gridspec_kw={"width_ratios": [2, 1]})
ejes[0].scatter(denominador(df), df["tasa_desercion"], s=7, alpha=0.35, color=AZUL, edgecolor="none")
ejes[0].set_xscale("log")
for corte in (200, 500, 1000, 5000):
    ejes[0].axvline(corte, color=GRIS, lw=0.8, ls="--")
ejes[0].axhline(0.5, color=NARANJA, lw=0.8, ls=":")
ejes[0].annotate("tasa 50 %", (denominador(df).max(), 0.5), xytext=(-4, 4), textcoords="offset points", ha="right", fontsize=8, color=NARANJA)
ejes[0].yaxis.set_major_formatter(pct)
ejes[0].set(title="Tasa frente al denominador (matrícula SPADIES en t-2)", xlabel="denominador (escala log)", ylabel="tasa anual")
ejes[1].bar(disp_t["tramo_tamano"].astype(str), disp_t["desv"], color=AZUL, label="observada")
ejes[1].bar(disp_t["tramo_tamano"].astype(str), disp_t["desv_binomial_ref"], color=NARANJA, width=0.4, label="referencia binomial")
ejes[1].tick_params(axis="x", labelrotation=30, labelsize=7)
ejes[1].legend(fontsize=8)
ejes[1].set(title="Desviación de la tasa por tramo")
plt.tight_layout()
plt.show()

print("Inconsistencia SPADIES/SNIES (exclusiones_calidad.csv y cola del ratio en el dataset):")
display(S.ratio_fuera_de_rango(exclusiones, df, cfg.get("umbral_ratio_inconsistencia", 3)))
print(df["ratio_spadies_snies"].describe().round(3).to_dict())
"""),
    md("""
## g) Fuga temporal

Para cada explicativa se recalcula su valor desde la fuente con el rezago permitido (Saber 11 y TerriData en t-1, matrícula
SNIES en t-2, denominador de SPADIES de la propia fila) y se cuentan las filas que no coinciden. Una explicativa que usara un
año posterior tendría discrepancias. El test equivalente es `tests/test_sesgo.py`.
"""),
    code("""
td = transformar_terridata(pd.read_csv(interim / "terridata_largo.csv", dtype=cod), cfg["anios"])
fuga = S.verificar_rezagos(
    df, pd.read_csv(interim / "saber11_municipio.csv", dtype=cod), pd.read_csv(interim / "snies_matricula_mpio.csv", dtype=cod),
    pd.read_csv(interim / "snies_matricula_programa.csv", dtype=cod), td, pd.read_csv(interim / "spadies.csv", dtype=cod),
    cfg.get("rezago_matricula", 2))
print(f"{len(fuga)} variables verificadas · {int(fuga['comparadas'].sum()):,} valores comparados · {int(fuga['discrepancias'].sum())} discrepancias")
display(fuga.groupby(["fuente", "rezago"], as_index=False).agg(variables=("columna", "size"), comparadas=("comparadas", "sum"), discrepancias=("discrepancias", "sum")))
from src.modelado.particion import columnas_explicativas
print("Explicativas por defecto sin sufijo de rezago:", S.explicativas_sin_rezago(columnas_explicativas(df)))
# TerriData arrastra la población hacia atrás (bfill): ¿alguna celda que usa el dataset sale de un año posterior?
largo_td = pd.read_csv(interim / "terridata_largo.csv", dtype=cod)
display(S.celdas_rellenadas_hacia_atras(largo_td, cfg["anios"], df))
"""),
    md("""
## Mitigación 1 — umbral de denominador

`min_matriculados_t2` es 100. Se reconstruye el dataset en memoria con 50 y 200 (sin escribir nada) para ver cuánto depende
la muestra del umbral. Con 100 debe reproducirse el dataset publicado.
"""),
    code("""
sens = S.sensibilidad_umbral(cfg, (50, 100, 200))
display(sens)
publicado = sens.loc[sens["umbral"] == cfg.get("min_matriculados_t2", 100)].iloc[0]
assert int(publicado["filas"]) == len(df) and int(publicado["ies"]) == df["cod_ies"].nunique(), "la reconstrucción no reproduce el dataset"
print("Con el umbral publicado la reconstrucción coincide con el dataset.")
"""),
    md("""
## Mitigación 2 — esquemas de validación, pesos y errores por subgrupo

Modelo lineal de referencia (Ridge, `alpha=10`, sin ajuste de hiperparámetros): **no es el modelo final**, solo muestra
qué cambia al usar cada utilidad. Las métricas son el error absoluto medio (MAE) sobre filas de prueba, sin ponderar.
"""),
    code("""
numericas, categoricas = variables_modelado_desde_config(df, cfg)
X = preparar_x(df, numericas, categoricas)
y = df["tasa_desercion"].to_numpy()


def modelo():
    return Pipeline([("pre", construir_preprocesador(numericas, categoricas)), ("ridge", Ridge(alpha=10.0))])


def evaluar(pliegues, pesos=None):
    \"\"\"Predicciones fuera de muestra (índices de prueba) con el modelo y con la mediana del entrenamiento.\"\"\"
    idx, pred, base = [], [], []
    for entrena, prueba in pliegues:
        m = modelo()
        if pesos is None:
            m.fit(X.iloc[entrena], y[entrena])
        else:
            m.fit(X.iloc[entrena], y[entrena], ridge__sample_weight=pesos[entrena])
        idx.append(prueba); pred.append(m.predict(X.iloc[prueba])); base.append(np.full(len(prueba), np.median(y[entrena])))
    idx = np.concatenate(idx)
    return idx, np.concatenate(pred), np.concatenate(base)


esquemas = {
    "GroupKFold por IES": list(particion_group_kfold(df, 5)),
    "GroupKFold estratificado (sector x carácter)": list(particion_group_kfold_estratificado(df, 5)),
    "GroupKFold por departamento": list(particion_por_departamento(df, 5)),
    "Entrena fuera de Bogotá, prueba en Bogotá": [particion_dejando_fuera(df, ("11",))],
    "Temporal 2015–2019 / 2020–2021": [particion_temporal(df, 2019)],
}
filas = []
resultados = {}
for nombre, pliegues in esquemas.items():
    idx, pred, base = evaluar(pliegues)
    resultados[nombre] = (idx, pred)
    filas.append({"esquema": nombre, "filas_prueba": len(idx), "mae_modelo": np.abs(pred - y[idx]).mean(), "mae_mediana": np.abs(base - y[idx]).mean()})
comp_esq = pd.DataFrame(filas)
comp_esq["mejora_vs_mediana"] = 1 - comp_esq["mae_modelo"] / comp_esq["mae_mediana"]
display(comp_esq)

fig, ax = plt.subplots(figsize=(9, 3.4))
y_pos = np.arange(len(comp_esq))
ax.barh(y_pos + 0.2, comp_esq["mae_mediana"], 0.38, color=GRIS, label="mediana del entrenamiento")
ax.barh(y_pos - 0.2, comp_esq["mae_modelo"], 0.38, color=AZUL, label="Ridge")
for i, (a, b) in enumerate(zip(comp_esq["mae_modelo"], comp_esq["mae_mediana"])):
    ax.annotate(f"{a:.3f}", (a, i - 0.2), xytext=(3, 0), textcoords="offset points", va="center", fontsize=8)
    ax.annotate(f"{b:.3f}", (b, i + 0.2), xytext=(3, 0), textcoords="offset points", va="center", fontsize=8)
ax.set_yticks(y_pos, comp_esq["esquema"], fontsize=8)
ax.invert_yaxis()
ax.legend(fontsize=8, loc="upper right")
ax.set(title="Error absoluto medio según el esquema de validación", xlabel="MAE", xlim=(0, 0.085))
plt.tight_layout()
plt.show()
"""),
    code("""
# Efecto de los pesos muestrales (GroupKFold por IES): MAE total y por subgrupo, sin ponderar
base_kf = esquemas["GroupKFold por IES"]
tablas = {}
for esquema in ESQUEMAS_PESOS:
    w = pesos_muestrales(df, esquema)
    idx, pred, _ = evaluar(base_kf, w)
    sub = metricas_por_subgrupo(df.iloc[idx], y[idx], pred, columnas=("sector", "caracter", "region", "ciudad_principal", "tramo_tamano"))
    tablas[esquema] = sub.set_index(["variable", "grupo"])["mae"]
mae_sub = pd.DataFrame(tablas)
display(mae_sub)

# Sesgo medio (predicho - observado) por subgrupo sin pesos
idx, pred, _ = evaluar(base_kf)
sub0 = metricas_por_subgrupo(df.iloc[idx], y[idx], pred, columnas=("sector", "caracter", "region", "ciudad_principal", "tramo_tamano"))
display(sub0)

fig, ax = plt.subplots(figsize=(10, 4.2))
quiere = [("total", "todas"), ("sector", "oficial"), ("sector", "privada"), ("caracter", "técnica profesional"), ("caracter", "universidad"),
          ("ciudad_principal", "Bogotá"), ("ciudad_principal", "Resto del país"), ("tramo_tamano", "menos de 200"), ("tramo_tamano", "5.000 o más")]
quiere = [q for q in quiere if q in mae_sub.index]
ancho = 0.16
colores = [GRIS, AZUL, "#6da7ec", NARANJA, OSCURO]
for k, esquema in enumerate(ESQUEMAS_PESOS):
    ax.bar(np.arange(len(quiere)) + (k - 2) * ancho, [mae_sub.loc[q, esquema] for q in quiere], ancho, color=colores[k], label=esquema)
ax.set_xticks(range(len(quiere)), [f"{g}" for _, g in quiere], rotation=25, ha="right", fontsize=8)
ax.legend(fontsize=7, ncol=5, loc="upper left")
ax.set(title="MAE por subgrupo según el esquema de pesos (GroupKFold por IES)", ylabel="MAE")
plt.tight_layout()
plt.show()
"""),
    code("""
# MAE por subgrupo en la validación que deja fuera a Bogotá y en la temporal
for nombre in ("Entrena fuera de Bogotá, prueba en Bogotá", "Temporal 2015–2019 / 2020–2021"):
    idx, pred = resultados[nombre]
    print(nombre)
    display(metricas_por_subgrupo(df.iloc[idx], y[idx], pred, columnas=("sector", "caracter", "region")))
"""),
    md("## Conclusiones"),
]

if INFORME.exists():
    texto = INFORME.read_text(encoding="utf-8")
    cola = texto.split("## Conclusiones", 1)
    celdas.append(md(cola[1] if len(cola) > 1 else "_Pendiente: escribir las conclusiones en reports/analisis_sesgo.md._"))
else:
    celdas.append(md("_Pendiente: escribir las conclusiones en reports/analisis_sesgo.md._"))

nb = nbf.v4.new_notebook()
nb.cells = celdas
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
ExecutePreprocessor(timeout=900, kernel_name="python3", resources={"metadata": {"path": str(AQUI)}}).preprocess(nb)
nbf.write(nb, DESTINO)
errores = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.output_type == "error"]
print(f"Notebook ejecutado: {sum(c.cell_type == 'code' for c in nb.cells)} celdas de código, {len(errores)} errores")
