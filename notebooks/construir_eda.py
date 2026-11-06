"""Genera notebooks/01_eda.ipynb (EDA del dataset IES x año) y lo ejecuta de punta a punta.

Uso, desde la raíz del proyecto:  python notebooks/construir_eda.py
Las conclusiones del final están en `notebooks/conclusiones_eda.md` y se escriben a mano después de leer los resultados.
"""
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

AQUI = Path(__file__).resolve().parent
DESTINO = AQUI / "01_eda.ipynb"


def md(texto: str):
    return nbf.v4.new_markdown_cell(texto.strip("\n"))


def code(texto: str):
    return nbf.v4.new_code_cell(texto.strip("\n"))


celdas = [
    md("""
# EDA — Deserción en educación superior (Colombia), IES × año

Análisis exploratorio del dataset que produce `pipeline.py` (`data/processed/dataset_ies.csv`).
Cada fila es una IES en un año (2015–2021); la variable objetivo es `tasa_desercion` (SPADIES, MEN) y las
explicativas llegan rezagadas (Saber 11 y TerriData en t-1; matrícula SNIES en t-2). El contexto territorial se
une por el municipio sede de la IES.

**Alcance.** Se describen asociaciones, no causas, y el nivel de análisis es la institución, no el estudiante.
"""),
    code("""
# Configuración: rutas relativas a la raíz del proyecto y librerías
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Polygon
from scipy import stats

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.homologation.divipola import cargar_divipola
from src.validate.dictionary import describir_columnas

# Paleta del proyecto: azul = oficial, naranja = privada; rampa azul para magnitudes
AZUL, NARANJA, GRIS = "#2a78d6", "#eb6834", "#8a8984"
RAMPA = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#1c5cab", "#0d366b"]
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
})
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 40)
"""),
    md("## 1. Carga y descripción general"),
    code("""
cfg = load_config(ROOT / "config" / "config.yaml")
assert cfg["unidad_analisis"] == "ies", "este EDA es para la unidad IES x año"
ruta = ROOT / cfg["rutas"]["processed"] / "dataset_ies.csv"

# Los códigos son texto para conservar los ceros a la izquierda (p. ej. 05001)
df = pd.read_csv(ruta, dtype={"cod_ies": str, "cod_divipola_mpio": str, "sector": str, "caracter": str})
divipola = cargar_divipola(ROOT / cfg["rutas"]["raw"] / "dane" / "divipola.csv")
deptos = divipola.drop_duplicates("cod_depto").set_index("cod_depto")["depto_norm"].str.title()

df["cod_depto"] = df["cod_divipola_mpio"].str[:2]
df["departamento"] = df["cod_depto"].map(deptos)
df["naturaleza"] = df["sector"].map({"1": "Oficial", "2": "Privada"})  # id_sector de SNIES
assert df["tasa_desercion"].notna().all(), "tasa_desercion tiene nulos: falta el export de SPADIES"

print(f"{len(df):,} filas · {df['cod_ies'].nunique()} IES · años {df['anio'].min()}–{df['anio'].max()} · {df.shape[1]} columnas")
print("IES por naturaleza:", df.drop_duplicates("cod_ies")["naturaleza"].value_counts().to_dict())
"""),
    code("""
# Control de la codificación de naturaleza contra el catálogo oficial de IES (sector Oficial/Privado)
catalogo = ROOT / cfg["rutas"]["raw"] / "snies" / "ies_catalogo.csv"
if catalogo.exists():
    cat = pd.read_csv(catalogo, dtype=str).rename(columns={"c_digo_instituci_n": "cod_ies", "sector": "sector_catalogo"})
    cruce = df.drop_duplicates("cod_ies").merge(cat[["cod_ies", "sector_catalogo"]], on="cod_ies", how="left")
    display(pd.crosstab(cruce["naturaleza"], cruce["sector_catalogo"].fillna("sin catálogo")))
"""),
    code("""
# Tipos de dato y resumen de las variables numéricas
display(df.dtypes.to_frame("tipo").T)
display(df.describe().T.round(3))
"""),
    md("## 2. Distribución de la tasa de deserción"),
    code("""
fig, ejes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [2, 1]})
ejes[0].hist(df["tasa_desercion"], bins=40, color=AZUL, edgecolor="#fcfcfb", linewidth=1)
mediana = df["tasa_desercion"].median()
ejes[0].axvline(mediana, color="#0b0b0b", lw=1.2, ls="--")
ejes[0].annotate(f"mediana {mediana:.1%}", (mediana, ejes[0].get_ylim()[1] * 0.92), xytext=(6, 0),
                 textcoords="offset points", fontsize=9)
ejes[0].set(title="Tasa anual de deserción por IES-año", xlabel="tasa de deserción", ylabel="IES-año")
ejes[0].xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")

ejes[1].boxplot(df["tasa_desercion"], orientation="vertical", widths=0.5, patch_artist=True,
                boxprops=dict(facecolor="#cde2fb", edgecolor=AZUL), medianprops=dict(color="#0b0b0b"),
                flierprops=dict(marker="o", markersize=3, markerfacecolor=GRIS, markeredgecolor="none", alpha=0.6))
ejes[1].set(title="Valores atípicos", xticks=[])
ejes[1].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
plt.tight_layout()
plt.show()

q = df["tasa_desercion"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(3)
display(q.to_frame("tasa_desercion").T)
print(f"Asimetría: {stats.skew(df['tasa_desercion']):.2f} · IES-año con tasa > 50 %: {(df['tasa_desercion'] > 0.5).sum()}")
"""),
    code("""
# Las tasas muy altas suelen venir de IES pequeñas o de crecimiento rápido (el denominador es la matrícula de hace un año)
altas = df[df["tasa_desercion"] > 0.5].copy()
display(altas[["cod_ies", "anio", "departamento", "naturaleza", "tasa_desercion", "matriculados_ies_t2"]]
        .sort_values("tasa_desercion", ascending=False).head(10).round(3))
print("Matrícula t-2 mediana de las IES-año con tasa > 50 %:", altas["matriculados_ies_t2"].median(),
      "· del resto:", df.loc[df["tasa_desercion"] <= 0.5, "matriculados_ies_t2"].median())
"""),
    md("""
### Efecto del denominador mínimo (`min_matriculados_t2`)

La tasa es desertores sobre la matrícula de dos semestres atrás; con denominadores chicos es muy inestable. El ETL excluye las
IES-año cuyo denominador (promedio de los dos semestres de t-2) es menor que `min_matriculados_t2` de `config.yaml`. Aquí se
compara el dataset sin ese filtro (mínimo = 0) con el que se usa.
"""),
    code("""
from src.extract.spadies import leer_spadies

minimo = cfg.get("min_matriculados_t2", 100)
# Las rutas de config.yaml son relativas a la raíz del proyecto, no a notebooks/
spadies_abs = {k: (str(ROOT / v) if k in ("archivo", "nacional") else v) for k, v in cfg["spadies"].items()}
base = {**cfg, "spadies": spadies_abs, "excluir_posible_cierre": False}  # el efecto de los posibles cierres se ve aparte
ventana = lambda t: t[t["anio"].between(min(cfg["anios"]), max(cfg["anios"]))]
sin_filtro = ventana(leer_spadies({**base, "min_matriculados_t2": 0}, []))
con_minimo = ventana(leer_spadies({**base, "min_matriculados_t2": minimo}, []))
llave = ["cod_ies", "anio"]
quitadas = sin_filtro.merge(con_minimo[llave], on=llave, how="left", indicator=True).query("_merge == 'left_only'")

print(f"Mínimo de matrícula t-2: {minimo}")
print(f"IES-año sin filtro: {len(sin_filtro):,} · con el mínimo: {len(con_minimo):,} · eliminadas: {len(quitadas):,} "
      f"({len(quitadas) / len(sin_filtro):.1%}) · IES que desaparecen por completo: "
      f"{sin_filtro['cod_ies'].nunique() - con_minimo['cod_ies'].nunique()}")
# Efecto de los posibles cierres: la misma lectura del mínimo, con la opción de la configuración
con_cierre = ventana(leer_spadies({**base, "min_matriculados_t2": minimo,
                                   "excluir_posible_cierre": cfg.get("excluir_posible_cierre", True)}, []))
n_cierre = len(con_minimo) - len(con_cierre)
calidad = pd.read_csv(ROOT / cfg["rutas"]["reports"] / "exclusiones_calidad.csv", dtype={"cod_ies": str})
n_calidad = int(calidad["excluida"].sum())
print(f"Después se excluyen {n_cierre:,} IES-año por posible cierre (excluir_posible_cierre = {cfg.get('excluir_posible_cierre', True)}) "
      f"y {n_calidad:,} por calidad (inconsistencia entre fuentes, régimen especial y, si `excluir_sena`, SENA): "
      f"dataset final {len(df):,} filas, {df['cod_ies'].nunique()} IES")
assert len(con_minimo) - n_cierre - n_calidad == len(df), "las exclusiones no cuadran con el dataset final"

def resumen(serie):
    return pd.Series({"filas": len(serie), "media": serie.mean(), "mediana": serie.median(), "desv. est.": serie.std(),
                      "p95": serie.quantile(0.95), "máximo": serie.max(), "asimetría": stats.skew(serie),
                      "filas con tasa > 50 %": int((serie > 0.5).sum())})

comparacion = pd.DataFrame({"sin filtro": resumen(sin_filtro["tasa_desercion"]),
                            "con el mínimo": resumen(con_minimo["tasa_desercion"]),
                            "eliminadas por el mínimo": resumen(quitadas["tasa_desercion"]),
                            "dataset final": resumen(df["tasa_desercion"])})
display(comparacion.round(3))

fig, ax = plt.subplots(figsize=(9, 3.8))
bins = np.linspace(0, 1, 51)
ax.hist(sin_filtro["tasa_desercion"], bins=bins, color=NARANJA, alpha=0.55, label="sin filtro")
ax.hist(con_minimo["tasa_desercion"], bins=bins, color=AZUL, alpha=0.8, label="con el mínimo")
ax.set(title="Distribución de la tasa con y sin denominador mínimo", xlabel="tasa de deserción", ylabel="IES-año")
ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
ax.legend(frameon=False)
plt.tight_layout()
plt.show()
"""),
    md("""
### IES-año con tasa > 50 % y posible cierre

`reports/tasas_altas.csv` (lo escribe el pipeline) lista las IES-año con tasa mayor que `umbral_tasa_alta` tras el denominador
mínimo, con el denominador de SPADIES, la matrícula t-2 de SNIES y el último periodo que reportó la IES. `posible_cierre` marca
las filas que caen en los 2 semestres previos al último reporte (o en ese reporte) de una IES que dejó de reportar antes del final del archivo.
"""),
    code("""
tasas_altas = pd.read_csv(ROOT / cfg["rutas"]["reports"] / "tasas_altas.csv", dtype={"cod_ies": str})
display(tasas_altas.round({"tasa_desercion": 3, "matriculados_t2_spadies": 0}))
print(f"{len(tasas_altas)} IES-año de {tasas_altas['cod_ies'].nunique()} IES · en ventana de posible cierre: "
      f"{int(tasas_altas['posible_cierre'].sum())} · IES con último reporte 2024-2 (siguen activas): "
      f"{(tasas_altas.drop_duplicates('cod_ies')['ultimo_periodo'] == '2024-2').sum()} de {tasas_altas['cod_ies'].nunique()}")
"""),
    md("""
### Exclusiones por calidad: inconsistencia entre fuentes y régimen especial

- `ratio_spadies_snies` = denominador de SPADIES (matrícula t-2) / matrícula SNIES t-2 de la IES; `inconsistencia_fuentes` se activa
  si el ratio es > 3 o < 1/3, o si SNIES no tiene dato dentro de su cobertura (SNIES por programa empieza en 2015: para los años
  cuyo t-2 es anterior no se marca, porque si no se perderían 2015 y 2016 enteros).
- `regimen_especial`: escuelas militares y policiales, por lista explícita de 16 códigos SNIES (2106, 2901–2906, 3114, 3901, 3902, 9101–9105, 9107); 9108 y 9109 no son militares y quedan fuera.
- `es_sena`: el SENA (9110); se marca pero no se excluye salvo que `excluir_sena` sea true.
- `excluir_inconsistencias` y `excluir_regimen_especial` (por defecto activas) las sacan del dataset, y `excluir_sena` (apagada) también al SENA; el detalle está en `reports/exclusiones_calidad.csv`.
"""),
    code("""
excluye = {"inconsistencia_fuentes": cfg.get("excluir_inconsistencias", True), "regimen_especial": cfg.get("excluir_regimen_especial", True),
           "es_sena": cfg.get("excluir_sena", False)}
resumen_calidad = pd.DataFrame({
    "IES-año marcadas": [int(calidad[c].sum()) for c in excluye] + [len(calidad)],
    "IES": [calidad.loc[calidad[c], "cod_ies"].nunique() for c in excluye] + [calidad["cod_ies"].nunique()],
    "IES-año excluidas": [int((calidad[c] & e).sum()) for c, e in excluye.items()] + [n_calidad],
}, index=list(excluye) + ["unión (filas con alguna bandera)"])
display(resumen_calidad)

# Qué pasa con las tasas > 50 %: cuántas de las 20 filas elimina cada bandera
altas = pd.read_csv(ROOT / cfg["rutas"]["reports"] / "tasas_altas.csv", dtype={"cod_ies": str})
altas[["inconsistencia_fuentes", "regimen_especial"]] = altas[["inconsistencia_fuentes", "regimen_especial"]].fillna(False).astype(bool)
print(f"Tasas > 50 % tras el mínimo: {len(altas)}")
print(f"  eliminadas por inconsistencia_fuentes: {int(altas['inconsistencia_fuentes'].sum())}")
print(f"  eliminadas por regimen_especial: {int(altas['regimen_especial'].sum())}")
print(f"  eliminadas por alguna: {int((altas['inconsistencia_fuentes'] | altas['regimen_especial']).sum())} · quedan {int((~(altas['inconsistencia_fuentes'] | altas['regimen_especial'])).sum())}")

print()
print("Ratio SPADIES / SNIES en el dataset final (IES-año con ambos datos):")
display(df["ratio_spadies_snies"].describe().round(2).to_frame("ratio").T)
"""),
    md("""
### Composición oficial / privada antes y después de las exclusiones

Tres etapas: **sin filtro** (todas las IES con tasa anual en 2015–2021), **tras el denominador mínimo y los posibles cierres** y
**dataset final** (además sin inconsistencias entre fuentes ni régimen especial). La naturaleza sale del catálogo de SNIES.
"""),
    code("""
from src.integrate.build import sede_por_ies

sede, _ = sede_por_ies(pd.read_csv(ROOT / cfg["rutas"]["interim"] / "snies_catalogo.csv", dtype=str))
naturaleza_ies = sede.set_index("cod_ies")["sector"].map({"1": "Oficial", "2": "Privada"})

def composicion(tabla):
    nat = tabla["cod_ies"].map(naturaleza_ies)
    por_ies = nat.groupby(tabla["cod_ies"]).first().value_counts()
    por_fila = nat.value_counts()
    return pd.Series({"IES": tabla["cod_ies"].nunique(), "IES oficiales": por_ies.get("Oficial", 0), "IES privadas": por_ies.get("Privada", 0),
                      "% IES oficiales": 100 * por_ies.get("Oficial", 0) / tabla["cod_ies"].nunique(),
                      "IES-año": len(tabla), "IES-año oficiales": por_fila.get("Oficial", 0), "IES-año privadas": por_fila.get("Privada", 0),
                      "% IES-año oficiales": 100 * por_fila.get("Oficial", 0) / len(tabla)})

etapas = pd.DataFrame({"sin filtro": composicion(sin_filtro), "tras mínimo y cierre": composicion(con_cierre),
                       "dataset final": composicion(df)})
display(etapas.round(1))
perdidas = sin_filtro.drop_duplicates("cod_ies").loc[~sin_filtro.drop_duplicates("cod_ies")["cod_ies"].isin(df["cod_ies"]), "cod_ies"]
print("IES que salen entre la primera y la última etapa, por naturaleza:", perdidas.map(naturaleza_ies).value_counts().to_dict())
"""),
    md("""
### ¿Es el SENA atípico? (por tamaño o por tasa)

El SENA (código 9110) está en el dataset (`es_sena`; `excluir_sena` apagada). Se compara su matrícula de referencia (SNIES, t-2) y su
tasa con la distribución de las demás IES del mismo año.
"""),
    code("""
sena = df[df["es_sena"]].copy()
otras = df[~df["es_sena"]]
filas = []
for anio, g in sena.groupby("anio"):
    resto = otras[otras["anio"] == anio]
    mat = g["matriculados_ies_t2"].iloc[0]
    filas.append({"anio": anio, "tasa SENA": g["tasa_desercion"].iloc[0],
                  "percentil de la tasa": 100 * (resto["tasa_desercion"] < g["tasa_desercion"].iloc[0]).mean(),
                  "matrícula t-2 SENA": mat,
                  "mediana de las demás": resto["matriculados_ies_t2"].median(),
                  "máxima de las demás": resto["matriculados_ies_t2"].max(),
                  "veces la mediana": mat / resto["matriculados_ies_t2"].median() if pd.notna(mat) else np.nan,
                  "veces la mayor otra IES": mat / resto["matriculados_ies_t2"].max() if pd.notna(mat) else np.nan})
display(pd.DataFrame(filas).set_index("anio").round(2))
tasa_sena, tasa_resto = sena["tasa_desercion"], otras["tasa_desercion"]
print(f"Tasa del SENA: mediana {tasa_sena.median():.1%} (rango {tasa_sena.min():.1%}–{tasa_sena.max():.1%}) frente a {tasa_resto.median():.1%} "
      f"de las demás IES (p5–p95: {tasa_resto.quantile(.05):.1%}–{tasa_resto.quantile(.95):.1%}).")
print(f"Matrícula t-2 del SENA: {sena['matriculados_ies_t2'].median():,.0f} frente a una mediana de {otras['matriculados_ies_t2'].median():,.0f} "
      f"y un máximo de {otras['matriculados_ies_t2'].max():,.0f} en las demás.")
"""),
    md("""
**Decisión.** El SENA se mantiene (`excluir_sena: false`) porque su tasa no es atípica; el tamaño se maneja con la transformación
logarítmica `log_matriculados_ies_t2 = log1p(matriculados_ies_t2)`, que es la versión explicativa (la bruta queda en el dataset pero
fuera de las explicativas por defecto). Abajo se ve cuánto comprime el logaritmo la asimetría y el efecto de quitar al SENA
(análisis de sensibilidad: `cargar_dataset(cfg, excluir_sena=True)` o `modelado.excluir_sena: true`).
"""),
    code("""
con_mat = df.dropna(subset=["matriculados_ies_t2"])
sin_sena = con_mat[~con_mat["es_sena"]]
def z(serie, valor):
    return (valor - serie.mean()) / serie.std()

fila = lambda nombre, d: {"escenario": nombre, "filas": len(d), "IES": d["cod_ies"].nunique(),
                          "asimetría matrícula bruta": stats.skew(d["matriculados_ies_t2"]),
                          "asimetría log1p": stats.skew(d["log_matriculados_ies_t2"]),
                          "rho Spearman bruta-tasa": d["matriculados_ies_t2"].corr(d["tasa_desercion"], method="spearman"),
                          "rho Spearman log-tasa": d["log_matriculados_ies_t2"].corr(d["tasa_desercion"], method="spearman"),
                          "tasa mediana": d["tasa_desercion"].median(), "tasa media": d["tasa_desercion"].mean()}
sens = pd.DataFrame([fila("con SENA (por defecto)", con_mat), fila("sin SENA (sensibilidad)", sin_sena)]).set_index("escenario")
display(sens.round(3).T)

fila_sena = con_mat[con_mat["es_sena"]].iloc[0]
print(f"Puntaje z del SENA ({int(fila_sena['anio'])}) frente a las demás IES: matrícula bruta {z(sin_sena['matriculados_ies_t2'], fila_sena['matriculados_ies_t2']):.0f} desviaciones; "
      f"log1p {z(sin_sena['log_matriculados_ies_t2'], fila_sena['log_matriculados_ies_t2']):.1f} desviaciones")

fig, ejes = plt.subplots(1, 2, figsize=(11, 3.6))
ejes[0].hist(sin_sena["matriculados_ies_t2"], bins=40, color=AZUL)
ejes[0].set(title="Matrícula t-2 (bruta), sin el SENA", xlabel="matriculados", ylabel="IES-año")
ejes[1].hist(con_mat["log_matriculados_ies_t2"], bins=40, color=AZUL)
sena_log = con_mat.loc[con_mat["es_sena"], "log_matriculados_ies_t2"]
ejes[1].plot(sena_log, [1] * len(sena_log), "v", color=NARANJA, markersize=9, label="SENA")
ejes[1].set(title="log1p(matrícula t-2), con el SENA", xlabel="log1p(matriculados)")
ejes[1].legend(frameon=False)
plt.tight_layout()
plt.show()
"""),
    md("## 3. Evolución 2015–2021"),
    code("""
por_anio = df.groupby("anio")["tasa_desercion"].agg(
    mediana="median", media="mean", p25=lambda s: s.quantile(0.25), p75=lambda s: s.quantile(0.75), n="size")
display(por_anio.round(3))

fig, ax = plt.subplots(figsize=(8.5, 4))
ax.fill_between(por_anio.index, por_anio["p25"], por_anio["p75"], color="#cde2fb", label="rango intercuartil")
ax.plot(por_anio.index, por_anio["mediana"], color=AZUL, lw=2, marker="o", markersize=6, label="mediana")
ax.plot(por_anio.index, por_anio["media"], color=NARANJA, lw=2, marker="o", markersize=6, label="media")
ax.set(title="Deserción anual: mediana y media entre IES", xlabel="año", ylabel="tasa de deserción")
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
ax.legend(frameon=False, loc="upper right")
plt.tight_layout()
plt.show()
"""),
    md("## 4. Oficial frente a privada"),
    code("""
resumen = df.groupby("naturaleza")["tasa_desercion"].agg(["size", "median", "mean", "std"]).round(3)
display(resumen)
oficial = df.loc[df["naturaleza"] == "Oficial", "tasa_desercion"]
privada = df.loc[df["naturaleza"] == "Privada", "tasa_desercion"]
u, p = stats.mannwhitneyu(oficial, privada)
print(f"Mann-Whitney U (oficial vs. privada): p = {p:.3g}. Las observaciones de una misma IES no son independientes, "
      "así que el valor p es solo orientativo.")

fig, ejes = plt.subplots(1, 2, figsize=(11, 3.8))
ejes[0].boxplot([oficial, privada], tick_labels=["Oficial", "Privada"], widths=0.5, patch_artist=True,
                medianprops=dict(color="#0b0b0b"),
                flierprops=dict(marker="o", markersize=3, markeredgecolor="none", alpha=0.5))
for caja, color in zip(ejes[0].patches, [AZUL, NARANJA]):
    caja.set(facecolor=color, alpha=0.55, edgecolor=color)
ejes[0].set(title="Distribución por naturaleza", ylabel="tasa de deserción")
ejes[0].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")

serie = df.groupby(["anio", "naturaleza"])["tasa_desercion"].median().unstack()
for nombre, color in (("Oficial", AZUL), ("Privada", NARANJA)):
    ejes[1].plot(serie.index, serie[nombre], color=color, lw=2, marker="o", markersize=6, label=nombre)
ejes[1].set(title="Mediana anual por naturaleza", xlabel="año")
ejes[1].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
ejes[1].legend(frameon=False)
plt.tight_layout()
plt.show()
"""),
    md("## 5. Por departamento (sede de la IES)"),
    code("""
dep = df.groupby("departamento").agg(ies=("cod_ies", "nunique"), mediana=("tasa_desercion", "median"),
                                       media=("tasa_desercion", "mean")).sort_values("mediana")
# Con pocas IES la mediana departamental es muy inestable: se grafican solo los departamentos con 3 o más
grande = dep[dep["ies"] >= 3]
fig, ax = plt.subplots(figsize=(8, max(3, 0.32 * len(grande))))
ax.barh(grande.index, grande["mediana"], color=AZUL, height=0.65)
for y, (m, n) in enumerate(zip(grande["mediana"], grande["ies"])):
    ax.text(m + 0.002, y, f"{m:.1%}  (n={n})", va="center", fontsize=8, color="#52514e")
ax.set(title="Mediana de la tasa de deserción por departamento (≥ 3 IES)", xlabel="mediana 2015–2021")
ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
ax.grid(axis="y", visible=False)
plt.tight_layout()
plt.show()
print(f"{len(dep)} departamentos con IES en el dataset; {len(dep) - len(grande)} tienen menos de 3 y no se grafican.")
"""),
    md("## 6. Valores nulos por fuente"),
    code("""
fuentes, _ = describir_columnas(list(df.columns))
explicativas = [c for c in df.columns if c in fuentes and c not in ("tasa_desercion", "cod_ies", "cod_divipola_mpio", "anio")]
nulos = (df[explicativas].isna().mean() * 100).rename("% nulos").to_frame()
nulos["fuente"] = [fuentes[c] for c in nulos.index]

por_fuente = nulos.groupby("fuente")["% nulos"].agg(["mean", "max", "size"]).round(1)
por_fuente.columns = ["% nulos promedio", "% nulos máximo", "variables"]
display(por_fuente.sort_values("% nulos promedio", ascending=False))

fig, ax = plt.subplots(figsize=(8, 0.26 * len(nulos) + 1))
orden = nulos.sort_values("% nulos")
ax.barh(orden.index, orden["% nulos"], color=[NARANJA if v > 25 else AZUL for v in orden["% nulos"]], height=0.65)
ax.set(title="% de nulos por variable explicativa (naranja: más de 25 %)", xlabel="% de filas con valor nulo")
ax.grid(axis="y", visible=False)
plt.tight_layout()
plt.show()
"""),
    code("""
# Variables descartadas por exceso de nulos y otras observaciones de calidad de la última corrida
log = (ROOT / cfg["rutas"]["reports"] / "etl_log.md").read_text(encoding="utf-8")
for linea in log.splitlines():
    if "descartada(s) por superar" in linea or "sin municipio sede" in linea or "cobertura incompleta" in linea:
        print(linea[:400])
"""),
    md("""
**Lectura.** Los nulos de `matriculados_ies_t2` en 2015–2016 son esperados (la matrícula por programa de SNIES empieza en 2015
y el rezago es de dos años). Las de Saber 11 vienen de los años con cobertura incompleta del calendario A, que el ETL excluye
en vez de usar una muestra sesgada.
"""),
    md("""
### Nulos por bloques de años, no al azar

Los nulos de las explicativas no son dispersos: vienen en **bloques de años** por la forma en que se construye cada fuente. El ETL los
marca con tres indicadores 0/1 (`falta_saber11`, `falta_mdm`, `falta_snies`) para que el modelo pueda distinguirlos:

- **MDM** (medición de desempeño municipal) empieza en 2016, así que con el rezago t-1 faltan **2015 y 2016**.
- **SNIES** (matrícula por programa) empieza en 2015, así que con el rezago t-2 faltan **2015 y 2016**.
- **Saber 11** pierde los años cuya cobertura del calendario A es incompleta (2018, 2020 y 2021), así que con el rezago t-1 faltan **2019 y 2021**.
"""),
    code("""
bloques = {"Saber 11": "falta_saber11", "MDM": "falta_mdm", "SNIES (matrícula t-2)": "falta_snies"}
por_anio = pd.DataFrame({nombre: 100 * df.groupby("anio")[col].mean() for nombre, col in bloques.items()}).T.round(0)
por_anio.index.name = "% de IES-año sin dato"
display(por_anio)

fig, ax = plt.subplots(figsize=(7, 2.4))
im = ax.imshow(por_anio.values, cmap=LinearSegmentedColormap.from_list("nulos", ["#fcfcfb", "#eb6834"]), vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(len(por_anio.columns)), por_anio.columns)
ax.set_yticks(range(len(por_anio)), por_anio.index)
for i in range(por_anio.shape[0]):
    for j in range(por_anio.shape[1]):
        ax.text(j, i, f"{por_anio.values[i, j]:.0f}", ha="center", va="center", fontsize=9)
ax.grid(False)
ax.set_title("Porcentaje de IES-año sin dato, por bloque y año")
plt.tight_layout()
plt.show()
"""),
    md("""
**Efecto en la partición temporal.** Con entrenamiento 2015–2019 y prueba 2020–2021: MDM y SNIES faltan solo en el entrenamiento (2015–2016) y la
prueba los tiene completos, de modo que el modelo aprende con una parte de filas imputadas pero se evalúa sobre datos completos; Saber 11 falta en
2019 (entrenamiento) y en **2021 (la mitad de la prueba)**, así que la mitad de la prueba se predice con Saber 11 imputado por la mediana del entrenamiento. La
sensibilidad `ventana_modelado = (2017, 2021)` quita los años donde faltan MDM y SNIES; entrena con 2017–2019.
"""),
    code("""
from src.modelado.particion import particion_temporal

filas = []
for ventana in ((2015, 2021), (2017, 2021)):
    sub = df[df["anio"].between(*ventana)].reset_index(drop=True)
    entrena, prueba = particion_temporal(sub, 2019)
    for etapa, idx in (("entrenamiento", entrena), ("prueba", prueba)):
        parte = sub.iloc[idx]
        filas.append({"ventana": f"{ventana[0]}–{ventana[1]}", "conjunto": f"{etapa} ({int(parte['anio'].min())}–{int(parte['anio'].max())})", "filas": len(parte),
                      "% sin Saber 11": 100 * parte["falta_saber11"].mean(), "% sin MDM": 100 * parte["falta_mdm"].mean(),
                      "% sin SNIES t-2": 100 * parte["falta_snies"].mean()})
display(pd.DataFrame(filas).set_index(["ventana", "conjunto"]).round(1))
"""),
    md("""
### Tamaño de la IES: denominador de SPADIES frente a matrícula de SNIES

`log_matriculados_spadies_t2` (log1p del denominador de la tasa; información de t-2, no fuga) es la variable de tamaño por defecto: viene
**completa en 2015–2021**. `log_matriculados_ies_t2` (SNIES) queda como alternativa: es casi la misma cosa (correlación ~0,99) pero falta en 2015–2016.
Las dos guardan una relación mecánica con la tasa porque el denominador es el de la propia tasa.
"""),
    code("""
tam = df[["log_matriculados_spadies_t2", "log_matriculados_ies_t2", "tasa_desercion"]]
display(pd.DataFrame({"% nulos": 100 * tam.isna().mean(), "rho con la tasa": tam.corr(method="spearman")["tasa_desercion"],
                      "rho entre las dos": [tam.corr(method="spearman").loc["log_matriculados_spadies_t2", "log_matriculados_ies_t2"]] * 3}).iloc[:2].round(3))
"""),
    md("## 7. Correlaciones de las variables explicativas con la tasa"),
    code("""
from src.modelado.particion import EXCLUIDAS_POR_DEFECTO, columnas_explicativas

# Las mismas variables que usará el modelado: sin la población absoluta (cambio de base 2005 -> 2018) ni el ratio de diagnóstico
numericas = columnas_explicativas(df)
print("Fuera de las explicativas por defecto:", EXCLUIDAS_POR_DEFECTO)
X = df[numericas].copy()
X["privada"] = (df["naturaleza"] == "Privada").astype(int)

# Spearman: resistente a los valores atípicos de la tasa y a las relaciones no lineales monótonas
corr = X.corrwith(df["tasa_desercion"], method="spearman").dropna().sort_values()
display(corr.round(3).to_frame("rho de Spearman").T)

fig, ax = plt.subplots(figsize=(8, 0.28 * len(corr) + 1))
ax.barh(corr.index, corr.values, color=[AZUL if v >= 0 else NARANJA for v in corr.values], height=0.65)
ax.axvline(0, color="#0b0b0b", lw=0.8)
ax.set(title="Correlación de Spearman con la tasa de deserción", xlabel="rho")
ax.grid(axis="y", visible=False)
plt.tight_layout()
plt.show()
"""),
    code("""
# Las 12 variables más asociadas con la tasa y su correlación entre sí (para detectar redundancia)
top = corr.abs().sort_values(ascending=False).head(12).index.tolist()
mat = X[top + ["privada"] if "privada" not in top else top].assign(tasa_desercion=df["tasa_desercion"]).corr(method="spearman")
fig, ax = plt.subplots(figsize=(8.5, 7))
mapa = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
im = ax.imshow(mat.values, cmap=mapa, vmin=-1, vmax=1)
ax.set_xticks(range(len(mat)), mat.columns, rotation=90, fontsize=8)
ax.set_yticks(range(len(mat)), mat.index, fontsize=8)
for i in range(len(mat)):
    for j in range(len(mat)):
        ax.text(j, i, f"{mat.values[i, j]:.1f}", ha="center", va="center", fontsize=6.5,
                color="#0b0b0b" if abs(mat.values[i, j]) < 0.6 else "#ffffff")
ax.grid(False)
fig.colorbar(im, ax=ax, shrink=0.7, label="rho de Spearman")
ax.set_title("Correlaciones entre las variables más asociadas con la tasa")
plt.tight_layout()
plt.show()
"""),
    md("""
**Precauciones.** `matriculados_ies_t2` casi coincide con el denominador de la tasa, de modo que su asociación es en parte
mecánica. Varias variables territoriales (TerriData) son muy parecidas entre sí; antes de modelar conviene
reducirlas o regularizarlas. Las IES con sede en una misma ciudad comparten todo su contexto territorial.
"""),
    md("## 8. Mapa por departamento"),
    code("""
# Polígonos departamentales del DANE (MGN 2018), guardados en data/external; se dibujan con matplotlib sin geopandas
geo = json.loads((ROOT / "data" / "external" / "departamentos_colombia.geojson").read_text(encoding="utf-8"))
valor_todos = df.groupby("cod_depto")["tasa_desercion"].median()
n_ies = df.groupby("cod_depto")["cod_ies"].nunique()
# Con 1 o 2 IES la mediana departamental no es representativa: se pintan aparte y no fijan la escala de color
valor = valor_todos[n_ies.reindex(valor_todos.index) >= 3]

norma = Normalize(vmin=valor.min(), vmax=valor.max())
cmap = LinearSegmentedColormap.from_list("rampa", RAMPA)
parches, colores, pocas = [], [], []
for f in geo["features"]:
    cod = f["properties"]["DPTO_CCDGO"]
    anillo = np.array(f["geometry"]["coordinates"][0])
    parches.append(Polygon(anillo, closed=True))
    if cod in valor.index:
        colores.append(cmap(norma(valor[cod])))
    else:
        colores.append("#fcfcfb" if cod in valor_todos.index else "#e6e5e1")
    pocas.append(cod in valor_todos.index and cod not in valor.index)

fig, ax = plt.subplots(figsize=(6.5, 8))
ax.add_collection(PatchCollection(parches, facecolor=colores, edgecolor="#fcfcfb", linewidth=0.8))
# trama diagonal sobre los departamentos con menos de 3 IES
ax.add_collection(PatchCollection([p for p, m in zip(parches, pocas) if m], facecolor="none", edgecolor=GRIS,
                                  hatch="////", linewidth=0.8))
ax.autoscale_view()
ax.set_aspect("equal")
ax.set_axis_off()
ax.set_title("Mediana de la tasa de deserción por departamento sede (2015–2021)\\nrayado: menos de 3 IES · gris: sin IES", fontsize=10)
barra = fig.colorbar(plt.cm.ScalarMappable(norm=norma, cmap=cmap), ax=ax, shrink=0.55, label="tasa de deserción")
barra.ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
plt.tight_layout()
plt.show()

mapa_tabla = pd.DataFrame({"departamento": valor_todos.index.map(deptos), "mediana": valor_todos.values,
                           "IES": n_ies.reindex(valor_todos.index).values})
display(mapa_tabla.sort_values("mediana", ascending=False).round(3).reset_index(drop=True))
print("Rayado: departamentos con 1 o 2 IES (la mediana describe solo a esas instituciones); gris: sin IES en el dataset.")
"""),
    md("## Conclusiones"),
]

conclusiones = AQUI / "conclusiones_eda.md"
celdas.append(md(conclusiones.read_text(encoding="utf-8") if conclusiones.exists()
                 else "_Pendiente: escribir las conclusiones tras revisar los resultados._"))

nb = nbf.v4.new_notebook()
nb.cells = celdas
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
ExecutePreprocessor(timeout=600, kernel_name="python3", resources={"metadata": {"path": str(AQUI)}}).preprocess(nb)
nbf.write(nb, DESTINO)
errores = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.output_type == "error"]
print(f"Notebook ejecutado: {sum(c.cell_type == 'code' for c in nb.cells)} celdas de código, {len(errores)} errores")
