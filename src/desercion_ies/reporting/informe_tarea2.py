"""Informe de la Tarea 2 (`reports/03_tarea2.md`), descripción de clusters y figuras.

Todo el texto con cifras sale de los resultados (`cribado`, `confirmacion` y sus resúmenes); no se escribe ninguna cifra a mano. Lo único
escrito a mano es el **nombre corto de cada cluster** (`NOMBRES_CLUSTERS`, una interpretación de quien analiza, asignada por los municipios
sede dominantes y no por el número del cluster).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from desercion_ies.reporting.informe_experimentos import _num, _pct  # noqa: E402
from desercion_ies.reporting.informe_v3 import _c, _ic, _p, _veredicto  # noqa: E402
from desercion_ies.evaluation.protocolo import comparar_pareado  # noqa: E402
from desercion_ies.evaluation.tarea2 import (  # noqa: E402
    BASE_GRUPO, BASE_MEDIANA, FACTORES, REDUCCIONES, efecto_contraste, efecto_por_nivel, factor_nb, matriz,
)

# Códigos DANE de municipio → nombre (solo las ciudades que aparecen como sede dominante de algún cluster)
MUNICIPIOS = {"11001": "Bogotá", "05001": "Medellín", "08001": "Barranquilla", "76001": "Cali", "68001": "Bucaramanga", "66001": "Pereira",
              "13001": "Cartagena", "19001": "Popayán", "17001": "Manizales", "23001": "Montería", "27001": "Quibdó", "54001": "Cúcuta",
              "50001": "Villavicencio", "73001": "Ibagué", "52001": "Pasto", "41001": "Neiva", "15001": "Tunja", "20001": "Valledupar",
              "70001": "Sincelejo", "76520": "Palmira", "47001": "Santa Marta", "63001": "Armenia", "44001": "Riohacha", "18001": "Florencia",
              "05088": "Bello", "05266": "Envigado", "05360": "Itagüí", "08758": "Soledad", "25754": "Soacha"}
# Nombre corto de cada cluster según sus municipios sede dominantes (los 2 con más filas). Manual: ver la descripción en el informe.
NOMBRES_CLUSTERS = {
    frozenset({"11001"}): "Bogotá",
    frozenset({"05001", "08001"}): "Grandes ciudades con buen desempeño municipal",
    frozenset({"76001", "17001"}): "Cali y Manizales: baja cobertura",
    frozenset({"68001", "66001"}): "Ciudades intermedias de alta cobertura",
    frozenset({"13001", "19001"}): "Pobreza media-alta, estrato bajo",
    frozenset({"23001", "27001"}): "Periferia con pobreza alta",
}
PROPORCIONES = ("pct_estrato_bajo_t1",)
VARIABLES_CLAVE = [("punt_global_media_t1", "Puntaje Saber 11 medio"), ("pct_estrato_bajo_t1", "% estrato bajo"),
                   ("td_nbi_pct_t1", "NBI (%)"), ("td_ipm_censo_t1", "IPM censal"), ("td_cob_bruta_media_t1", "Cobertura bruta media"),
                   ("td_mdm_t1", "MDM (desempeño municipal)"), ("mpio_matriculados_t2", "Matrícula ES del municipio")]
LEYENDA = {"modelo": "Modelo", "objetivo": "Objetivo", "imputacion": "Imputación", "reduccion": "Reducción", "clusters": "Clusters"}
COLORES = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#b279a2", "#9d755d", "#79706e", "#bab0ac"]


def _nivel(x) -> str:
    return {True: "con", False: "sin"}.get(x, str(x)) if isinstance(x, (bool, np.bool_)) else str(x)


# ---------- clusters ----------
def describir_clusters(df: pd.DataFrame, X: pd.DataFrame, etiquetas: np.ndarray, numericas: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(tabla de descripción, medias de todas las variables por cluster). Los clusters se ajustaron con todas las filas, solo para describir."""
    d = df.assign(cluster=etiquetas)
    filas, medias = [], {}
    nn = [c for c in numericas if not c.startswith("falta_")]
    for cl, g in d.groupby("cluster"):
        top = g["cod_divipola_mpio"].value_counts()
        top2 = frozenset(top.index[:2]) if len(top) > 1 and top.iloc[0] < 0.97 * len(g) else frozenset(top.index[:1])
        nombre = NOMBRES_CLUSTERS.get(top2, f"Cluster {cl}")
        fila = {"cluster": int(cl), "nombre": nombre, "n_filas": len(g), "n_ies": int(g["cod_ies"].nunique()),
                "municipios": ", ".join(f"{MUNICIPIOS.get(m, m)} ({100 * n / len(g):.0f} %)" for m, n in top.head(3).items()),
                "pct_oficial": 100 * float((g["sector"].astype(str) == "1").mean()), "pct_universidad": 100 * float((g["caracter"].astype(str) == "4").mean()),
                "tasa_mediana": float(g["tasa_desercion"].median()), "tasa_media": float(g["tasa_desercion"].mean()),
                "matricula_mediana": float(np.expm1(g["log_matriculados_spadies_t2"]).median()) if "log_matriculados_spadies_t2" in g else np.nan}
        for v, _ in VARIABLES_CLAVE:
            fila[v] = float(g[v].mean()) * (100 if v in PROPORCIONES else 1)  # las proporciones (0-1) se muestran en %
        filas.append(fila)
        medias[int(cl)] = g[nn].mean()
    z = pd.DataFrame(medias).T
    z.index.name = "cluster"
    return pd.DataFrame(filas), z


def _tabla_clusters(desc: pd.DataFrame) -> str:
    cab = ["Cluster", "Filas", "IES", "Municipios sede dominantes", "% oficial", "% universidad", "Matrícula mediana (t-2)", "Tasa mediana"] + [e for _, e in VARIABLES_CLAVE]
    out = ["| " + " | ".join(cab) + " |", "|" + "---|" * len(cab)]
    for _, f in desc.iterrows():
        celdas = [f"**{f['nombre']}** (id {f['cluster']})", f"{int(f['n_filas'])}", f"{int(f['n_ies'])}", f["municipios"], f"{f['pct_oficial']:.0f}", f"{f['pct_universidad']:.0f}",
                  f"{f['matricula_mediana']:,.0f}".replace(",", "."), _pct(100 * f["tasa_mediana"]).replace("+", "")]
        for v, _ in VARIABLES_CLAVE:
            celdas.append(f"{f[v]:,.0f}".replace(",", ".") if abs(f[v]) >= 1000 else f"{f[v]:.1f}".replace(".", ","))
        out.append("| " + " | ".join(celdas) + " |")
    return "\n".join(out)


def figura_clusters(ag, X: pd.DataFrame, etiquetas: np.ndarray, desc: pd.DataFrame, ruta: str | Path) -> None:
    """Clusters en 2D con PCA **solo para visualizar** (el clustering se hizo en el espacio completo)."""
    Z = ag.pre_.transform(X)
    pca = PCA(n_components=2, random_state=0).fit(Z)
    P, C = pca.transform(Z), pca.transform(ag.kmeans_.cluster_centers_)
    nombres = dict(zip(desc["cluster"], desc["nombre"]))
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for cl in range(ag.k_):
        m = etiquetas == cl
        ax.scatter(P[m, 0], P[m, 1], s=14, alpha=0.55, color=COLORES[cl % len(COLORES)], label=f"{nombres.get(cl, cl)} (n = {int(m.sum())})", edgecolor="none")
        ax.scatter(C[cl, 0], C[cl, 1], s=130, marker="X", color=COLORES[cl % len(COLORES)], edgecolor="black", linewidth=1.2, zorder=5)
    ax.set_xlabel(f"Componente 1 de PCA ({100 * pca.explained_variance_ratio_[0]:.0f} % de la varianza)")
    ax.set_ylabel(f"Componente 2 de PCA ({100 * pca.explained_variance_ratio_[1]:.0f} % de la varianza)")
    ax.set_title(f"K-means con k = {ag.k_} (silueta {_num(ag.siluetas_[ag.k_], 3)}); la proyección en 2D solo sirve para visualizar", fontsize=10)
    ax.legend(fontsize=8, loc="best", frameon=True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


# ---------- figura de metodologías ----------
def figura_metodologias(r: pd.DataFrame, ruta: str | Path) -> None:
    """Un panel por decisión: cada punto es una configuración del cribado (MAE medio); el rombo, la media del nivel; y las dos líneas base."""
    m = r[r["tipo"] == "modelo"]
    base = r.set_index("configuracion")
    fig, ejes = plt.subplots(1, len(FACTORES), figsize=(16, 5), sharey=True)
    rng = np.random.default_rng(0)
    for ax, factor in zip(ejes, FACTORES):
        niveles = list(dict.fromkeys(m[factor]))
        for i, nivel in enumerate(niveles):
            v = m.loc[m[factor] == nivel, "mae_media"]
            ax.scatter(i + rng.uniform(-0.2, 0.2, len(v)), v, s=10, alpha=0.45, color=COLORES[i % len(COLORES)], edgecolor="none")
            ax.scatter(i, v.mean(), marker="D", s=70, color=COLORES[i % len(COLORES)], edgecolor="black", zorder=5)
        ax.axhline(base.loc[BASE_GRUPO, "mae_media"], color="black", lw=1.3, label=f"mediana por grupo ({_num(base.loc[BASE_GRUPO, 'mae_media'])})")
        ax.axhline(base.loc[BASE_MEDIANA, "mae_media"], color="#555", ls="--", lw=1.3, label=f"mediana ({_num(base.loc[BASE_MEDIANA, 'mae_media'])})")
        ax.set_xticks(range(len(niveles)), [_nivel(n).replace("_", "\n") for n in niveles], fontsize=8)
        ax.set_title(LEYENDA[factor], fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    ejes[0].set_ylabel("MAE medio del cribado (escala de tasa)")
    ejes[0].legend(fontsize=8, loc="lower left", frameon=True)
    fig.suptitle("Cribado: MAE de las 120 configuraciones según cada decisión (punto = configuración; rombo = media del nivel)", fontsize=10)
    fig.tight_layout()
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


# ---------- tablas ----------
def _tabla_top(r: pd.DataFrame, n: int = 10) -> str:
    base = r.set_index("configuracion")
    ref = base.loc[BASE_GRUPO, "mae_media"]
    out = ["| # | Configuración (modelo \\| objetivo \\| imputación \\| reducción \\| clusters) | MAE medio ± desv. | Mejora vs. mediana por grupo |", "|---|---|---|---|"]
    for _, f in r[r["tipo"] == "linea_base"].iterrows():
        out.append(f"| — | **{f['configuracion']}** | {_num(f['mae_media'])} ± {_num(f['mae_desv'])} | {_pct(100 * (ref - f['mae_media']) / ref)} |")
    for i, (_, f) in enumerate(r[r["tipo"] == "modelo"].head(n).iterrows(), 1):
        out.append(f"| {i} | {_c(f['configuracion'])} | {_num(f['mae_media'])} ± {_num(f['mae_desv'])} | {_pct(100 * (ref - f['mae_media']) / ref)} |")
    return "\n".join(out)


def _tablas_niveles(r: pd.DataFrame) -> str:
    out = ["| Decisión | Nivel | Configuraciones | MAE medio | MAE mediano | Mejor MAE |", "|---|---|---|---|---|---|"]
    for factor in FACTORES:
        for _, f in efecto_por_nivel(r, factor).iterrows():
            out.append(f"| {LEYENDA[factor]} | {_nivel(f['nivel'])} | {int(f['n_configuraciones'])} | {_num(f['mae_medio'])} | {_num(f['mae_mediano'])} | {_num(f['mae_minimo'])} |")
    return "\n".join(out)


REFERENCIAS = {"modelo": "ridge", "objetivo": "ninguna", "imputacion": "mediana", "reduccion": "ninguna", "clusters": False}


def _tabla_contrastes(r: pd.DataFrame) -> str:
    out = ["| Decisión | Nivel | Frente a | Pares | Diferencia media de MAE | Diferencia mediana | % de pares en que reduce el error |", "|---|---|---|---|---|---|---|"]
    for factor in FACTORES:
        for _, f in efecto_contraste(r, factor, REFERENCIAS[factor]).iterrows():
            out.append(f"| {LEYENDA[factor]} | {_nivel(f['nivel'])} | {_nivel(f['referencia'])} | {int(f['n_pares'])} | {_num(0.0 if abs(f['dif_media']) < 5e-5 else f['dif_media'], 4)} | {_num(0.0 if abs(f['dif_mediana']) < 5e-5 else f['dif_mediana'], 4)} | {100 * f['prop_mejora']:.0f} % |")
    return "\n".join(out)


def _tabla_confirmacion(cr: pd.DataFrame) -> str:
    out = ["| Configuración | MAE media [IC 95 %] | R² medio | Diferencia vs. mediana por grupo [IC 95 %] | Mejora | p Wilcoxon | p t corregida | Gana en | Mejora vs. mediana | p Wilcoxon vs. mediana |", "|---|---|---|---|---|---|---|---|---|---|"]
    orden = list(cr[cr["tipo"] == "linea_base"].sort_values("mae_media")["configuracion"]) + list(cr[cr["tipo"] == "modelo"].sort_values("mae_media")["configuracion"])
    por = cr.set_index("configuracion")
    for n in orden:
        f = por.loc[n]
        negrita = "**" if f["tipo"] == "linea_base" else ""
        if n == BASE_GRUPO:
            c = ["—"] * 5
        else:
            c = [f"{_num(f['dif_grupo'], 4)} [{_num(f['dif_grupo_ic_inf'], 4)}; {_num(f['dif_grupo_ic_sup'], 4)}]", _pct(f["mejora_grupo_pct"]), _p(f["p_wilcoxon_grupo"]), _p(f["p_t_grupo"]), f"{100 * f['gana_grupo']:.0f} %"]
        mm = ("—", "—") if n == BASE_MEDIANA else (_pct(f["mejora_mediana_pct"]), _p(f["p_wilcoxon_mediana"]))
        out.append(f"| {negrita}{_c(n)}{negrita} | {_ic(f)} | {_num(f['r2_media'], 3)} | " + " | ".join(c) + f" | {mm[0]} | {mm[1]} |")
    return "\n".join(out)


def _comparar(conf: pd.DataFrame, a: str, b: str, rho: float) -> dict:
    """`a` frente a `b`: diferencia = MAE de b − MAE de a (positiva = a es mejor)."""
    m = matriz(conf)
    return comparar_pareado(m[b] - m[a], rho)


def _conclusion(cri_r: pd.DataFrame, cr: pd.DataFrame, conf: pd.DataFrame, k_info: dict) -> str:
    rho = factor_nb(conf)
    mejor_por = lambda factor, nivel: cr[(cr["tipo"] == "modelo") & (cr[factor] == nivel)]  # noqa: E731, F841
    por = cr.set_index("configuracion")
    modelos = cr[cr["tipo"] == "modelo"]
    lineas = []
    # 1. ¿alguno supera a la mediana por grupo?
    supera = modelos[(modelos["mejora_grupo_pct"] > 0) & (modelos["p_wilcoxon_grupo"] < 0.05) & (modelos["p_t_grupo"] < 0.05)]
    mejor = modelos.sort_values("mae_media").iloc[0]
    mejora_ns = modelos[modelos["mejora_grupo_pct"] > 0].drop(supera.index)
    p_t = mejor["p_t_grupo"]
    lineas.append(
        f"- **¿Alguna configuración supera a la línea base por carácter × sector?** De {len(modelos)} configuraciones confirmadas, {len(supera)} tienen menor error con significancia en las dos pruebas "
        f"(Wilcoxon y t corregida < 0,05), {len(mejora_ns)} tienen menor error medio sin esa significancia y {len(modelos) - len(supera) - len(mejora_ns)} no mejoran a la base. "
        f"La de menor MAE es `{mejor['configuracion']}`: {_ic(mejor)} frente a {_ic(por.loc[BASE_GRUPO])} de la mediana por grupo ({_pct(mejor['mejora_grupo_pct'])}; "
        f"p Wilcoxon {_p(mejor['p_wilcoxon_grupo'])}, p t corregida {_p(p_t)}, gana en {100 * mejor['gana_grupo']:.0f} % de los pliegues); {_veredicto(mejor['dif_grupo'], mejor['p_wilcoxon_grupo'], p_t)}. "
        f"Sin ajustar por comparaciones múltiples; con Bonferroni ({len(modelos)} comparaciones) la p t corregida de esa configuración sería {_p(min(1.0, p_t * len(modelos)))}. "
        f"Frente a la mediana global la mejor mejora {_pct(mejor['mejora_mediana_pct'])} (p Wilcoxon {_p(mejor['p_wilcoxon_mediana'])}). La mediana por grupo mejora a la global en "
        f"{_pct(100 * (por.loc[BASE_MEDIANA, 'mae_media'] - por.loc[BASE_GRUPO, 'mae_media']) / por.loc[BASE_MEDIANA, 'mae_media'])} de MAE ({_num(por.loc[BASE_GRUPO, 'mae_media'])} frente a {_num(por.loc[BASE_MEDIANA, 'mae_media'])}).")
    # 2. ¿qué método de reducción ayuda? (cribado y confirmación)
    lineas.append("- **¿Qué método de reducción ayuda?** En el cribado, efecto de cada reducción frente a no reducir con las demás decisiones iguales (diferencia de MAE; negativa = ayuda):")
    cont = efecto_contraste(cri_r, "reduccion", "ninguna")
    for _, f in cont.iterrows():
        lineas.append(f"  - `{f['nivel']}`: {_num(f['dif_media'], 4)} en promedio ({int(f['n_pares'])} pares; reduce el error en {100 * f['prop_mejora']:.0f} % de ellos).")
    ninguna = modelos[modelos["reduccion"] == "ninguna"].sort_values("mae_media")
    if len(ninguna):
        n0 = ninguna.iloc[0]["configuracion"]
        lineas.append(f"  En la confirmación, la mejor configuración de cada reducción frente a la mejor sin reducción (`{n0}`) (diferencia = MAE sin reducir − MAE con reducción; positiva = la reducción ayuda):")
        for red in [r_ for r_ in REDUCCIONES if r_ != "ninguna"]:
            g = modelos[modelos["reduccion"] == red].sort_values("mae_media")
            if g.empty:
                continue
            c = _comparar(conf, g.iloc[0]["configuracion"], n0, rho)
            lineas.append(f"  - `{g.iloc[0]['configuracion']}`: {_num(c['dif_media'], 4)} [{_num(c['dif_ic_inf'], 4)}; {_num(c['dif_ic_sup'], 4)}] ({_veredicto(c['dif_media'], c['p_wilcoxon'], c['p_t_corregido'])}; p Wilcoxon {_p(c['p_wilcoxon'])}, p t corregida {_p(c['p_t_corregido'])}).")
    # 3. PCA y las alternativas no lineales
    efecto = {f["nivel"]: f for _, f in cont.iterrows()}
    if {"pca", "kernel_pca", "autoencoder"} <= set(efecto):
        d = {k: efecto[k]["dif_media"] for k in ("pca", "kernel_pca", "autoencoder")}
        empeora = d["pca"] > 0
        corrige = [k for k in ("kernel_pca", "autoencoder") if d[k] < d["pca"]]
        sigue = [k for k in ("kernel_pca", "autoencoder") if d[k] > 0]
        frase = (f"{'PCA sigue empeorando' if empeora else 'PCA no empeora'} (diferencia media {_num(d['pca'], 4)} frente a no reducir, en el cribado). "
                 f"KernelPCA: {_num(d['kernel_pca'], 4)}; autoencoder: {_num(d['autoencoder'], 4)}. ")
        if empeora and not corrige:
            frase += "Ninguna alternativa no lineal reduce el daño de PCA."
        elif empeora and sigue:
            frase += f"{' y '.join('`' + k + '`' for k in corrige)} reduce{'n' if len(corrige) > 1 else ''} el daño respecto de PCA, pero {' y '.join('`' + k + '`' for k in sigue)} sigue{'n' if len(sigue) > 1 else ''} empeorando frente a no reducir: no lo corrige{'n' if len(sigue) > 1 else ''} del todo."
        elif empeora:
            frase += "Las alternativas no lineales eliminan el daño (ya no empeoran frente a no reducir)."
        lineas.append("- **¿PCA sigue empeorando y las alternativas no lineales lo corrigen?** " + frase)
    # 4. clusters
    ec = efecto_contraste(cri_r, "clusters", False)
    if len(ec):
        f = ec.iloc[0]
        lineas.append(f"- **¿Ayudan los clusters como variables nuevas?** En el cribado, con clusters frente a sin clusters (demás decisiones iguales): diferencia media de MAE {_num(f['dif_media'], 4)} "
                      f"({int(f['n_pares'])} pares; reduce el error en {100 * f['prop_mejora']:.0f} % de ellos).")
    # 5. modelos
    niv = efecto_por_nivel(cri_r, "modelo")
    lineas.append("- **Modelos** (MAE medio de sus configuraciones en el cribado): " + ", ".join(f"{f['nivel']} ({_num(f['mae_medio'])})" for _, f in niv.iterrows()) + ".")
    return "\n".join(lineas)


def _parrafo_autoencoder(d: pd.DataFrame) -> str:
    paro = int(d["paro_temprano"].sum())
    it = ", ".join(str(int(x)) for x in d["n_iter"])
    return (f"- **Parada temprana del autoencoder:** en los {len(d)} pliegues del cribado el autoencoder paró antes del máximo de {int(d['max_iter'].iloc[0])} iteraciones en {paro} de {len(d)} "
            f"(iteraciones por pliegue: {it}). " + ("Si llega al máximo, el criterio de parada de scikit-learn (15 iteraciones seguidas sin mejora de la validación interna mayor que 1e-4) no se cumplió: la parada temprana está activa pero no limitó el entrenamiento en esos pliegues, y el autoencoder se entrenó hasta el máximo. "
                                                    if paro < len(d) else "") +
            f"Error de reconstrucción medio (unidades estandarizadas): {_num(d['mse_entrenamiento'].mean(), 3)} en entrenamiento y {_num(d['mse_prueba'].mean(), 3)} en prueba.")


def escribir_informe(cri: pd.DataFrame, cri_r: pd.DataFrame, conf: pd.DataFrame, cr: pd.DataFrame, desc: pd.DataFrame, elegidas: list[str], meta: dict, ruta: str | Path,
                     diag_ae: pd.DataFrame | None = None) -> str:
    rho = factor_nb(conf)
    k = meta["k_clusters"]
    texto = f"""# Resultados de la Tarea 2: estimación sin historia («arranque en frío»)

Generado por `desercion evaluar tarea2` (motor en `src/desercion_ies/evaluation/tarea2.py`). Todas las cifras salen de la corrida; no se editan a mano (salvo el nombre corto de cada cluster, una interpretación documentada abajo).

**Tarea.** Estimar la tasa de deserción de una IES solo con su contexto institucional y territorial (formulación A: sin ninguna variable de historia ni de `NUNCA_EXPLICATIVAS`). La validación agrupa por IES, así que cada IES de prueba es nueva para el modelo, como una institución sin historia. El paper tiene dos tareas: la Tarea 1 (pronóstico con historia) está en `reports/02_tarea1.md`.

## Diseño

- **Datos:** `data/processed/dataset_ies.csv` sin cambios, {f"{meta['filas']:,}".replace(',', '.')} filas y {meta['ies']} IES; {meta['n_numericas']} variables numéricas (con los indicadores `falta_*`) y 2 categóricas (sector y carácter).
- **Líneas base:** `base_mediana` (mediana de la tasa del entrenamiento) y `base_mediana_grupo` (mediana por carácter × sector del entrenamiento; si un grupo no existe en el entrenamiento, la mediana global). Las predicciones de los modelos se recortan a [0, 1].
- **Metodologías.** Objetivo: ninguna o logit. Imputación: mediana o KNN (k = {meta['k_vecinos']}). Reducción (metodología 3): ninguna; información mutua (k = {meta['k_info_mutua']}); PCA ({int(round(100 * meta['varianza_pca']))} % de varianza, {meta['n_pca']} componentes en el conjunto completo); **KernelPCA** (RBF, mismo número de componentes que PCA en cada pliegue, gamma = {meta['gamma_texto']}); **autoencoder** (n → {meta['ae_oculta']} → {meta['ae_cuello']} → {meta['ae_oculta']} → n, activación {meta['ae_act']}, parada temprana con validación interna del {int(round(100 * meta['ae_val']))} %, semilla fija, devuelve el cuello de botella; construido con `MLPRegressor` de scikit-learn para no agregar una dependencia como PyTorch o Keras). **Clusters** (opción independiente, sí/no): K-means sobre las variables territoriales e institucionales imputadas y escaladas (sin los indicadores `falta_*`), con k elegido por silueta entre {meta['k_min']} y {meta['k_max']} dentro de cada pliegue; agrega el cluster (one-hot) y las distancias a los centroides como variables **nuevas** (no reemplazan ni se reducen). Modelos: ridge, random forest y xgboost con los hiperparámetros por defecto de la v1.
- **Todos los transformers** (imputación, escalado, selección, PCA, KernelPCA, autoencoder y clusters) se ajustan solo con el entrenamiento de cada pliegue.
- **Protocolo en dos etapas** (para no seleccionar sobre los mismos pliegues). **a) Cribado:** GroupKFold por IES, 5 pliegues, una semilla ({meta['semilla_cribado']}), sobre las {meta['n_configuraciones']} configuraciones de la grilla (2 × 2 × 5 × 2 × 3), en paralelo. **b) Confirmación:** protocolo de la v3 (GroupKFold repetido 5 × 5, {meta['n_ajustes']} ajustes por configuración) con semillas distintas ({meta['semillas_confirmacion']}); incluye las dos líneas base, la mejor configuración de cada modelo y la mejor de cada método de reducción según el cribado ({len(elegidas)} configuraciones sin repetir). El IC 95 % de la media y la «p t corregida» usan la varianza corregida de Nadeau y Bengio con `n_prueba/n_entrenamiento` = {_num(rho, 3)}; el Wilcoxon se aplica a las {meta['n_ajustes']} diferencias pareadas y no corrige la dependencia (más optimista).
- **Tiempo de la corrida:** cribado {meta['t_cribado_s']:.0f} s, confirmación {meta['t_confirmacion_s']:.0f} s, total {meta['t_total_s']:.0f} s ({_num(meta['t_total_s'] / 60, 1)} min), con `n_jobs=-1`. scikit-learn {meta['sklearn']}, xgboost {meta['xgboost']}.

## Conclusión

{_conclusion(cri_r, cr, conf, meta)}

**Sobre LDA.** No se implementa LDA: es un método lineal y supervisado para **clases** (busca las direcciones que mejor separan categorías), y aquí el objetivo es una tasa continua. Solo aplicaría discretizando la tasa (por ejemplo, en terciles), lo que descarta información del objetivo y convierte la regresión en una clasificación, con otro problema y otras métricas.

## Cribado ({meta['n_configuraciones']} configuraciones, una semilla)

### Las diez mejores configuraciones

{_tabla_top(cri_r)}

### Efecto medio de cada metodología y de cada modelo

MAE medio (y mediano) de las configuraciones que usan cada nivel, promediando sobre las demás decisiones, y el mejor MAE de ese nivel.

{_tablas_niveles(cri_r)}

### Contrastes pareados (demás decisiones iguales)

Cada fila compara un nivel con el de referencia en cada par de configuraciones que solo difieren en esa decisión. Diferencia negativa = el nivel reduce el MAE.

{_tabla_contrastes(cri_r)}

![Efecto por metodología](figures/tarea2_metodologias.png)

## Confirmación (GroupKFold repetido 5 × 5, semillas distintas a las del cribado)

Configuraciones elegidas en el cribado: la mejor de cada modelo y la mejor de cada método de reducción. La referencia es la mediana por carácter × sector; «Diferencia» y «Mejora» son positivas cuando la configuración tiene menos error.

{_tabla_confirmacion(cr)}

## Clusters (k = {k})

K-means ajustado con **todas** las filas (solo para describirlos; la evaluación ajusta los clusters dentro de cada pliegue). k por silueta entre {meta['k_min']} y {meta['k_max']}: {', '.join(f'k = {kk}: {_num(v, 3)}' for kk, v in meta['siluetas'].items())}. Los nombres cortos son una interpretación manual a partir de los municipios sede dominantes y de las medias de abajo.

{_tabla_clusters(desc)}

Medias de todas las variables por cluster en `reports/tables/clusters_tarea2.csv`.

![Clusters](figures/clusters.png)

**Lectura.** Los clusters separan sobre todo **ciudades**: cada uno está dominado por uno o dos municipios sede, porque las variables territoriales no cambian entre las IES de un mismo municipio y año. Por eso el cluster funciona casi como un identificador de ciudad y de su perfil socioeconómico. La composición de algunos cambia con los años (por ejemplo, el de ciudades intermedias crece hacia 2021), de modo que también recoge cambios de las fuentes de un año a otro. La silueta es baja ({_num(meta['siluetas'][k], 3)}): los grupos se traslapan y no son una partición nítida.

## Supuestos y limitaciones

- **Contexto compartido entre IES:** en la validación por IES, las demás IES de la misma ciudad están en el entrenamiento y comparten todas las variables territoriales; el modelo puede aprender el nivel de cada ciudad. La mediana por carácter × sector no usa la ciudad.
- **Selección de la confirmación:** el cribado elige con una semilla de pliegues y la confirmación usa otras; aun así ambas usan el mismo dataset y la grilla es grande ({meta['n_configuraciones']} configuraciones), por lo que el cribado sobrestima a los ganadores.
{_parrafo_autoencoder(diag_ae) if diag_ae is not None else ''}
- **Autoencoder:** la validación interna de la parada temprana es una partición aleatoria de las filas de entrenamiento (no agrupa por IES) y solo decide cuándo parar. Los tiempos se midieron con las configuraciones corriendo en paralelo y no son comparables con los de otras corridas.
- **Hiperparámetros por defecto:** no se ajustaron (es una comparación de metodologías). `RidgeCV` elige su alfa con validación interna sin agrupar por IES dentro del entrenamiento.
- **Dependencia entre pliegues:** la corrección de Nadeau y Bengio es una aproximación; con 5 × 5 pliegues, los p de Wilcoxon sobre pares dependientes son optimistas.
- El sesgo de selección, supervivencia y geografía de `reports/05_sesgo_dataset.md` aplica igual a estos resultados.
"""
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    Path(ruta).write_text(texto, encoding="utf-8")
    return texto
