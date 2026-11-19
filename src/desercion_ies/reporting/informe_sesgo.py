"""Informe del sesgo medido en los modelos finales, la sensibilidad, el diagnóstico del colapso y la interpretabilidad.

`reports/06_sesgo_modelos.md`, las figuras y la actualización de `reports/05_sesgo_dataset.md` salen de los CSV que escribe
`desercion evaluar sesgo`; ninguna cifra se escribe a mano.
"""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from desercion_ies.analysis.subgrupos import ETIQUETA_VARIABLE, N_PEQUENO, VARIABLES
from desercion_ies.reporting.informe_experimentos import _num, _pct
from desercion_ies.reporting.informe_v3 import _c, _p

XGB_T1, XGB_T2 = (
    "T1 B historia+contexto | xgboost logit (ajustado)",
    "T2 xgboost con clusters (ajustado)",
)
RIDGE_T1, RIDGE_T2 = "T1 C solo historia | ridge (ajustado)", "T2 ridge info_mutua (ajustado)"
CLUSTERS = "clusters (agregados)"
GLOSARIO = {
    "tasa_t1": "tasa de deserción del año anterior",
    "tasa_t2": "tasa de hace dos años",
    "tasa_media_previa": "media de las tasas anteriores",
    "n_anios_previos": "años con tasa anteriores",
    "falta_historia": "no tiene tasa del año anterior",
    "log_matriculados_spadies_t2": "tamaño de la IES (log de la matrícula de hace 2 años)",
    "caracter_universidad": "carácter: universidad",
    "caracter_institución universitaria": "carácter: institución universitaria",
    "caracter_tecnológica": "carácter: tecnológica",
    "caracter_técnica profesional": "carácter: técnica profesional",
    "sector_privada": "sector privado",
    "sector_oficial": "sector oficial",
    "td_miseria_pct_t1": "% de miseria del municipio",
    "td_nbi_pct_t1": "NBI del municipio (%)",
    "td_ipm_censo_t1": "pobreza multidimensional (IPM) del municipio",
    "td_pct_poblacion_rural_t1": "% de población rural del municipio",
    "td_idf_transferencias_t1": "desempeño fiscal: transferencias",
    "pct_estrato_bajo_t1": "% de estudiantes de estrato 1–2",
    "punt_global_media_t1": "puntaje Saber 11 medio",
    "mpio_matriculados_t2": "matrícula de educación superior del municipio",
    "td_cob_neta_media_t1": "cobertura neta del municipio",
    "td_cob_bruta_media_t1": "cobertura bruta del municipio",
    "mpio_ies_con_oferta_t2": "IES con oferta en el municipio",
    "td_pct_pob_20_24_t1": "% de población de 20–24 años del municipio",
    "td_mdm_t1": "desempeño municipal (MDM)",
    "n_estudiantes_t1": "estudiantes que presentaron Saber 11",
}
NOMBRE_CLASE = {
    "supera": "supera",
    "empeora": "empeora",
    "sin diferencia firme": "sin diferencia firme",
}


def _glosa(v: str) -> str:
    if v.startswith("distancia_cluster_"):
        return f"distancia al centroide del cluster {v.rsplit('_', 1)[-1]}"
    if v.startswith("cluster_"):
        return f"pertenece al cluster {v.rsplit('_', 1)[-1]}"
    return GLOSARIO.get(v, v)


def _es_categoria(v: str) -> bool:
    return v.startswith(("caracter_", "sector_", "cluster_")) and not v.startswith(
        "cluster_distancia"
    )


def _fmt_ic(media, lo, hi, d: int = 4) -> str:
    if lo is None or (isinstance(lo, float) and np.isnan(lo)):
        return f"{_num(media, d)} [sin IC]"
    return f"{_num(media, d)} [{_num(lo, d)}; {_num(hi, d)}]"


# ---------- tablas ----------
def _tabla_hp(hp: pd.DataFrame) -> str:
    out = ["| Modelo final | Variante | Hiperparámetros |", "|---|---|---|"]
    for _, f in hp.iterrows():
        d = json.loads(f["hiperparametros"])
        txt = (
            ", ".join(
                f"`{k}` = {re.sub(r'(?<=[0-9])[.](?=[0-9])', ',', f'{v:.4g}' if isinstance(v, float) else str(v))}"
                for k, v in d.items()
            )
            or "por defecto de la v1 (`RidgeCV`, k = 15)"
        )
        out.append(f"| {_c(f['modelo'])} | {f['variante']} | {txt} |")
    return "\n".join(out)


def _tabla_subgrupos(sub: pd.DataFrame, modelo: str) -> str:
    g = sub[sub["modelo"] == modelo]
    out = [
        "| Variable | Subgrupo | Filas (IES) | MAE del modelo [IC 95 %] | MAE de la línea base [IC 95 %] | Diferencia (línea base − modelo) [IC 95 %] | Lectura |",
        "|---|---|---|---|---|---|---|",
    ]
    for variable in VARIABLES:
        for _, f in g[g["variable"] == variable].iterrows():
            marca = "**" if f["clasificacion"].startswith("peor") else ""
            pequeno = " (n pequeño)" if f["n_pequeno"] else ""
            out.append(
                f"| {ETIQUETA_VARIABLE[variable]} | {f['nivel']} | {int(f['n_filas'])} ({int(f['n_ies'])}) | {_fmt_ic(f['mae_modelo'], f['mae_modelo_ic_inf'], f['mae_modelo_ic_sup'])} "
                f"| {_fmt_ic(f['mae_base'], f['mae_base_ic_inf'], f['mae_base_ic_sup'])} | {marca}{_fmt_ic(f['dif'], f['dif_ic_inf'], f['dif_ic_sup'])}{marca} | {marca}{f['clasificacion']}{pequeno}{marca} |"
            )
    return "\n".join(out)


def _tabla_sensibilidad(sens: pd.DataFrame, tarea: str) -> str:
    g = sens[sens["tarea"] == tarea]
    modelos = list(dict.fromkeys(g[g["tipo"] == "modelo"]["configuracion"]))
    base = {"T1": "T1 base_ingenua", "T2": "T2 base_mediana_grupo"}[tarea]
    cab = (
        ["Escenario", "Filas", "IES", f"MAE {base.split(' ', 1)[1]}"]
        + [f"{_c(m)}: MAE; mejora vs. línea base; clase" for m in modelos]
        + ["¿Cambia la conclusión?"]
    )
    out = ["| " + " | ".join(cab) + " |", "|" + "---|" * len(cab)]
    for esc in dict.fromkeys(g["escenario"]):
        e = g[g["escenario"] == esc].set_index("configuracion")
        celdas = [
            esc,
            f"{int(e['n_filas'].iloc[0]):,}".replace(",", "."),
            str(int(e["n_ies"].iloc[0])),
            _fmt_ic(e.loc[base, "mae"], e.loc[base, "mae_ic_inf"], e.loc[base, "mae_ic_sup"]),
        ]
        cambios = []
        for m in modelos:
            f = e.loc[m]
            celdas.append(f"{_num(f['mae'])}; {_pct(f['mejora_base_pct'])}; {f['clase']}")
            if f["cambia_conclusion"]:
                cambios.append(f"{m.split(' | ')[-1]}: «{f['clase_base']}» → «{f['clase']}»")
        celdas.append("sí (" + "; ".join(cambios) + ")" if cambios else "no")
        out.append("| " + " | ".join(celdas) + " |")
    return "\n".join(out)


def _tabla_colapso(col: pd.DataFrame) -> str:
    ref, defecto = (
        col[col["variante"] == "referencia"].iloc[0],
        col[col["variante"] == "ii_defecto_completo"].iloc[0],
    )
    exceso = ref["mae"] - defecto["mae"]
    out = [
        "| Variante | alfa | k | Filas de entrenamiento | ¿Selecciona `mpio_matriculados_t2`? (coeficiente) | MAE | MAE 2018 | MAE 2019 | MAE Bogotá | MAE resto | De las 40 peores filas, en Bogotá | Exceso de error que elimina |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, f in col.iterrows():
        quita = (
            "—"
            if f["variante"] == "referencia"
            else f"{100 * (ref['mae'] - f['mae']) / exceso:.0f} %"
        )
        coef = f"sí ({_num(f['coef_variable'], 3)})" if f["selecciona_variable"] else "no"
        out.append(
            f"| {f['descripcion']} | {_num(f['alpha'], 3)} | {int(f['k'])} | {int(f['n_entrenamiento'])} | {coef} | {_num(f['mae'])} | {_num(f['mae_2018'])} | {_num(f['mae_2019'])} | {_num(f['mae_bogota'])} | {_num(f['mae_resto'])} | {int(f['peores_en_bogota'])} de {int(f['n_peores'])} | {quita} |"
        )
    return "\n".join(out)


def _tabla_z(z: pd.DataFrame, n: int = 10) -> str:
    out = [
        "| Variable | ¿Seleccionada? | Coeficiente | z absoluto máximo | z absoluto máximo en Bogotá | % de filas con z absoluto mayor que el umbral | % de filas de Bogotá fuera del rango de entrenamiento |",
        "|---|---|---|---|---|---|---|",
    ]
    sospechosa = z[z["variable"] == "mpio_matriculados_t2"]
    mostrar = pd.concat(
        [z.head(n), sospechosa[~sospechosa["variable"].isin(z.head(n)["variable"])]]
    )
    for _, f in mostrar.iterrows():
        out.append(
            f"| `{f['variable']}` | {'sí' if f['seleccionada'] else 'no'} | {_num(f['coef'], 3)} | {_num(f['z_max_abs'], 2)} | {_num(f['z_max_abs_bogota'], 2)} | {_num(f['pct_filas_z_sobre_umbral'], 1)} | {_num(f['pct_bogota_fuera_del_rango'], 1)} |"
        )
    return "\n".join(out)


def _tabla_peores(pe: pd.DataFrame, n: int = 10) -> str:
    out = [
        "| # | IES | Municipio | Año | Carácter | Tasa real | Predicha | Error | Aporte de `mpio_matriculados_t2` (z; coeficiente × z) | Mayor aporte (z; coeficiente × z) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, f in pe.head(n).iterrows():
        out.append(
            f"| {int(f['rango'])} | {f['cod_ies']} | {f['municipio']} | {int(f['anio'])} | {f['caracter']} | {_num(f['y'], 3)} | {_num(f['pred'], 3)} | {_num(f['error'], 3)} "
            f"| {_num(f['z_variable_sospechosa'], 2)}; {_num(f['aporte_variable_sospechosa'], 2)} | `{f['aporte_1_variable']}` ({_num(f['aporte_1_z'], 2)}; {_num(f['aporte_1_coef_por_z'], 2)}) |"
        )
    return "\n".join(out)


def _tabla_shap(top: pd.DataFrame, modelo: str) -> str:
    g = top[top["modelo"] == modelo]
    out = [
        "| # | Variable | SHAP absoluto medio | Dirección (Spearman valor–SHAP) |",
        "|---|---|---|---|",
    ]
    for _, f in g.iterrows():
        d = f["spearman"]
        txt = (
            "sin dirección clara"
            if np.isnan(d) or abs(d) < 0.2
            else (
                f"sube la deserción al aumentar ({_num(d, 2)})"
                if d > 0
                else f"baja la deserción al aumentar ({_num(d, 2)})"
            )
        )
        out.append(
            f"| {int(f['rank'])} | `{f['variable']}` ({_glosa(f['variable'])}) | {_num(f['mean_abs_shap'], 3)} | {txt} |"
        )
    return "\n".join(out)


def _tabla_permutacion(perm: pd.DataFrame, modelo: str, n: int = 10) -> str:
    g = perm[perm["modelo"] == modelo].head(n)
    out = [
        "| # | Variable | Aumento del MAE al permutarla (media ± DE entre 5 pliegues) |",
        "|---|---|---|",
    ]
    for _, f in g.iterrows():
        out.append(
            f"| {int(f['rank_permutacion'])} | `{f['variable']}` ({_glosa(f['variable'])}) | {_num(f['aumento_mae_medio'], 4)} ± {_num(f['aumento_mae_sd'], 4)} |"
        )
    return "\n".join(out)


def _tabla_coef(coef: pd.DataFrame, modelo: str, n: int = 12) -> str:
    g = coef[coef["modelo"] == modelo].head(n)
    out = [
        "| Variable | Entra en | Coeficiente (media ± DE) | ¿Signo constante? | Efecto |",
        "|---|---|---|---|---|",
    ]
    for _, f in g.iterrows():
        efecto = (
            "sube la deserción al aumentar" if f["media"] > 0 else "baja la deserción al aumentar"
        )
        out.append(
            f"| `{f['variable']}` ({_glosa(f['variable'])}) | {int(f['n_pliegues_seleccionada'])} de {int(f['de_pliegues'])} pliegues | {_num(f['media'], 3)} ± {_num(f['desviacion'], 3)} | {'sí' if f['signo_constante'] else 'no'} | {efecto} |"
        )
    return "\n".join(out)


# ---------- consistencia entre métodos ----------
def _raiz(variable: str) -> str:
    """Variable original de una columna del modelo (las categorías de sector y carácter vuelven a su variable)."""
    return (
        "sector"
        if variable.startswith("sector_")
        else ("caracter" if variable.startswith("caracter_") else variable)
    )


def tabla_simple(
    top: pd.DataFrame,
    perm: pd.DataFrame,
    coef: pd.DataFrame,
    modelo_xgb: str,
    modelo_ridge: str | None,
    n_perm: int = 10,
) -> pd.DataFrame:
    """Para las variables del top de SHAP del modelo: dirección del efecto y si coinciden los métodos.

    Consistente = «sí» si la variable (o su variable original) está entre las 10 primeras por permutación y su dirección no contradice al ridge (cuando éste la usa de forma estable);
    «no» si la contradice; «parcial» si solo aparece en un método. En la Tarea 1 el ridge predice el *cambio* frente a la naive (otro objetivo), así que no se usa para la dirección.
    """
    g = top[top["modelo"] == modelo_xgb]
    p = perm[perm["modelo"] == modelo_xgb]
    relevante = p[
        (p["aumento_mae_medio"] > p["aumento_mae_sd"]) & (p["aumento_mae_medio"] > 0)
    ]  # el aumento supera su variación entre pliegues
    top_perm = set(relevante.head(n_perm)["variable"])
    cr = (
        coef[coef["modelo"] == modelo_ridge].set_index("variable")
        if modelo_ridge
        else pd.DataFrame()
    )
    filas = []
    for _, f in g.iterrows():
        v, d = f["variable"], f["spearman"]
        signo = 0 if np.isnan(d) or abs(d) < 0.2 else (1 if d > 0 else -1)
        en_perm = _raiz(v) in top_perm
        signo_ridge, estable = 0, False
        if (
            v in cr.index
            and cr.loc[v, "n_pliegues_seleccionada"] >= cr.loc[v, "de_pliegues"] / 2
            and cr.loc[v, "signo_constante"]
        ):
            signo_ridge, estable = (1 if cr.loc[v, "media"] > 0 else -1), True
        if estable and signo and signo_ridge != signo:
            cons = "no"
        elif en_perm and (not estable or not signo or signo_ridge == signo):
            cons = "sí"
        else:
            cons = "parcial"
        if signo == 0:
            direccion = "sin dirección clara (efecto no monótono o débil)"
        elif _es_categoria(v):
            direccion = (
                "pertenecer a esta categoría: más deserción"
                if signo > 0
                else "pertenecer a esta categoría: menos deserción"
            )
        else:
            direccion = "a más, más deserción" if signo > 0 else "a más, menos deserción"
        filas.append(
            {
                "variable": v,
                "significado": _glosa(v),
                "direccion": direccion,
                "mean_abs_shap": f["mean_abs_shap"],
                "en_permutacion_top10": en_perm,
                "ridge": (
                    "sin uso estable" if not estable else ("sube" if signo_ridge > 0 else "baja")
                )
                if modelo_ridge
                else "no aplica",
                "consistente": cons,
            }
        )
    return pd.DataFrame(filas)


def _md_simple(t: pd.DataFrame) -> str:
    out = [
        "| Variable | Qué mide | Efecto sobre la deserción (SHAP) | SHAP absoluto medio | Importancia por permutación | Ridge | ¿Consistente entre métodos? |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, f in t.iterrows():
        out.append(
            f"| `{f['variable']}` | {f['significado']} | {f['direccion']} | {_num(f['mean_abs_shap'], 3)} | {'entre las 10 primeras y por encima de su variación' if f['en_permutacion_top10'] else 'no relevante'} | {f['ridge']} | **{f['consistente']}** |"
        )
    return "\n".join(out)


def coincidencia(grupos: pd.DataFrame, modelo: str, k: int = 5) -> tuple[float, int, int]:
    """(Spearman entre el orden por SHAP y por permutación, variables en común en el top-k, número de variables comparadas), sin los clusters agregados."""
    g = grupos[(grupos["modelo"] == modelo) & (grupos["variable"] != CLUSTERS)].dropna(
        subset=["rank_permutacion"]
    )
    rho = float(spearmanr(g["mean_abs_shap"], g["aumento_mae_medio"]).statistic)
    a = set(g.sort_values("mean_abs_shap", ascending=False).head(k)["variable"])
    b = set(g.sort_values("aumento_mae_medio", ascending=False).head(k)["variable"])
    return rho, len(a & b), len(g)


# ---------- figuras ----------
def figura_subgrupos(sub: pd.DataFrame, ruta: str | Path) -> None:
    fig, ejes = plt.subplots(1, 2, figsize=(15, 7.5), sharey=True)
    colores = ["#08519c", "#e6550d", "#31a354"]
    evaluables = sub[
        ~sub["clasificacion"].str.startswith("no evaluable")
    ]  # los subgrupos de menos de 5 IES (SENA) no tienen IC válido: se omiten
    niveles = [
        (v, n)
        for v in VARIABLES
        for n in dict.fromkeys(evaluables[evaluables["variable"] == v]["nivel"])
    ]
    for ax, tarea in zip(ejes, ("T1", "T2"), strict=False):
        modelos = list(dict.fromkeys(sub[sub["tarea"] == tarea]["modelo"]))
        for j, m in enumerate(modelos):
            g = sub[sub["modelo"] == m].set_index(["variable", "nivel"])
            y = np.arange(len(niveles))[::-1] + (j - (len(modelos) - 1) / 2) * 0.22
            d = np.array([g.loc[k, "dif"] for k in niveles])
            lo = np.array([g.loc[k, "dif_ic_inf"] for k in niveles])
            hi = np.array([g.loc[k, "dif_ic_sup"] for k in niveles])
            peor = np.array([g.loc[k, "clasificacion"].startswith("peor que") for k in niveles])
            ax.errorbar(
                d,
                y,
                xerr=[d - lo, hi - d],
                fmt="none",
                ecolor=colores[j],
                elinewidth=1.3,
                capsize=2,
                alpha=0.8,
            )
            ax.scatter(
                d,
                y,
                color=colores[j],
                s=34,
                zorder=3,
                label=m.split(" | ")[-1] if tarea == "T1" else m.replace("T2 ", ""),
                edgecolor=np.where(peor, "red", "none"),
                linewidth=1.5,
            )
        ax.axvline(0, color="black", lw=1)
        ax.set_yticks(
            np.arange(len(niveles))[::-1],
            [f"{ETIQUETA_VARIABLE[v].split(' (')[0]}: {n}" for v, n in niveles],
            fontsize=8,
        )
        ax.set_xlabel("diferencia de MAE (línea base − modelo); < 0 = el modelo es peor")
        ax.set_title(
            "Tarea 1 (línea base: ingenua)"
            if tarea == "T1"
            else "Tarea 2 (línea base: mediana por carácter × sector)",
            fontsize=10,
        )
        ax.legend(fontsize=8, loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Error por subgrupo frente a la línea base (IC 95 % por bootstrap sobre IES; borde rojo = peor con IC que excluye 0; SENA omitido: una sola IES)",
        fontsize=10,
    )
    fig.tight_layout()
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def figura_shap(cfg: dict, df: pd.DataFrame, nombre: str, ruta: str | Path, titulo: str) -> None:
    """Beeswarm de SHAP del xgboost final, ajustado con todo el dataset."""
    import shap

    from desercion_ies.analysis import interpretabilidad as it
    from desercion_ies.evaluation import ajuste as aj
    from desercion_ies.evaluation import evaluacion_final as ef
    from desercion_ies.evaluation import tarea2 as t2
    from desercion_ies.evaluation.experimentos import parametros_experimentos

    m = next(x for x in ef.modelos_finales(cfg) if x["nombre"] == nombre)
    X, num, cat = aj.preparar(m["spec"], cfg, df)
    est = ef.construir_final(
        m, num, cat, parametros_experimentos(cfg), t2.parametros_tarea2(cfg)
    ).fit(X, df["tasa_desercion"].astype(float).to_numpy())
    sv, M, nombres = it.calcular_shap(est, X)
    expl = shap.Explanation(values=sv, data=M, feature_names=[it._limpio(n) for n in nombres])
    plt.figure(figsize=(9, 7))
    shap.plots.beeswarm(
        expl,
        max_display=cfg["modelado"]["sesgo_sensibilidad"]["shap"]["max_display"],
        show=False,
        plot_size=None,
    )
    plt.title(titulo, fontsize=10)
    plt.tight_layout()
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close("all")


def figura_coeficientes(coef: pd.DataFrame, ruta: str | Path) -> None:
    fig, ejes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2, 1]})
    for ax, modelo, titulo, n in (
        (ejes[0], RIDGE_T2, "Tarea 2: ridge con información mutua", 15),
        (ejes[1], RIDGE_T1, "Tarea 1: C solo historia (predice el cambio frente a la naive)", 5),
    ):
        g = coef[coef["modelo"] == modelo].head(n).iloc[::-1]
        colores = np.where(g["signo_constante"], "#2b8cbe", "#bdbdbd")
        ax.barh(
            g["variable"], g["media"], xerr=g["desviacion"], color=colores, ecolor="#333", capsize=2
        )
        for i, (_, f) in enumerate(g.iterrows()):
            ax.text(
                0,
                i,
                f"  {int(f['n_pliegues_seleccionada'])}/{int(f['de_pliegues'])}",
                va="center",
                fontsize=7,
                color="#444",
            )
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("coeficiente estandarizado (media ± DE en 25 pliegues)")
        ax.set_title(titulo, fontsize=10)
        ax.tick_params(axis="y", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Coeficientes del ridge y su estabilidad entre pliegues (azul = signo constante; gris = el signo cambia; n/25 = pliegues en que entra)",
        fontsize=9,
    )
    fig.tight_layout()
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


# ---------- conclusión ----------
def _peor_servidos(sub: pd.DataFrame) -> list[str]:
    lineas = []
    for modelo in dict.fromkeys(sub["modelo"]):
        g = sub[sub["modelo"] == modelo]
        firmes = g[g["clasificacion"].str.startswith("peor que")]
        sinf = g[(g["clasificacion"] == "peor (sin firmeza)") & (~g["n_pequeno"])]
        f_txt = ", ".join(
            f"{r['variable']}: {r['nivel']} ({_num(r['dif'], 4)} [{_num(r['dif_ic_inf'], 4)}; {_num(r['dif_ic_sup'], 4)}], {int(r['n_filas'])} filas)"
            for _, r in firmes.iterrows()
        )
        s_txt = ", ".join(
            f"{r['variable']}: {r['nivel']} ({_num(r['dif'], 4)})" for _, r in sinf.iterrows()
        )
        lineas.append(
            f"  - `{modelo}`: peor que su línea base con IC que excluye 0 en "
            + (f_txt if f_txt else "ningún subgrupo")
            + "; peor sin firmeza (n ≥ 30) en "
            + (s_txt if s_txt else "ningún subgrupo")
            + "."
        )
    return lineas


def _absolutos(sub: pd.DataFrame, modelo: str, k: int = 3) -> str:
    g = (
        sub[(sub["modelo"] == modelo) & (~sub["n_pequeno"])]
        .sort_values("mae_modelo", ascending=False)
        .head(k)
    )
    return ", ".join(
        f"{r['variable']}: {r['nivel']} (MAE {_num(r['mae_modelo'])}; línea base {_num(r['mae_base'])}; {int(r['n_filas'])} filas)"
        for _, r in g.iterrows()
    )


def _conclusion(sub, sens, col, top, perm, grupos, coef) -> str:
    L = []
    L.append(
        "**1. ¿Qué subgrupos quedan peor servidos?** Con IC 95 % por bootstrap sobre IES; «peor» = el modelo tiene más error que su línea base."
    )
    L += _peor_servidos(sub)
    L.append(
        f"- Con más error absoluto (modelo con menos error de cada tarea): Tarea 1 `{RIDGE_T1}`: {_absolutos(sub, RIDGE_T1)}. Tarea 2 `{RIDGE_T2}`: {_absolutos(sub, RIDGE_T2)}."
    )
    # 2. sensibilidad
    L.append(
        "\n**2. ¿Alguna sensibilidad cambia la conclusión?** («Cambia» = cambia la clase frente a la línea base —`supera`, `empeora` o `sin diferencia firme`, con significancia en las dos pruebas— respecto del escenario base.)"
    )
    camb = sens[sens["cambia_conclusion"]]
    if camb.empty:
        L.append(
            "- Ninguna: en los cuatro escenarios cada modelo conserva la clase que tiene en el dataset completo."
        )
    else:
        for _, f in camb.iterrows():
            mb = sens[
                (sens["escenario"] == "base") & (sens["configuracion"] == f["configuracion"])
            ]["mejora_base_pct"].iloc[0]
            L.append(
                f"- Escenario `{f['escenario']}`, `{f['configuracion']}`: de «{f['clase_base']}» a «{f['clase']}» (mejora frente a la línea base {_pct(mb)} en el dataset completo y {_pct(f['mejora_base_pct'])} en el escenario; p Wilcoxon {_p(f['p_wilcoxon'])}, p t corregida {_p(f['p_t_corregida'])})."
            )
    b = sens[(sens["escenario"] == "base") & (sens["tipo"] == "modelo")].set_index("configuracion")
    d = sens[(sens["escenario"] == "desde_2017") & (sens["tipo"] == "modelo")].set_index(
        "configuracion"
    )
    t2m = [m for m in b.index if m.startswith("T2")]
    L.append(
        "- **Hipótesis del SNIES imputado en 2015–2016** (escenario 2017–2021, solo con matrícula SNIES observada): mejora de los modelos de la Tarea 2 frente a la mediana por carácter × sector, dataset completo → 2017–2021: "
        + "; ".join(
            f"`{m}`: {_pct(b.loc[m, 'mejora_base_pct'])} → {_pct(d.loc[m, 'mejora_base_pct'])}"
            for m in t2m
        )
        + "."
    )
    # 3. colapso
    ref, dflt = (
        col[col["variante"] == "referencia"].iloc[0],
        col[col["variante"] == "ii_defecto_completo"].iloc[0],
    )
    exceso = ref["mae"] - dflt["mae"]
    v = {r["variante"]: r for _, r in col.iterrows()}
    quita = {k: 100 * (ref["mae"] - r["mae"]) / exceso for k, r in v.items()}
    L.append(
        "\n**3. ¿Se confirma la causa del colapso?** (ridge ajustado de la Tarea 2, partición prepandemia; «exceso» = MAE de la referencia − MAE de la configuración por defecto)"
    )
    L.append(
        f"- Se reproduce el colapso: MAE {_num(ref['mae'])} frente a {_num(dflt['mae'])} por defecto (exceso {_num(exceso)}); por año, 2018: {_num(ref['mae_2018'])} y 2019: {_num(ref['mae_2019'])}. "
        f"{int(ref['peores_en_bogota'])} de las {int(ref['n_peores'])} filas de mayor error son de Bogotá (MAE en Bogotá {_num(ref['mae_bogota'])} frente a {_num(ref['mae_resto'])} en el resto)."
    )
    L.append(
        f"- **(i) Quitar `mpio_matriculados_t2`:** MAE {_num(v['i_sin_variable']['mae'])}, elimina {quita['i_sin_variable']:.0f} % del exceso. "
        + (
            "Esa variable explica la mayor parte del colapso."
            if quita["i_sin_variable"] >= 75
            else "Esa variable explica solo una parte del colapso."
            if quita["i_sin_variable"] >= 25
            else "Esa variable no explica el colapso."
        )
    )
    L.append(
        f"- **(ii) Alfa por defecto:** con `RidgeCV` y el mismo k, MAE {_num(v['ii_alfa_defecto']['mae'])} (elimina {quita['ii_alfa_defecto']:.0f} %); con la configuración por defecto completa (k = {int(dflt['k'])}) el coeficiente de la variable es {_num(dflt['coef_variable'], 3)} frente a {_num(ref['coef_variable'], 3)} en la referencia: "
        + (
            "con poca regularización y k = 20 el modelo le da un peso grande."
            if abs(ref["coef_variable"]) > 5 * abs(dflt["coef_variable"])
            else "el peso de la variable es parecido."
        )
    )
    L.append(
        f"- **(iii) Recortar las variables escaladas a [-3, 3]:** MAE {_num(v['iii_recorte']['mae'])} ({quita['iii_recorte']:.0f} % del exceso). "
        + (
            "El recorte no corrige el colapso: no se debe a valores extremos (|z| > 3)."
            if quita["iii_recorte"] < 25
            else "El recorte corrige parte del colapso: hay valores extremos."
        )
    )
    L.append(
        f"- **(iv, adicional) Entrenar solo con 2017** (único año con SNIES observado): MAE {_num(v['iv_solo_2017']['mae'])} ({quita['iv_solo_2017']:.0f} % del exceso; {int(v['iv_solo_2017']['n_entrenamiento'])} filas de entrenamiento). "
        + (
            "Con SNIES observado el colapso persiste: la hipótesis del SNIES imputado no se sostiene como causa principal."
            if quita["iv_solo_2017"] < 50
            else "Sin las filas con SNIES imputado el colapso desaparece en buena parte: apoya la hipótesis del SNIES imputado."
        )
    )
    L.append(
        "- **Respuesta:** "
        + (
            "se confirma una causa **específica de una variable** (`mpio_matriculados_t2` con un coeficiente grande), no una extrapolación a valores extremos ni, por sí sola, la imputación del SNIES; "
            if quita["i_sin_variable"] >= 75 and quita["iii_recorte"] < 25
            else "la causa se confirma solo en parte (ver los puntos anteriores); "
        )
        + "el detalle de z por variable y las filas de mayor error está en las tablas de abajo y en `colapso_ridge_z.csv` y `colapso_ridge_peores.csv`."
    )
    # 4. variables
    L.append("\n**4. ¿Qué variables usan los modelos y coinciden los métodos?**")
    for modelo, etiqueta in (
        (XGB_T1, "Tarea 1, B xgboost"),
        (XGB_T2, "Tarea 2, xgboost con clusters"),
    ):
        g = top[top["modelo"] == modelo].head(5)
        rho, comun, n = coincidencia(grupos, modelo)
        L.append(
            f"- **{etiqueta}.** Top 5 por |SHAP|: "
            + ", ".join(
                f"`{r['variable']}` ({_num(r['mean_abs_shap'], 3)})" for _, r in g.iterrows()
            )
            + f". Frente a la importancia por permutación: Spearman entre los órdenes {_num(rho, 2)} ({n} variables originales), {comun} de 5 variables en común en el top 5."
        )
    c = coef[coef["modelo"] == RIDGE_T2]
    estables = c[c["signo_constante"] & (c["n_pliegues_seleccionada"] >= c["de_pliegues"] / 2)]
    L.append(
        f"- **Ridge de la Tarea 2:** {len(estables)} de {len(c)} variables que entran en al menos un pliegue lo hacen en la mitad o más con signo constante; las mayores: "
        + ", ".join(
            f"`{r['variable']}` ({_num(r['media'], 3)})" for _, r in estables.head(4).iterrows()
        )
        + "."
    )
    return "\n".join(L)


# ---------- actualización de 05_sesgo_dataset.md ----------
MARCA_ES = ("<!-- sesgo_final:inicio -->", "<!-- sesgo_final:fin -->")
VARIABLE_EN = {
    "sector": "sector",
    "caracter": "character",
    "region": "region",
    "es_sena": "SENA",
    "tamano": "size",
    "falta_historia": "history",
}


def _parrafos_sesgo(sub, sens, col, top, grupos) -> tuple[str, str]:
    """(párrafos en español, párrafos en inglés) con las cifras medidas en los modelos finales: la medición y el nuevo «Poder predictivo»."""
    firmes = sub[sub["clasificacion"].str.startswith("peor que")]
    n_en = lambda x, d=4: _num(x, d).replace(",", ".")  # noqa: E731  (lambda local de un solo uso)
    lista = lambda d, en=False: (  # noqa: E731  (lambda local de un solo uso)
        "; ".join(
            f"{r['modelo'].split(' (')[0]}: {(VARIABLE_EN[r['variable']] if en else r['variable'])} {r['nivel']} ({(n_en(r['dif']) if en else _num(r['dif'], 4))})"
            for _, r in d.iterrows()
        )
        or ("none" if en else "ninguno")
    )
    camb = sens[sens["cambia_conclusion"]]
    ref, dflt = (
        col[col["variante"] == "referencia"].iloc[0],
        col[col["variante"] == "ii_defecto_completo"].iloc[0],
    )
    sin_v = col[col["variante"] == "i_sin_variable"].iloc[0]
    rho1, c1, _ = coincidencia(grupos, XGB_T1)
    rho2, c2, _ = coincidencia(grupos, XGB_T2)
    b = sens[(sens["escenario"] == "base") & (sens["tipo"] == "modelo")].set_index("configuracion")
    r1, x1, r2, x2 = b.loc[RIDGE_T1], b.loc[XGB_T1], b.loc[RIDGE_T2], b.loc[XGB_T2]
    mae_ing = sens[(sens["escenario"] == "base") & (sens["configuracion"] == "T1 base_ingenua")][
        "mae"
    ].iloc[0]
    firme = lambda f: f["clase"] == "supera"  # noqa: E731  (lambda local de un solo uso)
    detalle_es = "; ".join(
        f"{r['escenario']}, {r['configuracion']}: «{r['clase_base']}» → «{r['clase']}»"
        for _, r in camb.iterrows()
    )
    detalle_en = "; ".join(
        f"{r['escenario']}, {r['configuracion']}: '{r['clase_base']}' → '{r['clase']}'"
        for _, r in camb.iterrows()
    )
    es_med = (
        f"**Medición en los modelos finales** (GroupKFold por IES 5 × 5; detalle en `reports/06_sesgo_modelos.md`). Subgrupos donde un modelo es peor que su línea base con IC 95 % que excluye 0: "
        f"Tarea 1, {lista(firmes[firmes['tarea'] == 'T1'])}; Tarea 2, {lista(firmes[firmes['tarea'] == 'T2'])} (SENA, con una sola IES, no tiene IC válido). "
        + (
            "Ninguno de los cuatro escenarios de sensibilidad (sin Bogotá, denominador mínimo 200, sin SENA, solo 2017–2021) cambia la clase de ningún modelo frente a su línea base. "
            if camb.empty
            else f"En {len(camb)} combinaciones de escenario y modelo cambia la clase frente a la línea base ({detalle_es}): la clase de varios modelos depende de qué IES pequeñas se incluyan y de cuántos datos hay. "
        )
        + f"El colapso del ridge ajustado de la Tarea 2 en la prepandemia (MAE {_num(ref['mae'])} frente a {_num(dflt['mae'])} por defecto) se reproduce: {int(ref['peores_en_bogota'])} de las 40 filas de mayor error son de Bogotá, y quitar `mpio_matriculados_t2` lo baja a {_num(sin_v['mae'])}. "
        f"Los métodos de interpretación coinciden en lo principal: Spearman entre SHAP y permutación de {_num(rho1, 2)} (Tarea 1) y {_num(rho2, 2)} (Tarea 2), y {c1} y {c2} de las 5 variables principales en común."
    )
    es_pp = (
        f"**Poder predictivo.** Con los modelos finales (hiperparámetros ajustados), en la Tarea 1 el modelo con más ventaja es C solo historia con ridge (MAE {_num(r1['mae'])} frente a {_num(mae_ing)} de la ingenua, {_pct(r1['mejora_base_pct'])}; "
        f"{'supera' if firme(r1) else 'no supera'} a la ingenua con firmeza) y B con xgboost queda en {_pct(x1['mejora_base_pct'])} ({'con' if firme(x1) else 'sin'} firmeza). En la Tarea 2 ningún modelo supera con firmeza a la mediana por carácter × sector: "
        f"ridge con información mutua {_pct(r2['mejora_base_pct'])} y xgboost con clusters {_pct(x2['mejora_base_pct'])}. Las explicativas de contexto casi no agregan información más allá de la historia de cada IES, de su carácter y de su sector."
    )
    en_med = (
        f"**Measured on the final models** (GroupKFold by institution 5 × 5; details in `reports/06_sesgo_modelos.md`). Subgroups where a model is worse than its baseline with a 95% CI that excludes 0: "
        f"Task 1, {lista(firmes[firmes['tarea'] == 'T1'], True)}; Task 2, {lista(firmes[firmes['tarea'] == 'T2'], True)} (SENA, a single institution, has no valid CI). "
        + (
            "None of the four sensitivity scenarios (without Bogotá, minimum denominator 200, without SENA, only 2017–2021) changes the class of any model relative to its baseline. "
            if camb.empty
            else f"In {len(camb)} scenario–model combinations the class relative to the baseline changes ({detalle_en}): the class of several models depends on which small institutions are included and on how much data there is. "
        )
        + f"The collapse of the tuned Task 2 ridge in the pre-pandemic split (MAE {n_en(ref['mae'])} vs. {n_en(dflt['mae'])} with defaults) is reproduced: {int(ref['peores_en_bogota'])} of the 40 largest errors are in Bogotá, and removing `mpio_matriculados_t2` brings it down to {n_en(sin_v['mae'])}. "
        f"The interpretation methods agree on the main points: Spearman between SHAP and permutation importance is {n_en(rho1, 2)} (Task 1) and {n_en(rho2, 2)} (Task 2), with {c1} and {c2} of the top 5 variables in common."
    )
    en_pp = (
        f"**Predictive power.** With the final (tuned) models, in Task 1 the model with the largest advantage is the C history-only ridge (MAE {n_en(r1['mae'])} vs. {n_en(mae_ing)} for the naive baseline, "
        f"{_pct(r1['mejora_base_pct']).replace(',', '.')}; it {'does' if firme(r1) else 'does not'} beat the naive baseline firmly), and the B xgboost reaches {_pct(x1['mejora_base_pct']).replace(',', '.')} ({'with' if firme(x1) else 'without'} firmness). "
        f"In Task 2 no model beats the median by character × sector firmly: ridge with mutual information {_pct(r2['mejora_base_pct']).replace(',', '.')} and xgboost with clusters {_pct(x2['mejora_base_pct']).replace(',', '.')}. "
        "The context variables add almost nothing beyond each institution's own history, its character and its sector."
    )
    lf = chr(10)
    return es_med + lf + lf + es_pp, en_med + lf + lf + en_pp


def actualizar_analisis_sesgo(ruta: str | Path, es: str, en: str) -> None:
    """Reemplaza el párrafo «Poder predictivo» (y su versión en inglés) por la medición en los modelos finales, entre marcas para poder regenerarlo.

    Conserva los saltos de línea del archivo. La primera vez quita el párrafo antiguo (basado en el ridge de referencia); luego solo reemplaza lo que hay entre marcas."""
    with open(ruta, encoding="utf-8", newline="") as f:
        t = f.read()
    lf, crlf = chr(10), chr(13) + chr(10)
    eol = crlf if crlf in t else lf
    ini, fin = MARCA_ES
    bloque = lambda txt: (ini + lf + txt + lf + fin).replace(lf, eol)  # noqa: E731  (lambda local de un solo uso)
    existentes = list(re.finditer(re.escape(ini) + r".*?" + re.escape(fin), t, re.S))
    if (
        len(existentes) == 2
    ):  # ya estaban: se reemplazan (primero el segundo, para no mover los índices)
        for m, txt in ((existentes[1], en), (existentes[0], es)):
            t = t[: m.start()] + bloque(txt) + t[m.end() :]
    else:  # primera vez: el párrafo antiguo de poder predictivo, hasta la línea en blanco o el fin del archivo
        for titulo, txt in (("**Poder predictivo.**", es), ("**Predictive power.**", en)):
            antiguo = re.compile(re.escape(titulo) + r".*?(?=" + eol + eol + r"|\Z)", re.S)
            assert antiguo.search(t), f"no se encontró el párrafo {titulo}"
            t = antiguo.sub(lambda _m: bloque(txt), t, count=1)  # noqa: B023  (se usa dentro de la misma iteración)
    with open(ruta, "w", encoding="utf-8", newline="") as f:
        f.write(t)


# ---------- informe ----------
def generar_informe(cfg: dict, s: dict[str, Path], tiempos: dict) -> str:
    from desercion_ies.features.particion import cargar_dataset

    p = cfg["modelado"]["sesgo_sensibilidad"]
    sub, sens, col = (pd.read_csv(s[k]) for k in ("subgrupos", "sensibilidad", "colapso"))
    z, pe = pd.read_csv(s["colapso_z"]), pd.read_csv(s["colapso_peores"])
    top, grupos, perm, coef = (
        pd.read_csv(s[k]) for k in ("shap_top", "shap_grupos", "permutacion", "coeficientes")
    )
    hp = pd.read_csv(s["hiperparametros"])
    df = cargar_dataset(cfg)
    figura_subgrupos(sub, s["figura_subgrupos"])
    figura_shap(
        cfg,
        df,
        XGB_T1,
        s["figura_shap_t1"],
        "SHAP — Tarea 1: B historia + contexto (xgboost, ajustado)",
    )
    figura_shap(
        cfg, df, XGB_T2, s["figura_shap_t2"], "SHAP — Tarea 2: xgboost con clusters (ajustado)"
    )
    figura_coeficientes(coef, s["figura_coeficientes"])
    cortes = tiempos.get("cortes_tamano")
    cortes_txt = (
        f"{_num(cortes[1], 0)} y {_num(cortes[2], 0)} matriculados (t-2)"
        if cortes
        else "terciles de la matrícula"
    )
    simple_t1 = tabla_simple(top, perm, coef, XGB_T1, None)
    simple_t2 = tabla_simple(top, perm, coef, XGB_T2, RIDGE_T2)
    esc_desc = {  # noqa: F841  (se conserva: documenta el valor)
        "base": "dataset completo",
        "sin_bogota": "sin Bogotá",
        "umbral_200": "denominador mínimo 200",
        "sin_sena": "sin SENA",
        "desde_2017": "solo 2017–2021",
    }
    tiempo = (
        f"{tiempos.get('t_total_s', 0):.0f} s ({_num(tiempos.get('t_total_s', 0) / 60, 1)} min)"
    )
    texto = f"""# Sesgo medido en los modelos finales, sensibilidad e interpretabilidad

Generado por `desercion evaluar sesgo` (módulos `src/desercion_ies/evaluation/evaluacion_final.py`, `subgrupos.py`, `sensibilidad.py`, `diagnostico_colapso.py` e `interpretabilidad.py`). Todas las cifras salen de la corrida; no se editan a mano. Continúa `reports/05_sesgo_dataset.md` (donde se declararon los sesgos), `reports/02_tarea1.md`, `reports/03_tarea2.md` y `reports/04_ajuste_temporal.md`.

## Diseño

- **Modelos finales.** Tarea 1: naive (tasa de t-1), C solo historia con ridge (ajustado) y B historia + contexto con xgboost (ajustado). Tarea 2: mediana por carácter × sector, ridge con información mutua (por defecto y ajustado) y xgboost con clusters (ajustado).
- **Hiperparámetros finales.** En el ajuste anidado cada pliegue externo eligió los suyos; los finales son el consenso de los 25 pliegues (valor más frecuente; mediana en los continuos). Esa combinación no se validó como tal en el ajuste, y como salió de los mismos datos el resultado es algo optimista (igual en todos los escenarios).

{_tabla_hp(hp)}

- **Validación.** GroupKFold por IES, 5 pliegues × las semillas 301–305 (como en el ajuste); predicciones recortadas a [0, 1]. El error de cada fila es el promedio de sus 5 predicciones fuera de muestra (una por repetición).
- **Subgrupos.** Sector, carácter, región (Bogotá, Medellín, Cali y resto), SENA, tamaño (terciles de `log_matriculados_spadies_t2`; cortes en {cortes_txt}) y `falta_historia`. IC 95 % por bootstrap sobre IES (1.000 réplicas, semilla {p["bootstrap"]["semilla"]}; el mismo remuestreo para el modelo y su línea base, de modo que la diferencia es pareada). Los subgrupos de menos de {N_PEQUENO} filas se marcan «n pequeño».
- **Sensibilidad.** Se re-entrena todo con el mismo protocolo en el dataset ya construido (no se re-corre el ETL): sin Bogotá (entrenamiento y prueba), denominador mínimo 200, sin SENA y solo 2017–2021. Los hiperparámetros finales **no se vuelven a ajustar** en cada escenario. «Cambia la conclusión» = cambia la clase frente a la línea base (`supera`, `empeora` o `sin diferencia firme`, exigiendo significancia en Wilcoxon y t corregida de Nadeau y Bengio).
- **Tiempo de la corrida:** {tiempo}.

## Conclusión

{_conclusion(sub, sens, col, top, perm, grupos, coef)}

## 1. Error por subgrupo

La diferencia es MAE de la línea base − MAE del modelo: **negativa = el modelo es peor** (en negrita cuando el IC excluye 0). La lectura «sin firmeza» significa que la diferencia puntual es negativa pero el IC incluye 0.

### Tarea 1 · C solo historia, ridge (ajustado) frente a la ingenua

{_tabla_subgrupos(sub, RIDGE_T1)}

### Tarea 1 · B historia + contexto, xgboost (ajustado) frente a la ingenua

{_tabla_subgrupos(sub, XGB_T1)}

### Tarea 2 · ridge con información mutua (ajustado) frente a la mediana por carácter × sector

{_tabla_subgrupos(sub, RIDGE_T2)}

### Tarea 2 · ridge con información mutua (por defecto) frente a la mediana por carácter × sector

{_tabla_subgrupos(sub, "T2 ridge info_mutua (defecto)")}

### Tarea 2 · xgboost con clusters (ajustado) frente a la mediana por carácter × sector

{_tabla_subgrupos(sub, XGB_T2)}

![Error por subgrupo](figures/error_subgrupos.png)

## 2. Sensibilidad

MAE de la línea base de la tarea con IC 95 % (Nadeau y Bengio), y de cada modelo con su mejora frente a la línea base y su clase en cada escenario.

### Tarea 1

{_tabla_sensibilidad(sens, "T1")}

### Tarea 2

{_tabla_sensibilidad(sens, "T2")}

## 3. Diagnóstico reproducible del colapso del ridge ajustado de la Tarea 2 (prepandemia)

Solo diagnóstico: no cambia los modelos finales. Se entrena con 2015–2017 y se prueba en 2018–2019 con los hiperparámetros que la búsqueda interna eligió en esa partición. «Exceso de error que elimina» = (MAE de la referencia − MAE de la variante) / (MAE de la referencia − MAE de la configuración por defecto completa).

{_tabla_colapso(col)}

### z de las variables de prueba respecto del entrenamiento 2015–2017 (las 10 con mayor |z| máximo)

z de cada variable ya imputada y escalada, calculado con la media y la desviación del entrenamiento; umbral {p["colapso"]["z_umbral"]:g}. Tabla completa en `reports/tables/colapso_ridge_z.csv`.

{_tabla_z(z)}

### Las 10 filas de mayor error (de {p["colapso"]["n_peores"]}; en `colapso_ridge_peores.csv`)

{_tabla_peores(pe)}

## 4. Interpretabilidad

SHAP con `TreeExplainer`, ajustado con todo el dataset y los hiperparámetros finales. xgboost se entrena sobre el logit de la tasa: un SHAP positivo sube la deserción predicha. La dirección es el signo de la correlación de Spearman entre el valor de la variable y su SHAP.

### Tarea 1 · B historia + contexto (xgboost): top 10 por |SHAP| medio

{_tabla_shap(top, XGB_T1)}

![SHAP Tarea 1](figures/shap_t1.png)

### Tarea 2 · xgboost con clusters: top 10 por |SHAP| medio

{_tabla_shap(top, XGB_T2)}

![SHAP Tarea 2](figures/shap_t2.png)

### Importancia por permutación en validación (GroupKFold por IES, 5 pliegues)

Aumento del MAE al permutar cada variable **original** en el pliegue de prueba (5 repeticiones por pliegue).

**Tarea 1 · B xgboost**

{_tabla_permutacion(perm, XGB_T1)}

**Tarea 2 · xgboost con clusters**

{_tabla_permutacion(perm, XGB_T2)}

### Coeficientes del ridge y su estabilidad (25 pliegues de entrenamiento)

Coeficientes sobre variables estandarizadas. Tarea 2: ridge con información mutua (k variables por pliegue); Tarea 1: C solo historia, que predice el **cambio** frente a la naive.

**Tarea 2 · ridge con información mutua (ajustado)**

{_tabla_coef(coef, RIDGE_T2)}

**Tarea 1 · C solo historia, ridge (ajustado)**

{_tabla_coef(coef, RIDGE_T1)}

![Coeficientes del ridge](figures/coeficientes_ridge.png)

### En lenguaje simple: qué hace cada variable y si los métodos coinciden

«Consistente» = la variable está entre las 10 primeras por permutación y su dirección no contradice al ridge cuando éste la usa de forma estable; «parcial» = aparece en un solo método; «no» = el ridge la contradice.

**Tarea 1 · B historia + contexto (xgboost).** El ridge de la Tarea 1 predice el cambio frente a la naive (otro objetivo), así que no se usa para la dirección.

{_md_simple(simple_t1)}

**Tarea 2 · xgboost con clusters (con el ridge de la Tarea 2 como tercer método).**

{_md_simple(simple_t2)}

## Supuestos y limitaciones

- **Hiperparámetros finales por consenso:** no se validaron como combinación y salieron de los mismos datos; el resultado es algo optimista, igual en todos los escenarios. En la sensibilidad no se re-ajustan.
- **IC por bootstrap sobre IES:** recoge la variabilidad por la muestra de IES (la relevante para un subgrupo), no la de elegir otros pliegues; el error de cada fila es el promedio de sus 5 predicciones fuera de muestra. Con muchos subgrupos y modelos hay comparaciones múltiples: las marcas «peor» son una guía, no pruebas.
- **Subgrupos pequeños:** SENA, y varios de carácter o región, tienen pocas filas o IES; se marcan «n pequeño» y no deben leerse en detalle.
- **SHAP y permutación** describen lo que el modelo usa, no causas: con variables territoriales casi constantes dentro de una ciudad, el modelo puede estar identificando la ciudad. Las categorías de sector y carácter se suman por variable original; los clusters (derivados de todas las variables) se muestran aparte.
- **Dirección por Spearman:** resume una relación posiblemente no monótona; «sin dirección clara» significa |ρ| < 0,2.
- **El diagnóstico del colapso** usa una sola partición temporal; que quitar una variable lo corrija no prueba que esa variable sea la causa de fondo, solo que el modelo la usa mal en esa partición.
- El sesgo de selección, supervivencia y geografía de `reports/05_sesgo_dataset.md` aplica igual a estos resultados.
"""
    s["informe"].parent.mkdir(parents=True, exist_ok=True)
    s["informe"].write_text(texto, encoding="utf-8")
    es, en = _parrafos_sesgo(sub, sens, col, top, grupos)
    actualizar_analisis_sesgo(Path("reports/05_sesgo_dataset.md"), es, en)
    return texto
