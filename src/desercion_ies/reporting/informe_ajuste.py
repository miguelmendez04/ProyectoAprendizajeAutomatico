"""Informe del ajuste de hiperparámetros, las particiones temporales y el arranque en frío estricto.

`reports/04_ajuste_temporal.md` y las figuras salen de los CSV que escribe `desercion evaluar ajuste`; ninguna cifra se escribe a mano.
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from desercion_ies.analysis.descriptivo_temporal import descriptivo_temporal
from desercion_ies.reporting.informe_experimentos import _num, _pct
from desercion_ies.reporting.informe_v3 import _c, _ic, _p, _veredicto

BASE_T1, BASE_T2 = "T1 base_ingenua", "T2 base_mediana_grupo"
REF = {"T1": BASE_T1, "T2": BASE_T2}
NOMBRE_REF = {"T1": "ingenua (tasa de t-1)", "T2": "mediana por carácter × sector"}
COLORES = {"defecto": "#9ecae1", "ajustado": "#08519c"}


def corto(nombre: str) -> str:
    return (
        nombre.replace("T1 ", "", 1).replace("T2 | ", "", 1)
        if nombre.startswith(("T1 ", "T2 | "))
        else nombre
    )


def _ci(lo: float, hi: float) -> str:
    return f"[{_num(lo, 4)}; {_num(hi, 4)}]"


def _veredicto_ci(lo: float, hi: float) -> str:
    return (
        "gana (el IC excluye 0)"
        if lo > 0
        else ("pierde (el IC excluye 0)" if hi < 0 else "no distinguible (el IC incluye 0)")
    )


# ---------- tablas del ajuste ----------
def _tabla_ajuste(r: pd.DataFrame, tarea: str) -> str:
    base = r[(r["tarea"] == tarea) & (r["variante"] == "linea_base")]
    ref = REF[tarea]
    out = [
        "| Configuración | MAE por defecto [IC 95 %] | MAE ajustado [IC 95 %] | Mejora por el ajuste [IC 95 %] | p t corregida (ajuste) | Defecto vs. línea base (mejora; p t) | Ajustado vs. línea base (mejora; p Wilcoxon; p t; gana en) |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, f in base[base["configuracion"].isin([ref, f"{tarea} base_mediana"])].iterrows():
        out.append(f"| **{f['configuracion']}** | — | {_ic(f)} | — | — | — | — |")
    d = r[(r["tarea"] == tarea) & (r["variante"] == "defecto")].set_index("configuracion")
    for _, a in r[(r["tarea"] == tarea) & (r["variante"] == "ajustado")].iterrows():
        x = d.loc[a["configuracion"]]
        out.append(
            f"| {_c(corto(a['configuracion']))} | {_ic(x)} | {_ic(a)} | {_num(a['dif_defecto'], 4)} {_ci(a['dif_defecto_ic_inf'], a['dif_defecto_ic_sup'])} | {_p(a['p_t_defecto'])} "
            f"| {_pct(x['mejora_base_pct'])}; {_p(x['p_t_base'])} | {_pct(a['mejora_base_pct'])}; {_p(a['p_wilcoxon_base'])}; {_p(a['p_t_base'])}; {100 * a['gana_base']:.0f} % |"
        )
    return "\n".join(out)


def _tabla_hiperparametros(h: pd.DataFrame) -> str:
    out = ["| Configuración | Hiperparámetro | Elegido en los pliegues externos |", "|---|---|---|"]
    for _, f in h.iterrows():
        out.append(
            f"| {_c(corto(f['configuracion']))} | `{f['hiperparametro']}` | {re.sub(r'(?<=\d)[.](?=\d)', ',', str(f['resumen']))} |"
        )
    return "\n".join(out)


def _tabla_tiempos(r: pd.DataFrame) -> str:
    out = [
        "| Configuración | Variante | Ajuste medio (s) | Predicción media (s) | Búsqueda media (s) |",
        "|---|---|---|---|---|",
    ]
    for _, f in r[r["variante"] != "linea_base"].iterrows():
        bu = "—" if pd.isna(f["t_busqueda_media_s"]) else _num(f["t_busqueda_media_s"], 1)
        out.append(
            f"| {_c(corto(f['configuracion']))} | {f['variante']} | {_num(f['t_ajuste_media_s'], 3)} | {_num(f['t_prediccion_media_s'], 4)} | {bu} |"
        )
    return "\n".join(out)


# ---------- tablas temporales ----------
def _tabla_temporal(tm: pd.DataFrame, particion: str, tarea: str) -> str:
    g = tm[(tm["particion"] == particion) & (tm["tarea"] == tarea)]
    ref = REF[tarea]
    out = [
        "| Configuración | Variante | MAE [IC 95 %] | RMSE [IC 95 %] | R² [IC 95 %] | Diferencia de MAE vs. "
        + NOMBRE_REF[tarea]
        + " [IC 95 %] | Gana en (réplicas) |",
        "|---|---|---|---|---|---|---|",
    ]
    orden = list(g[g["variante"] == "linea_base"].index) + list(
        g[g["variante"] != "linea_base"].sort_values(["configuracion", "variante"]).index
    )
    for i in orden:
        f = g.loc[i]
        d = (
            "—"
            if f["configuracion"] == ref
            else f"{_num(f['dif_ref'], 4)} {_ci(f['dif_ref_ic_inf'], f['dif_ref_ic_sup'])}"
        )
        gana = "—" if f["configuracion"] == ref else f"{100 * f['prop_gana_ref']:.0f} %"
        neg = "**" if f["variante"] == "linea_base" else ""
        out.append(
            f"| {neg}{_c(corto(f['configuracion']))}{neg} | {f['variante'].replace('linea_base', 'línea base')} | {_num(f['mae'], 4)} {_ci(f['mae_ic_inf'], f['mae_ic_sup'])} | {_num(f['rmse'], 4)} {_ci(f['rmse_ic_inf'], f['rmse_ic_sup'])} "
            f"| {_num(f['r2'], 3)} [{_num(f['r2_ic_inf'], 3)}; {_num(f['r2_ic_sup'], 3)}] | {d} | {gana} |"
        )
    return "\n".join(out)


def _tabla_grupos(gr: pd.DataFrame, particion: str, tarea: str) -> str:
    g = gr[
        (gr["particion"] == particion)
        & (gr["tarea"] == tarea)
        & (gr["variante"].isin(["ajustado", "linea_base"]))
    ]
    cols = [
        ("sector", "oficial"),
        ("sector", "privada"),
        ("caracter", "técnica profesional"),
        ("caracter", "tecnológica"),
        ("caracter", "institución universitaria"),
        ("caracter", "universidad"),
    ]
    anios = sorted(g.loc[g["variable"] == "anio", "nivel"].unique())
    cols += [("anio", a) for a in anios]
    if "bogota" in set(g["variable"]):
        cols += [("bogota", "Bogotá"), ("bogota", "resto")]
    out = [
        "| Configuración | "
        + " | ".join(
            f"año {n}" if v == "anio" else (f"{n}" if v == "bogota" else f"{v}: {n}")
            for v, n in cols
        )
        + " |",
        "|---|" + "---|" * len(cols),
    ]
    primero = g.iloc[0]["configuracion"]
    n = g[g["configuracion"] == primero]
    out.append(
        "| *filas de prueba* | "
        + " | ".join(
            str(int(n[(n["variable"] == v) & (n["nivel"] == nv)]["n"].sum())) for v, nv in cols
        )
        + " |"
    )
    orden = list(dict.fromkeys(g[g["variante"] == "linea_base"]["configuracion"])) + list(
        dict.fromkeys(g[g["variante"] == "ajustado"]["configuracion"])
    )
    for cfg in orden:
        x = g[g["configuracion"] == cfg]
        celdas = []
        for v, nv in cols:
            m = x[(x["variable"] == v) & (x["nivel"] == nv)]["mae"]
            celdas.append(_num(m.iloc[0], 4) if len(m) else "—")
        neg = "**" if cfg in (BASE_T1, BASE_T2) or "base_" in cfg else ""
        out.append(f"| {neg}{_c(corto(cfg))}{neg} | " + " | ".join(celdas) + " |")
    return "\n".join(out)


def _tabla_degradacion(tm: pd.DataFrame, tarea: str) -> str:
    g = tm[(tm["tarea"] == tarea) & (tm["variante"].isin(["ajustado", "linea_base"]))]
    a = g[g["particion"] == "prepandemia"].set_index("configuracion")
    b = g[g["particion"] == "estres"].set_index("configuracion")
    out = [
        "| Configuración | MAE prepandemia (2018–2019) | MAE estrés (2020–2021) | Cambio | Razón estrés / prepandemia |",
        "|---|---|---|---|---|",
    ]
    for cfg in a.index:
        out.append(
            f"| {_c(corto(cfg))} | {_num(a.loc[cfg, 'mae'])} | {_num(b.loc[cfg, 'mae'])} | {_num(b.loc[cfg, 'mae'] - a.loc[cfg, 'mae'], 4)} | {_num(b.loc[cfg, 'mae'] / a.loc[cfg, 'mae'], 2)} |"
        )
    return "\n".join(out)


# ---------- arranque en frío ----------
def _tabla_pliegues(t: pd.DataFrame) -> str:
    out = [
        "| Repetición (semilla) | Filas de prueba por pliegue | % del total | Pliegue con Bogotá (filas de Bogotá) | Municipios por pliegue |",
        "|---|---|---|---|---|",
    ]
    for rep, g in t.groupby("repeticion"):
        b = g[g["contiene_bogota"]].iloc[0]
        out.append(
            f"| {rep} ({int(g['semilla'].iloc[0])}) | {', '.join(str(int(x)) for x in g['n_prueba'])} | {', '.join(f'{x:.0f}' for x in g['pct_prueba'])} | pliegue {int(b['pliegue'])} ({int(b['filas_bogota'])}) | {', '.join(str(int(x)) for x in g['n_municipios_prueba'])} |"
        )
    return "\n".join(out)


def _tabla_frio(frio: pd.DataFrame, ies: pd.DataFrame) -> str:
    i = ies.set_index("configuracion")
    out = [
        "| Configuración | MAE por IES [IC 95 %] | MAE por municipio [IC 95 %] | Mejora vs. mediana por grupo, por municipio | p Wilcoxon | p t corregida | Gana en | Mejora vs. mediana por grupo, por IES |",
        "|---|---|---|---|---|---|---|---|",
    ]
    orden = list(
        frio[frio["tipo"] == "linea_base"].sort_values("mae_media")["configuracion"]
    ) + list(frio[frio["tipo"] == "modelo"].sort_values("mae_media")["configuracion"])
    f_ = frio.set_index("configuracion")
    for n in orden:
        f = f_.loc[n]
        propia = n == "base_mediana_grupo"
        x = i.loc[n] if n in i.index else None
        neg = "**" if f["tipo"] == "linea_base" else ""
        out.append(
            f"| {neg}{_c(n)}{neg} | {_ic(x) if x is not None else '—'} | {_ic(f)} | {'—' if propia else _pct(f['mejora_grupo_pct'])} | {'—' if propia else _p(f['p_wilcoxon_grupo'])} | "
            f"{'—' if propia else _p(f['p_t_grupo'])} | {'—' if propia else str(round(100 * f['gana_grupo'])) + ' %'} | {'—' if x is None or propia else _pct(x['mejora_grupo_pct'])} |"
        )
    return "\n".join(out)


# ---------- conclusión ----------
def _clase(dif: float, pw: float, pt: float) -> str:
    """Cambio que importa frente a la línea base: `supera` o `empeora` solo si las DOS pruebas son significativas; si no, `sin diferencia firme`."""
    firme = (not np.isnan(pw)) and (not np.isnan(pt)) and pw < 0.05 and pt < 0.05
    return ("supera" if dif > 0 else "empeora") if firme else "sin diferencia firme"


def _conclusion(
    r: pd.DataFrame,
    tm: pd.DataFrame,
    gr: pd.DataFrame,
    frio: pd.DataFrame,
    ies: pd.DataFrame,
    tam: pd.DataFrame,
) -> str:
    lineas = []
    # 1. ¿el ajuste cambia alguna conclusión?
    lineas.append(
        "**1. ¿El ajuste de hiperparámetros cambia alguna conclusión?** (validación anidada, 25 ajustes por configuración; diferencia positiva = menos error)"
    )
    d = r[r["variante"] == "defecto"].set_index("configuracion")
    cambios = 0
    for _, a in r[r["variante"] == "ajustado"].iterrows():
        x = d.loc[a["configuracion"]]
        c0, c1 = (
            _veredicto(x["dif_base"], x["p_wilcoxon_base"], x["p_t_base"]),
            _veredicto(a["dif_base"], a["p_wilcoxon_base"], a["p_t_base"]),
        )
        k0, k1 = (
            _clase(x["dif_base"], x["p_wilcoxon_base"], x["p_t_base"]),
            _clase(a["dif_base"], a["p_wilcoxon_base"], a["p_t_base"]),
        )
        cambia = k0 != k1
        cambios += int(cambia)
        lineas.append(
            f"- `{corto(a['configuracion'])}`: MAE {_num(x['mae_media'])} por defecto → {_num(a['mae_media'])} ajustado (el ajuste cambia {_num(a['dif_defecto'], 4)} {_ci(a['dif_defecto_ic_inf'], a['dif_defecto_ic_sup'])}, p t corregida {_p(a['p_t_defecto'])}). "
            f"Frente a la línea base: por defecto {_pct(x['mejora_base_pct'])} ({c0}); ajustado {_pct(a['mejora_base_pct'])} ({c1}; p Wilcoxon {_p(a['p_wilcoxon_base'])}, p t corregida {_p(a['p_t_base'])}). "
            + (
                f"**Cambia la conclusión** (de «{k0}» a «{k1}»)."
                if cambia
                else f"No cambia la conclusión («{k1}»)."
            )
        )
    sup = r[
        (r["variante"] == "ajustado")
        & (r["mejora_base_pct"] > 0)
        & (r["p_wilcoxon_base"] < 0.05)
        & (r["p_t_base"] < 0.05)
    ]
    mejoran = r[(r["variante"] == "ajustado") & (r["dif_defecto"] > 0) & (r["p_t_defecto"] < 0.05)]
    lineas.append(
        f"  El ajuste mejora de forma significativa (t corregida < 0,05) a su versión por defecto en {len(mejoran)} de {int((r['variante'] == 'ajustado').sum())} configuraciones"
        + (
            ": " + ", ".join(f"`{corto(n)}`" for n in mejoran["configuracion"])
            if len(mejoran)
            else ""
        )
        + "."
    )
    lineas.append(
        f"  «Cambia la conclusión» significa que cambia la clase frente a la línea base (`supera`, `empeora` o `sin diferencia firme`, exigiendo significancia en las dos pruebas): ocurre en {cambios} de {int((r['variante'] == 'ajustado').sum())} configuraciones; "
        + (
            f"superan a su línea base con significancia en las dos pruebas: {', '.join('`' + corto(n) + '`' for n in sup['configuracion'])}."
            if len(sup)
            else "ninguna configuración ajustada supera a su línea base con significancia en las dos pruebas."
        )
    )
    # 2. ventaja de la naive en el tiempo
    t1 = tm[(tm["tarea"] == "T1") & (tm["variante"] == "ajustado")]
    lineas.append(
        "\n**2. ¿La ventaja de la naive se mantiene en años futuros y en pandemia?** (Tarea 1, pronóstico a un paso con la tasa observada de t-1; bootstrap sobre IES, diferencia = MAE de la naive − MAE del modelo ajustado; positiva = gana el modelo)"
    )
    for part, titulo in (
        ("prepandemia", "Prepandemia (entrena 2015–2017, prueba 2018–2019)"),
        ("estres", "Estrés (entrena 2015–2019, prueba 2020–2021)"),
    ):
        base = tm[(tm["particion"] == part) & (tm["configuracion"] == BASE_T1)].iloc[0]
        g = t1[t1["particion"] == part]
        lineas.append(
            f"- **{titulo}:** la naive tiene MAE {_num(base['mae'])} {_ci(base['mae_ic_inf'], base['mae_ic_sup'])}."
        )
        for _, f in g.iterrows():
            lineas.append(
                f"  - `{corto(f['configuracion'])}`: MAE {_num(f['mae'])}; diferencia {_num(f['dif_ref'], 4)} {_ci(f['dif_ref_ic_inf'], f['dif_ref_ic_sup'])}: {_veredicto_ci(f['dif_ref_ic_inf'], f['dif_ref_ic_sup'])}."
            )
    firmes = t1[(t1["dif_ref_ic_inf"] > 0) | (t1["dif_ref_ic_sup"] < 0)]
    lineas.append(
        f"- **Resumen:** en {len(t1) - len(firmes)} de {len(t1)} comparaciones (modelo ajustado frente a la naive, por partición) el IC 95 % de la diferencia incluye 0: "
        + (
            "ningún modelo ajustado se distingue de la naive ni antes ni durante la pandemia."
            if len(firmes) == 0
            else f"se distinguen {len(firmes)}: "
            + ", ".join(
                f"`{corto(f['configuracion'])}` ({f['particion']})" for _, f in firmes.iterrows()
            )
            + "."
        )
        + f" La naive misma pasa de MAE {_num(tm[(tm['particion'] == 'prepandemia') & (tm['configuracion'] == BASE_T1)]['mae'].iloc[0])} a {_num(tm[(tm['particion'] == 'estres') & (tm['configuracion'] == BASE_T1)]['mae'].iloc[0])} en pandemia, es decir, se mantiene competitiva en años futuros y en pandemia."
    )
    # 3. degradación en 2020–2021
    lineas.append(
        "\n**3. ¿En 2020–2021 alguna línea base o modelo se degrada más?** (razón entre el MAE del estrés y el de la prepandemia, con el entrenamiento ajustado en cada partición; los entrenamientos son distintos)"
    )
    for tarea in ("T1", "T2"):
        g = tm[(tm["tarea"] == tarea) & (tm["variante"].isin(["ajustado", "linea_base"]))]
        a, b = (
            g[g["particion"] == "prepandemia"].set_index("configuracion"),
            g[g["particion"] == "estres"].set_index("configuracion"),
        )
        razon = (b["mae"] / a["mae"]).sort_values(ascending=False)
        es_base = {n: ("base_" in n) for n in razon.index}
        peor = razon.index[0]
        peor_modelo = next((n for n in razon.index if not es_base[n]), None)
        peor_base = next((n for n in razon.index if es_base[n]), None)
        lineas.append(
            f"- **Tarea {tarea[1]}:** la mayor degradación es la de `{corto(peor)}` (×{_num(razon[peor], 2)}; MAE {_num(a.loc[peor, 'mae'])} → {_num(b.loc[peor, 'mae'])}). "
            f"Entre las líneas base, la que más se degrada es `{corto(peor_base)}` (×{_num(razon[peor_base], 2)}); entre los modelos, `{corto(peor_modelo)}` (×{_num(razon[peor_modelo], 2)}). "
            f"Los modelos se degradan entre ×{_num(razon[[n for n in razon.index if not es_base[n]]].min(), 2)} y ×{_num(razon[[n for n in razon.index if not es_base[n]]].max(), 2)}; las líneas base, entre ×{_num(razon[[n for n in razon.index if es_base[n]]].min(), 2)} y ×{_num(razon[[n for n in razon.index if es_base[n]]].max(), 2)}."
        )
    for tarea in ("T1", "T2"):
        g = tm[(tm["tarea"] == tarea) & (tm["variante"].isin(["ajustado", "linea_base"]))]
        a_, b_ = (
            g[g["particion"] == "prepandemia"].set_index("configuracion"),
            g[g["particion"] == "estres"].set_index("configuracion"),
        )
        for n in a_.index:
            if "base_" not in n and b_.loc[n, "mae"] / a_.loc[n, "mae"] < 0.8:
                d_ = tm[
                    (tm["particion"] == "prepandemia")
                    & (tm["configuracion"] == n)
                    & (tm["variante"] == "defecto")
                ]["mae"].iloc[0]
                lineas.append(
                    f"- **Atención, `{corto(n)}`:** su razón ×{_num(b_.loc[n, 'mae'] / a_.loc[n, 'mae'], 2)} no es una mejora en pandemia: su MAE de la prepandemia ({_num(a_.loc[n, 'mae'])}) es muy superior al de su versión por defecto ({_num(d_)}), es decir, el ajuste "
                    f"degradó ese modelo al predecir 2018–2019 (ver la tabla de error por subgrupo, con Bogotá aparte, y la nota de diagnóstico)."
                )
    anio = gr[
        (gr["variable"] == "anio")
        & (gr["variante"].isin(["ajustado", "linea_base"]))
        & (gr["tarea"] == "T1")
    ]
    base_anio = anio[anio["configuracion"] == BASE_T1].set_index("nivel")["mae"]
    lineas.append(
        "- MAE de la naive (Tarea 1) por año de prueba: "
        + ", ".join(f"{a}: {_num(v)}" for a, v in base_anio.items())
        + "."
    )
    # 4. municipio
    lineas.append(
        "\n**4. ¿Agrupar por municipio cambia la Tarea 2?** (confirmación 5 × 5 con GroupKFold por municipio sede; ninguna ciudad de prueba aparece en el entrenamiento)"
    )
    mod = frio[frio["tipo"] == "modelo"]
    sup_f = mod[
        (mod["mejora_grupo_pct"] > 0) & (mod["p_wilcoxon_grupo"] < 0.05) & (mod["p_t_grupo"] < 0.05)
    ]
    mejor = mod.sort_values("mae_media").iloc[0]
    i = ies.set_index("configuracion")
    bi, bf = i.loc["base_mediana_grupo"], frio.set_index("configuracion").loc["base_mediana_grupo"]
    lineas.append(
        f"- La mediana por carácter × sector pasa de MAE {_num(bi['mae_media'])} (por IES) a {_num(bf['mae_media'])} (por municipio). "
        f"{len(sup_f)} de {len(mod)} configuraciones superan a esa línea base con significancia en las dos pruebas. La de menor MAE por municipio es `{mejor['configuracion']}`: {_ic(mejor)} ({_pct(mejor['mejora_grupo_pct'])}; "
        f"p Wilcoxon {_p(mejor['p_wilcoxon_grupo'])}, p t corregida {_p(mejor['p_t_grupo'])}); por IES esa misma configuración tenía {_pct(i.loc[mejor['configuracion'], 'mejora_grupo_pct'])}."
    )
    cambios4 = "; ".join(
        f"`{n}`: {_pct(i.loc[n, 'mejora_grupo_pct'])} por IES → {_pct(f_['mejora_grupo_pct'])} por municipio"
        for n, f_ in mod.set_index("configuracion").iterrows()
        if n in i.index
    )
    lineas.append(
        "- **Respuesta:** agrupar por municipio "
        + (
            "no cambia la conclusión principal de la Tarea 2: ningún modelo supera con firmeza a la mediana por carácter × sector."
            if len(sup_f) == 0
            else f"cambia la conclusión: {len(sup_f)} configuraciones superan con firmeza a la línea base."
        )
        + " Mejora frente a la mediana por carácter × sector (la línea base se recalcula con cada agrupación): "
        + cambios4
        + "."
    )
    b = tam[tam["contiene_bogota"]]
    lineas.append(
        f"- Bogotá desbalancea los pliegues: el pliegue que la contiene tiene entre {int(b['n_prueba'].min())} y {int(b['n_prueba'].max())} filas de prueba ({b['pct_prueba'].min():.0f} % a {b['pct_prueba'].max():.0f} % del total), "
        f"frente a {int(tam[~tam['contiene_bogota']]['n_prueba'].min())}–{int(tam[~tam['contiene_bogota']]['n_prueba'].max())} en los demás; cuando Bogotá está en la prueba, el entrenamiento no tiene ninguna fila de Bogotá."
    )
    return "\n".join(lineas)


def _nota_diagnostico(pred: pd.DataFrame, umbral: float = 0.005, peores: int = 40) -> str:
    """Si el ajuste empeoró a un modelo en alguna partición (MAE ajustado − MAE por defecto > `umbral`), describe dónde se concentra el error."""
    if "cod_divipola_mpio" not in pred:
        return (
            "No hay municipio en las predicciones: no se pudo describir la concentración del error."
        )
    p = pred.assign(
        err=(pred["pred"] - pred["y"]).abs(),
        bogota=pred["cod_divipola_mpio"].astype(str) == "11001",
    )
    hallazgos = []
    for (part, nombre), g in p[p["variante"].isin(["defecto", "ajustado"])].groupby(
        ["particion", "configuracion"]
    ):
        d, a = g[g["variante"] == "defecto"], g[g["variante"] == "ajustado"]
        if a["err"].mean() - d["err"].mean() <= umbral:
            continue
        top = a.sort_values("err", ascending=False).head(peores)
        por_anio = ", ".join(
            f"{int(an)}: {_num(v)}" for an, v in a.groupby("anio")["err"].mean().items()
        )
        por_anio_d = ", ".join(
            f"{int(an)}: {_num(v)}" for an, v in d.groupby("anio")["err"].mean().items()
        )
        hallazgos.append(
            f"- **`{corto(nombre)}`, partición {part}:** el ajuste empeora el MAE de {_num(d['err'].mean())} a {_num(a['err'].mean())} (por año de prueba, ajustado: {por_anio}; por defecto: {por_anio_d}). "
            f"De sus {peores} filas con mayor error, {int(top['bogota'].sum())} son de Bogotá (Bogotá es el {100 * a['bogota'].mean():.0f} % de las filas de prueba); hiperparámetros elegidos: `{a['hiperparametros'].iloc[0]}`. "
            f"MAE en Bogotá {_num(a.loc[a['bogota'], 'err'].mean())} frente a {_num(a.loc[~a['bogota'], 'err'].mean())} fuera de Bogotá."
        )
    if not hallazgos:
        return f"Ningún modelo empeoró en más de {_num(umbral, 3)} de MAE al ajustarlo en las particiones temporales."
    return (
        "Modelos a los que el ajuste empeoró en más de "
        + _num(umbral, 3)
        + " de MAE en una partición temporal (la búsqueda interna agrupa por IES dentro de los años de entrenamiento, así que no puede detectar un desplazamiento temporal de las variables):\n\n"
        + "\n".join(hallazgos)
    )


def _tabla_descriptiva(d: pd.DataFrame) -> str:
    lineas = [
        "| Tarea | Configuración | Variante | n | MAE | Error abs. mediano | R² | Error ≤ 0,02 | Error ≤ 0,05 | Spearman | Precisión 25 % mayor riesgo |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for f in d.itertuples():
        lineas.append(
            f"| {f.tarea[1]} | `{_c(corto(f.configuracion))}` | {f.variante} | {f.n} | {_num(f.mae)} | {_num(f.error_absoluto_mediano)} | "
            f"{_num(f.r2, 2)} | {f.pct_error_hasta_2pp:.0f} % | {f.pct_error_hasta_5pp:.0f} % | {_num(f.spearman, 2)} | "
            f"{100 * f.precision_25_mayor_riesgo:.0f} % |"
        )
    return "\n".join(lineas)


# ---------- figuras ----------
def figura_ajuste(r: pd.DataFrame, ruta: str | Path) -> None:
    fig, ejes = plt.subplots(1, 2, figsize=(14, 5.8))
    for ax, tarea in zip(ejes, ("T1", "T2"), strict=False):
        g = r[(r["tarea"] == tarea) & (r["variante"].isin(["defecto", "ajustado"]))]
        nombres = list(dict.fromkeys(g["configuracion"]))[::-1]
        for i, n in enumerate(nombres):
            for variante, dy in (("defecto", 0.14), ("ajustado", -0.14)):
                f = g[(g["configuracion"] == n) & (g["variante"] == variante)].iloc[0]
                ax.errorbar(
                    f["mae_media"],
                    i + dy,
                    xerr=[[f["mae_media"] - f["mae_ic_inf"]], [f["mae_ic_sup"] - f["mae_media"]]],
                    fmt="o",
                    color=COLORES[variante],
                    ecolor="#444",
                    capsize=3,
                    markersize=7,
                    label=variante if i == 0 else None,
                )
        for base, color, estilo in (
            (REF[tarea], "black", "-"),
            (f"{tarea} base_mediana", "#555", "--"),
        ):
            b = r[(r["configuracion"] == base)].iloc[0]
            ax.axvline(
                b["mae_media"],
                color=color,
                ls=estilo,
                lw=1.3,
                label=f"{corto(base)} ({_num(b['mae_media'])})",
            )
        ax.set_yticks(range(len(nombres)), [corto(n) for n in nombres], fontsize=8)
        ax.set_xlabel("MAE (media e IC 95 %; 25 ajustes externos)")
        ax.set_title(
            "Tarea 1: con historia" if tarea == "T1" else "Tarea 2: sin historia", fontsize=10
        )
        ax.legend(
            fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False
        )
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Hiperparámetros por defecto frente a ajustados (validación anidada, mismos pliegues)",
        fontsize=10,
    )
    fig.tight_layout()
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def figura_temporal(gr: pd.DataFrame, ruta: str | Path) -> None:
    fig, ejes = plt.subplots(2, 2, figsize=(13, 8), sharey="row")
    g = gr[(gr["variable"] == "anio") & (gr["variante"].isin(["ajustado", "linea_base"]))]
    paleta = plt.cm.tab10.colors
    for fila, tarea in enumerate(("T1", "T2")):
        for col, (part, titulo) in enumerate(
            (
                ("prepandemia", "Prepandemia: entrena 2015–2017"),
                ("estres", "Estrés: entrena 2015–2019"),
            )
        ):
            ax = ejes[fila, col]
            x = g[(g["tarea"] == tarea) & (g["particion"] == part)]
            for i, n in enumerate(dict.fromkeys(x["configuracion"])):
                s = x[x["configuracion"] == n].sort_values("nivel")
                base = "base_" in n
                ax.plot(
                    s["nivel"],
                    s["mae"],
                    marker="o",
                    ls="--" if base else "-",
                    lw=2.2 if n in (BASE_T1, BASE_T2) else 1.4,
                    color="black" if n in (BASE_T1, BASE_T2) else paleta[i % 10],
                    label=corto(n),
                )
            ax.set_title(f"Tarea {tarea[1]} · {titulo}", fontsize=10)
            ax.set_xlabel("año de prueba")
            ax.spines[["top", "right"]].set_visible(False)
            if col == 0:
                ax.set_ylabel("MAE")
            ax.legend(fontsize=6.5, loc="best", ncol=1)
    fig.suptitle(
        "MAE por año de prueba: modelos ajustados (continuas) y líneas base (discontinuas; la de la tarea en negro continuo)",
        fontsize=10,
    )
    fig.tight_layout()
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


# ---------- informe ----------
def generar_informe(cfg: dict, s: dict[str, Path], tiempos: dict) -> str:
    p = cfg["modelado"]["ajuste_temporal"]
    r = pd.read_csv(s["ajuste_resumen"])
    h = pd.read_csv(s["hiperparametros"])
    tm = pd.read_csv(s["temporal"])
    gr = pd.read_csv(s["temporal_grupos"])
    frio = pd.read_csv(s["arranque_frio_resumen"])
    tam = pd.read_csv(s["arranque_frio_pliegues"])
    ies = pd.read_csv(p["confirmacion_tarea2"])
    pred = pd.read_csv(s["temporal_predicciones"], dtype={"cod_ies": str, "cod_divipola_mpio": str})
    descriptivo = descriptivo_temporal(pred)
    descriptivo.to_csv(Path(s["temporal"]).with_name("temporal_descriptivo.csv"), index=False)
    figura_ajuste(r, s["figura_ajuste"])
    figura_temporal(gr, s["figura_temporal"])
    b = p["bootstrap"]
    secciones_temporales = []
    for part, titulo in (
        ("prepandemia", "Prepandemia (entrena 2015–2017, prueba 2018–2019)"),
        ("estres", "Estrés (entrena 2015–2019, prueba 2020–2021)"),
    ):
        for tarea, nombre in (
            ("T1", "Tarea 1 (con historia, pronóstico a un paso)"),
            ("T2", "Tarea 2 (sin historia)"),
        ):
            secciones_temporales.append(
                f"#### {titulo} · {nombre}\n\n{_tabla_temporal(tm, part, tarea)}"
            )
    grupos = []
    for part, titulo in (("prepandemia", "Prepandemia"), ("estres", "Estrés")):
        for tarea, nombre in (("T1", "Tarea 1"), ("T2", "Tarea 2")):
            grupos.append(f"#### {titulo} · {nombre}\n\n{_tabla_grupos(gr, part, tarea)}")
    texto = f"""# Ajuste de hiperparámetros, particiones temporales y arranque en frío estricto

Generado por `desercion evaluar ajuste` (módulos `src/desercion_ies/evaluation/ajuste.py`, `temporal.py` y `arranque_frio.py`). Todas las cifras salen de la corrida; no se editan a mano. Continúa `reports/02_tarea1.md` (Tarea 1) y `reports/03_tarea2.md` (Tarea 2).

**Objetivo.** Descartar dos objeciones antes de cerrar la conclusión de que ningún modelo supera con firmeza a su línea base: (1) que los modelos rindan poco por usar hiperparámetros por defecto, y (2) que la validación por IES no represente el uso real (predecir años futuros, o IES en ciudades no vistas).

## Diseño

- **Ajuste con validación anidada (sin fuga).** Externa: GroupKFold por `cod_ies`, 5 pliegues repetidos con las semillas {p["semillas_externas"]} (distintas a las del cribado, la confirmación y la v3), 25 ajustes por configuración. Interna: dentro de cada pliegue externo, `RandomizedSearchCV` de {p["n_iter"]} iteraciones, scoring MAE, GroupKFold por IES de {p["n_splits_interno"]} pliegues, **solo con el entrenamiento externo** (una prueba lo verifica). El mejor modelo se reajusta con todo el entrenamiento externo y se mide en la prueba externa; la versión por defecto (hiperparámetros de la v1) se mide en los mismos pliegues. Predicciones recortadas a [0, 1].
- **Espacios.** ridge: `alpha` log-uniforme 1e-3–1e3 (se usa `Ridge`; la versión por defecto usa `RidgeCV`). random forest: `n_estimators` {{200, 500}}, `max_depth` {{3, 5, 8, None}}, `min_samples_leaf` {{1, 5, 10, 20}}, `max_features` {{0,3, 0,5, sqrt}}. xgboost: `n_estimators` {{200, 500}}, `max_depth` {{2, 3, 4, 6}}, `learning_rate` log-uniforme 0,01–0,3, `subsample` {{0,7, 1,0}}, `colsample_bytree` {{0,5, 0,8, 1,0}}, `min_child_weight` {{1, 5, 10}}, `reg_lambda` log-uniforme 0,1–10. Si el preprocesamiento incluye información mutua, se ajusta también `k` ({p["k_info_mutua"]}).
- **Configuraciones** (fijadas antes de correr). Tarea 1: C solo historia (ridge, random forest, xgboost) y B historia + contexto (xgboost, logit). Tarea 2: la mejor configuración de cada modelo de la confirmación de la Tarea 2. Líneas base: ingenua (tasa de t-1) en la Tarea 1; mediana por carácter × sector en la Tarea 2 (más la mediana global).
- **Inferencia del ajuste:** IC 95 % y «p t corregida» con la varianza corregida de Nadeau y Bengio; el Wilcoxon sobre los 25 pares no corrige la dependencia (más optimista). `n_jobs=-1` se aplica sobre (configuración, pliegue externo) y cada búsqueda interna corre en un hilo, para no sobresuscribir los núcleos.
- **Particiones temporales.** Prepandemia: entrena 2015–2017, prueba 2018–2019. Estrés: entrena 2015–2019, prueba 2020–2021. Para cada partición se re-ajusta con la misma búsqueda interna **solo con los años de entrenamiento** (una prueba verifica que ningún año de prueba entra al entrenamiento ni a la búsqueda). **Tarea 1 es un pronóstico a un paso:** para predecir el año t se usa la tasa observada de t-1 aunque t-1 esté en el período de prueba (en la práctica está disponible cuando se predice t); lo único que se oculta al entrenar son las filas y los objetivos de los años de prueba. IC 95 % por bootstrap sobre IES ({b["replicas"]} réplicas, semilla {b["semilla"]}; se remuestrean IES completas y el mismo remuestreo se usa para todas las configuraciones, así que las diferencias con la línea base son pareadas).
- **Arranque en frío estricto (solo Tarea 2).** Se repite la confirmación (5 × 5, las mismas configuraciones y semillas) con GroupKFold por municipio sede (`cod_divipola_mpio`), de modo que ninguna ciudad de prueba aparece en el entrenamiento.
- **Tiempo de la corrida:** ajuste anidado {tiempos["t_ajuste_s"]:.0f} s, particiones temporales {tiempos["t_temporal_s"]:.0f} s, arranque en frío {tiempos["t_frio_s"]:.0f} s; total {tiempos["t_total_s"]:.0f} s ({_num(tiempos["t_total_s"] / 60, 1)} min).

## Conclusión

{_conclusion(r, tm, gr, frio, ies, tam)}

## 1. Por defecto frente a ajustado (validación anidada)

«Mejora por el ajuste» = MAE por defecto − MAE ajustado (positiva = el ajuste ayuda). La línea base es la de la tarea.

### Tarea 1: con historia (línea base: ingenua)

{_tabla_ajuste(r, "T1")}

### Tarea 2: sin historia (línea base: mediana por carácter × sector)

{_tabla_ajuste(r, "T2")}

![Por defecto frente a ajustado](figures/ajuste_vs_defecto.png)

### Hiperparámetros más elegidos en los 25 pliegues externos

Valor más frecuente (y en cuántos pliegues); para los continuos, la mediana y el rango.

{_tabla_hiperparametros(h)}

### Tiempos de ajuste y de predicción (para el análisis de complejidad)

Medias por pliegue externo, con las configuraciones corriendo en paralelo (no son comparables con corridas sin paralelismo). «Ajuste» de la variante ajustada es el reajuste del mejor modelo; la búsqueda interna completa está en la última columna.

{_tabla_tiempos(r)}

## 2. Particiones temporales

{chr(10).join(secciones_temporales)}

![MAE por año de prueba](figures/temporal_por_anio.png)

### Resumen descriptivo con las dos particiones juntas

**Descriptivo: sin intervalos ni pruebas de significancia.** Junta las filas de prueba de prepandemia y de estrés (cada partición con su propio reajuste), por configuración; no sirve para afirmar que un modelo supera a otro. «Precisión 25 % mayor riesgo»: de la cuarta parte con mayor predicción, cuánto está en la cuarta parte con mayor tasa observada (empates repartidos en proporción). Tabla completa: `reports/tables/temporal_descriptivo.csv`.

{_tabla_descriptiva(descriptivo)}

### Degradación en 2020–2021 (modelos ajustados y líneas base)

#### Tarea 1

{_tabla_degradacion(tm, "T1")}

#### Tarea 2

{_tabla_degradacion(tm, "T2")}

### Nota de diagnóstico: deterioro por el ajuste en una partición temporal

{_nota_diagnostico(pred)}

### Error por sector, carácter, año de prueba y Bogotá frente al resto (MAE; modelos ajustados y líneas base)

{chr(10).join(grupos)}

## 3. Arranque en frío estricto: pliegues por municipio (Tarea 2)

### Tamaño de los pliegues de prueba

{_tabla_pliegues(tam)}

### Resultados (confirmación 5 × 5 por municipio, y la confirmación por IES de la Tarea 2 como referencia)

{_tabla_frio(frio, ies)}

## Supuestos y limitaciones

- **Una sola partición temporal por periodo:** el bootstrap sobre IES captura la variabilidad por IES, no la de elegir otro corte de años. Los entrenamientos de las dos particiones difieren (3 y 5 años), así que la comparación entre ellas mezcla el efecto del periodo con el del tamaño del entrenamiento.
- **Tarea 1 a un paso:** usa la tasa observada de t-1 incluso dentro del periodo de prueba; no es un pronóstico a varios pasos. En la práctica supone que la tasa de t-1 está disponible al predecir t (supuesto sin verificar de `reports/01_etl.md`).
- **Tarea 2 en el tiempo:** las IES de prueba también aparecen (en años anteriores) en el entrenamiento; la partición temporal no mide el arranque en frío, solo la deriva en el tiempo del contexto. El arranque en frío se mide con la agrupación por IES y por municipio.
- **Búsqueda interna agrupada por IES** (3 pliegues): en las particiones temporales la validación interna no es temporal, solo evita que una misma IES esté en entrenamiento y validación.
- **Pocos municipios:** con GroupKFold por municipio, Bogotá concentra una parte grande de las filas y se evalúa sin ninguna fila de Bogotá en el entrenamiento; los pliegues quedan desbalanceados (ver la tabla de tamaños).
- El sesgo de selección, supervivencia y geografía de `reports/05_sesgo_dataset.md` aplica igual a estos resultados.
"""
    s["informe"].parent.mkdir(parents=True, exist_ok=True)
    s["informe"].write_text(texto, encoding="utf-8")
    return texto
