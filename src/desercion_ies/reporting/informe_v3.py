"""Informe del protocolo v3: `reports/02_tarea1.md` y la figura `mae_v3_ic.png`.

Todo el texto con cifras se genera a partir de `resultados` y `resumen` (ver `protocolo.py`); no se escribe ninguna cifra a mano.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from desercion_ies.evaluation.protocolo import (
    BASE_INGENUA,
    BASE_INGENUA_REZAGADA,
    BASE_MEDIANA,
    comparar_pareado,
    factor_nadeau_bengio,
    por_pliegue,
)
from desercion_ies.reporting.informe_experimentos import _num, _pct

SUBS = {
    "": "todas las filas",
    "_con_historia": "filas con historia",
    "_sin_historia": "filas sin historia",
}


def _c(nombre: str) -> str:
    """Nombre de configuración apto para una celda de tabla markdown (escapa las barras)."""
    return nombre.replace("|", "\\|")


def _p(x: float) -> str:
    if x is None or np.isnan(x):
        return "—"
    return "< 0,001" if x < 0.001 else f"{x:.3f}".replace(".", ",")


def _ic(fila, suf: str = "", d: int = 4) -> str:
    return f"{_num(fila[f'mae{suf}_media'], d)} [{_num(fila[f'mae{suf}_ic_inf'], d)}; {_num(fila[f'mae{suf}_ic_sup'], d)}]"


def _ref(fila) -> str:
    """Referencia que corresponde a la configuración: la ingenua rezagada si no usa t-1, la ingenua en los demás casos."""
    return "rez" if fila["escenario"] == "rezagada" else "ing"


def _veredicto(dif: float, p_w: float, p_t: float) -> str:
    sw = (not np.isnan(p_w)) and p_w < 0.05
    st = (not np.isnan(p_t)) and p_t < 0.05
    verbo = "mejora" if dif > 0 else ("empeora" if (sw or st) else "no mejora")
    if sw and st:
        return f"{verbo}, significativa en ambas pruebas"
    if sw:
        return f"{verbo}; significativa solo en Wilcoxon, que no corrige la dependencia"
    if st:
        return f"{verbo}; significativa solo en la t corregida"
    return f"{verbo}, sin significancia"


def _tabla_principal(
    res: pd.DataFrame, filas: list, tag: str, ref_nombre: str, extra: bool = False
) -> str:
    cab = f"| Configuración | MAE media [IC 95 %] | Diferencia vs. {ref_nombre} [IC 95 %] | Mejora | p Wilcoxon | p t corregida | Gana en |"
    sep = "|---|---|---|---|---|---|---|"
    if extra:
        cab += " Mejora vs. ingenua t-1 | p Wilcoxon vs. ingenua t-1 |"
        sep += "---|---|"
    out = [cab, sep]
    for _, f in filas:
        negrita = "**" if f["tipo"] == "linea_base" else ""
        propia = f["configuracion"] == ref_nombre
        celdas = [f"{negrita}{_c(f['configuracion'])}{negrita}", _ic(f)]
        if propia:
            celdas += ["—", "—", "—", "—", "—"]
        else:
            celdas += [
                f"{_num(f[f'dif_{tag}'], 4)} [{_num(f[f'dif_{tag}_ic_inf'], 4)}; {_num(f[f'dif_{tag}_ic_sup'], 4)}]",
                _pct(f[f"mejora_{tag}_pct"]),
                _p(f[f"p_wilcoxon_{tag}"]),
                _p(f[f"p_t_{tag}"]),
                f"{100 * f[f'gana_{tag}']:.0f} %",
            ]
        if extra:
            celdas += [_pct(f["mejora_ing_pct"]), _p(f["p_wilcoxon_ing"])]
        out.append("| " + " | ".join(celdas) + " |")
    return "\n".join(out)


def _tabla_subconjuntos(res: pd.DataFrame) -> str:
    out = [
        "| Configuración | MAE todas las filas | MAE con historia | Mejora (con historia) | MAE sin historia | Mejora (sin historia) |",
        "|---|---|---|---|---|---|",
    ]
    for _, f in res.iterrows():
        t = _ref(f)
        ref = BASE_INGENUA_REZAGADA if t == "rez" else BASE_INGENUA  # noqa: F841  (se conserva: documenta el valor)
        if f["tipo"] == "linea_base":
            mc = ms = "—"
        else:
            mc = f"{_pct(f[f'mejora_{t}_con_historia_pct'])} (p {_p(f[f'p_wilcoxon_{t}_con_historia'])})"
            ms = f"{_pct(f[f'mejora_{t}_sin_historia_pct'])} (p {_p(f[f'p_wilcoxon_{t}_sin_historia'])})"
        out.append(
            f"| {_c(f['configuracion'])} | {_num(f['mae_media'])} | {_ic(f, '_con_historia')} | {mc} | {_ic(f, '_sin_historia')} | {ms} |"
        )
    return "\n".join(out)


PARES_CONTEXTO = [  # (con contexto, sin contexto)
    ("B historia+contexto | ridge logit", "B solo historia | ridge logit"),
    ("C corrección con contexto | ridge", "C corrección solo historia | ridge"),
    ("C corrección con contexto | random_forest", "C corrección solo historia | random_forest"),
    ("B historia+contexto | ridge logit [rezagada]", "B solo historia | ridge logit [rezagada]"),
    (
        "C corrección con contexto | ridge [rezagada]",
        "C corrección solo historia | ridge [rezagada]",
    ),
    (
        "C corrección con contexto | random_forest [rezagada]",
        "C corrección solo historia | random_forest [rezagada]",
    ),
]
PARES_CORRECCION = [  # (corrección, directa)
    ("C corrección con contexto | ridge", "B historia+contexto | ridge logit"),
    ("C corrección solo historia | ridge", "B solo historia | ridge logit"),
    (
        "C corrección con contexto | ridge [rezagada]",
        "B historia+contexto | ridge logit [rezagada]",
    ),
    ("C corrección solo historia | ridge [rezagada]", "B solo historia | ridge logit [rezagada]"),
    (
        "C corrección con contexto | random_forest",
        "B historia+contexto | xgboost logit",
    ),  # el mejor aprendiz de cada formulación en la v2
]


def _existentes(pares: list[tuple[str, str]], resultados: pd.DataFrame) -> list[tuple[str, str]]:
    """Los pares cuyas dos configuraciones están en los resultados (permite correr conjuntos reducidos)."""
    hay = set(resultados["configuracion"])
    return [(a, b) for a, b in pares if a in hay and b in hay]


def _pareo(resultados: pd.DataFrame, a: str, b: str, columna: str, rho: float) -> dict:
    """a frente a b: diferencia = error de b − error de a (positiva = a es mejor)."""
    m = por_pliegue(resultados, columna)
    return comparar_pareado(m[b] - m[a], rho)


def _tabla_pares(
    resultados: pd.DataFrame, pares: list[tuple[str, str]], rho: float, titulo_a: str, titulo_b: str
) -> str:
    out = [
        f"| {titulo_a} frente a {titulo_b} | Todas: diferencia [IC 95 %] | p Wilcoxon | p t corregida | Gana en | Con historia: diferencia (p Wilcoxon) | Sin historia: diferencia (p Wilcoxon) |",
        "|---|---|---|---|---|---|---|",
    ]
    for a, b in _existentes(pares, resultados):
        t, c, s = (
            _pareo(resultados, a, b, col, rho)
            for col in ("mae", "mae_con_historia", "mae_sin_historia")
        )
        out.append(
            f"| {_c(a)} frente a {_c(b)} | {_num(t['dif_media'])} [{_num(t['dif_ic_inf'])}; {_num(t['dif_ic_sup'])}] | {_p(t['p_wilcoxon'])} | {_p(t['p_t_corregido'])} | "
            f"{100 * t['prop_gana']:.0f} % | {_num(c['dif_media'])} ({_p(c['p_wilcoxon'])}) | {_num(s['dif_media'])} ({_p(s['p_wilcoxon'])}) |"
        )
    return "\n".join(out)


def _conclusion(res: pd.DataFrame, resultados: pd.DataFrame, rho: float) -> str:
    r = res.set_index("configuracion")
    ing, rez, med = r.loc[BASE_INGENUA], r.loc[BASE_INGENUA_REZAGADA], r.loc[BASE_MEDIANA]
    modelos = res[res["tipo"] == "modelo"]
    lineas = []
    # 1. ¿alguna supera a la ingenua? (cada escenario contra su referencia)
    for esc, ref_nombre, tag, etiqueta in (
        ("t-1", BASE_INGENUA, "ing", "con la tasa de t-1 disponible"),
        ("rezagada", BASE_INGENUA_REZAGADA, "rez", "en el escenario conservador (sin t-1)"),
    ):
        g = modelos[modelos["escenario"] == esc]
        supera = g[
            (g[f"mejora_{tag}_pct"] > 0)
            & (g[f"p_wilcoxon_{tag}"] < 0.05)
            & (g[f"p_t_{tag}"] < 0.05)
        ]
        mejora_ns = g[(g[f"mejora_{tag}_pct"] > 0)].drop(supera.index)
        peor = g[g[f"mejora_{tag}_pct"] <= 0]
        mejor = g.sort_values("mae_media").iloc[0]
        p_t = mejor[f"p_t_{tag}"]
        lineas.append(
            f"- **¿Alguna configuración supera a la ingenua {etiqueta}?** Entre las {len(g)} configuraciones fijadas: {len(supera)} con menor error y significancia en las dos pruebas "
            f"(p de Wilcoxon y p de la t corregida < 0,05), {len(mejora_ns)} con menor error medio sin esa significancia y {len(peor)} sin mejora. "
            f"La de menor MAE es `{mejor['configuracion']}`: {_ic(mejor)} frente a {_ic(r.loc[ref_nombre])} de la referencia "
            f"({_pct(mejor[f'mejora_{tag}_pct'])}; p Wilcoxon {_p(mejor[f'p_wilcoxon_{tag}'])}, p t corregida {_p(p_t)}, gana en {100 * mejor[f'gana_{tag}']:.0f} % de los pliegues). "
            f"Sin ajustar por comparaciones múltiples; con Bonferroni ({len(g)} comparaciones) la p t corregida de esa configuración sería {_p(min(1.0, p_t * len(g)))}."
            + (
                " Las que superan a la ingenua en las dos pruebas: "
                + ", ".join(f"`{n}`" for n in supera["configuracion"])
                + "."
                if len(supera)
                else ""
            )
        )
    # 2. mediana e ingenuas
    lineas.append(
        f"- **Líneas base:** ingenua {_ic(ing)}, ingenua rezagada {_ic(rez)}, mediana {_ic(med)}. No tener la tasa de t-1 cuesta "
        f"{_num(rez['mae_media'] - ing['mae_media'])} de MAE en la ingenua ({_pct(100 * (ing['mae_media'] - rez['mae_media']) / ing['mae_media'])} de mejora de la rezagada frente a la normal; "
        f"p Wilcoxon {_p(r.loc[BASE_INGENUA_REZAGADA, 'p_wilcoxon_ing'])})."
    )
    # 3. contexto sobre la historia
    lineas.append(
        "- **Aporte del contexto sobre la historia** (diferencia = MAE sin contexto − MAE con contexto; positiva = el contexto ayuda):"
    )
    for a, b in _existentes(PARES_CONTEXTO, resultados):
        t, c, s = (
            _pareo(resultados, a, b, col, rho)
            for col in ("mae", "mae_con_historia", "mae_sin_historia")
        )
        lineas.append(
            f"  - `{a}` frente a `{b}`: {_num(t['dif_media'])} de MAE en todas las filas ({_veredicto(t['dif_media'], t['p_wilcoxon'], t['p_t_corregido'])}; "
            f"p Wilcoxon {_p(t['p_wilcoxon'])}, p t corregida {_p(t['p_t_corregido'])}); en filas con historia {_num(c['dif_media'])} (p {_p(c['p_wilcoxon'])})."
        )
    # 4. corrección frente a directa
    lineas.append(
        "- **Corrección sobre la ingenua frente a predicción directa** (diferencia = MAE de la directa − MAE de la corrección; positiva = la corrección es mejor):"
    )
    for a, b in _existentes(PARES_CORRECCION, resultados):
        t = _pareo(resultados, a, b, "mae", rho)
        lineas.append(
            f"  - `{a}` frente a `{b}`: {_num(t['dif_media'])} ({_veredicto(t['dif_media'], t['p_wilcoxon'], t['p_t_corregido'])}; p Wilcoxon {_p(t['p_wilcoxon'])}, p t corregida {_p(t['p_t_corregido'])})."
        )
    # 5. filas sin historia
    una = resultados[resultados["configuracion"] == BASE_INGENUA]
    n_sin = int(una["n_sin_historia"].sum() / (resultados["repeticion"].nunique()))
    lineas.append(
        f"- **Filas sin historia:** son {n_sin} de {int(una['n_prueba'].sum() / resultados['repeticion'].nunique())} (cada fila se evalúa una vez por repetición), "
        f"en promedio {_num(una['n_sin_historia'].mean(), 1)} por pliegue; el MAE de ese subconjunto tiene mucho más ruido y debe leerse con cautela (tabla por subconjunto)."
    )
    return "\n".join(lineas)


def _lectura_v2(res: pd.DataFrame, resultados: pd.DataFrame, rho: float) -> str:
    r = res.set_index("configuracion")
    una = resultados[resultados["configuracion"] == BASE_INGENUA]
    n_sin = int(una["n_sin_historia"].sum() / resultados["repeticion"].nunique())
    lineas = [
        "- **v2: «la mejor configuración global (B con xgboost y logit) supera a la ingenua por poco (+3,1 %)».**"
    ]
    for nombre in [
        n
        for n in ("B historia+contexto | xgboost logit", "B historia+contexto | ridge logit")
        if n in r.index
    ]:
        f = r.loc[nombre]
        lineas.append(
            f"  - v3, `{nombre}` frente a la ingenua (que ahora usa t-1 también en 2015): {_pct(f['mejora_ing_pct'])} ({_veredicto(f['dif_ing'], f['p_wilcoxon_ing'], f['p_t_ing'])}; "
            f"p Wilcoxon {_p(f['p_wilcoxon_ing'])}, p t corregida {_p(f['p_t_ing'])})."
        )
    b_filas = [
        r.loc[n]
        for n in ("B historia+contexto | xgboost logit", "B historia+contexto | ridge logit")
        if n in r.index
    ]
    if b_filas and all(f["mejora_ing_pct"] <= 0 for f in b_filas):
        conclusion = "la ingenua deja de usar la mediana en 2015 y la ventaja desaparece (ninguna de las dos B mejora a la ingenua)"
    else:
        conclusion = "aun así al menos una B conserva menor error medio que la ingenua"
    lineas.append(
        f"  - La ventaja de la v2 venía de las filas sin historia: con la historia desde 1999 esas filas pasan de 305 a {n_sin}; {conclusion}."
    )
    if _existentes([PARES_CONTEXTO[0]], resultados):
        t, c, sn = (
            _pareo(resultados, *PARES_CONTEXTO[0], col, rho)
            for col in ("mae", "mae_con_historia", "mae_sin_historia")
        )
        lineas.append(
            f"- **v2: «el contexto casi no aporta sobre la historia; sí en las filas sin historia».** v3, B con contexto frente a B solo historia (ridge logit): "
            f"{_num(t['dif_media'])} de MAE ({_veredicto(t['dif_media'], t['p_wilcoxon'], t['p_t_corregido'])}); en filas con historia {_num(c['dif_media'])} (p {_p(c['p_wilcoxon'])}); "
            f"en filas sin historia {_num(sn['dif_media'])} (p {_p(sn['p_wilcoxon'])}, con solo {n_sin} filas)."
        )
    lineas.append("- **v2: «la corrección (C) no mejora a la predicción directa (B)».**")
    for a, b in _existentes([PARES_CORRECCION[0], PARES_CORRECCION[-1]], resultados):
        t = _pareo(resultados, a, b, "mae", rho)
        lineas.append(
            f"  - v3, `{a}` frente a `{b}`: {_num(t['dif_media'])} ({_veredicto(t['dif_media'], t['p_wilcoxon'], t['p_t_corregido'])}; p Wilcoxon {_p(t['p_wilcoxon'])}, p t corregida {_p(t['p_t_corregido'])})."
        )
    return "\n".join(lineas)


def figura_ic(res: pd.DataFrame, ruta: str | Path) -> None:
    """MAE medio con barras de IC 95 % por configuración; las líneas base como referencias verticales (con su IC)."""
    m = res[res["tipo"] == "modelo"].sort_values("mae_media", ascending=False)
    base = res.set_index("configuracion")
    colores = {"t-1": "#4c78a8", "rezagada": "#f58518"}
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(m) + 2.2))
    y = np.arange(len(m))
    ax.errorbar(
        m["mae_media"],
        y,
        xerr=[m["mae_media"] - m["mae_ic_inf"], m["mae_ic_sup"] - m["mae_media"]],
        fmt="none",
        ecolor="#333",
        elinewidth=1.4,
        capsize=3,
        zorder=2,
    )
    ax.scatter(
        m["mae_media"],
        y,
        c=[colores[e] for e in m["escenario"]],
        s=55,
        zorder=3,
        edgecolor="black",
        linewidth=0.6,
    )
    for nombre, color, estilo in (
        (BASE_INGENUA, "black", "-"),
        (BASE_INGENUA_REZAGADA, "#b35806", "-."),
        (BASE_MEDIANA, "#555", "--"),
    ):
        b = base.loc[nombre]
        ax.axvline(
            b["mae_media"],
            color=color,
            ls=estilo,
            lw=1.3,
            label=f"{nombre.removeprefix('base_')} ({_num(b['mae_media'])})",
        )
        if nombre != BASE_MEDIANA:
            ax.axvspan(b["mae_ic_inf"], b["mae_ic_sup"], color=color, alpha=0.07)
    ax.set_yticks(y, m["configuracion"], fontsize=8)
    ax.set_xlabel(
        "MAE en escala de tasa (media e IC 95 %; 5 repeticiones × 5 pliegues, GroupKFold por IES)",
        fontsize=9,
    )
    ax.legend(
        handles=ax.get_legend_handles_labels()[0]
        + [
            plt.Line2D([0], [0], marker="o", ls="", color=c, mec="black", label=f"escenario {e}")
            for e, c in colores.items()
        ],
        loc="upper right",
        fontsize=8,
        frameon=True,
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def escribir_informe(
    resultados: pd.DataFrame, res: pd.DataFrame, meta: dict, ruta: str | Path
) -> str:
    rho = factor_nadeau_bengio(resultados)
    por_nombre = {f["configuracion"]: f for _, f in res.iterrows()}
    t1 = [(n, por_nombre[n]) for n in [BASE_INGENUA, BASE_MEDIANA, BASE_INGENUA_REZAGADA]] + [
        (f["configuracion"], f)
        for _, f in res[(res["tipo"] == "modelo") & (res["escenario"] == "t-1")].iterrows()
    ]
    rz = [(BASE_INGENUA_REZAGADA, por_nombre[BASE_INGENUA_REZAGADA])] + [
        (f["configuracion"], f)
        for _, f in res[(res["tipo"] == "modelo") & (res["escenario"] == "rezagada")].iterrows()
    ]
    una = resultados[resultados["configuracion"] == BASE_INGENUA]
    texto = f"""# Resultados v3: protocolo de evaluación riguroso

Generado por `desercion evaluar tarea1` (motor en `src/desercion_ies/evaluation/protocolo.py`). Todas las cifras salen de la corrida; no se editan a mano. Continúa `reports/historico/resultados_v2.md`.

**Importante:** la v2 se calculó con la historia que empieza en 2015; en esta v3 la historia (`tasa_t1`, `tasa_t2`, `tasa_media_previa`, `n_anios_previos`) usa también las tasas anuales de SPADIES de 1999 a 2014 (misma definición y filtros), así que las cifras de la v2 y las de esta v3 no son comparables una a una.

## Protocolo

- **Datos:** `data/processed/dataset_ies.csv`, {f"{meta['filas']:,}".replace(",", ".")} filas y {meta["ies"]} IES (objetivos {meta["ventana"][0]}–{meta["ventana"][1]}; las filas no cambian respecto a la v2).
- **Validación:** GroupKFold por `cod_ies` con mezcla, {meta["n_splits"]} pliegues, repetido con las semillas {meta["semillas"]}: **{meta["n_ajustes"]} ajustes por configuración**. Cada repetición se genera una sola vez y se usa en todas las configuraciones y líneas base (`hash_prueba` en el CSV).
- **Configuraciones:** las {meta["n_configuraciones"]} del conjunto **fijado en `config.yaml` antes de correr** (no se eligieron mirando resultados): A contexto (ridge logit info_mutua); B historia + contexto (xgboost logit; ridge logit); B solo historia (ridge logit); C corrección con y sin contexto (ridge y random_forest); y las mismas B y C en versión «historia rezagada» (sin usar la tasa de t-1). Imputación mediana, hiperparámetros por defecto de la v1, predicciones recortadas a [0, 1]. B solo historia usa logit para que la comparación con B solo cambie el contexto.
- **Líneas base:** `base_ingenua` (tasa t-1 de la misma IES, o la mediana del entrenamiento si no hay), `base_ingenua_rezagada` (tasa t-2) y `base_mediana`. Cada configuración «rezagada» se compara con la ingenua rezagada, que tiene la misma información; las demás, con la ingenua.
- **Intervalo de confianza y pruebas:** los {meta["n_ajustes"]} MAE por pliegue no son independientes (los entrenamientos se traslapan y las repeticiones reutilizan las mismas IES). El IC 95 % de la media y la «p t corregida» usan la varianza corregida de Nadeau y Bengio, `(1/J + n_prueba/n_entrenamiento) · s²` con `n_prueba/n_entrenamiento` = {_num(rho, 3)} y J = {meta["n_ajustes"]}. La prueba de Wilcoxon de rangos con signo se aplica a las {meta["n_ajustes"]} diferencias pareadas (ingenua − configuración) y **no** corrige la dependencia, por lo que es más optimista; se reporta porque se pidió, y se considera evidencia solo cuando coincide con la t corregida. «Gana en» es la proporción de pliegues con error estrictamente menor que el de la referencia. «Mejora» y «Diferencia» son positivas cuando la configuración tiene menos error.
- **Subconjuntos:** «con historia» son las filas que tienen `tasa_t1`; «sin historia», las que no ({int(una["n_sin_historia"].sum() / resultados["repeticion"].nunique())} filas distintas; ver `reports/etl/sin_historia.csv`). Se usa esa misma definición para todas las configuraciones, incluidas las rezagadas.
- **Tiempo total de la corrida:** {meta["tiempo_total_s"]:.0f} s ({_num(meta["tiempo_total_s"] / 60, 1)} min). scikit-learn {meta["sklearn"]}, xgboost {meta["xgboost"]}.

## Conclusión

{_conclusion(res, resultados, rho)}

### ¿Se sostienen las conclusiones de la v2?

{_lectura_v2(res, resultados, rho)}

## Tabla principal: escenario con la tasa de t-1 disponible (referencia: ingenua)

{_tabla_principal(res, t1, "ing", BASE_INGENUA)}

## Tabla principal: escenario conservador, historia rezagada (referencia: ingenua rezagada)

Las dos últimas columnas comparan además con la ingenua que sí usa t-1 (lo que se pierde por no tenerla).

{_tabla_principal(res, rz, "rez", BASE_INGENUA_REZAGADA, extra=True)}

![MAE con IC 95 % por configuración](figures/mae_v3_ic.png)

## MAE por subconjunto de filas

La mejora de cada subconjunto es frente a la referencia que corresponde (ingenua, o ingenua rezagada para las rezagadas) en ese mismo subconjunto; entre paréntesis, el p de Wilcoxon. Con pocas filas sin historia, el IC de ese subconjunto es ancho.

{_tabla_subconjuntos(res)}

## Aporte del contexto y de la corrección

{_tabla_pares(resultados, PARES_CONTEXTO, rho, "con contexto", "sin contexto")}

{_tabla_pares(resultados, PARES_CORRECCION, rho, "corrección", "directa")}

## Supuestos y limitaciones

- **Disponibilidad de `tasa_t1`:** se asume que la tasa anual de t-1 está disponible al inicio de t; no está verificado (ver `reports/01_etl.md`). El escenario «historia rezagada» es el conservador: si esa tasa no estuviera publicada, solo valen las configuraciones rezagadas y su ingenua rezagada.
- **Historia desde 1999:** la media de las tasas anteriores (`tasa_media_previa`, `tasa_media_rezagada`) usa toda la serie de SPADIES de la IES desde 1999, incluidos años muy anteriores a la ventana y sin los filtros de inconsistencia entre fuentes (SNIES por programa empieza en 2015).
- **Conjunto fijado, pero sobre los mismos datos:** el conjunto de configuraciones no se eligió con estos pliegues, pero sí con la información de las v1 y v2 sobre el mismo dataset.
- **Hiperparámetros por defecto;** `RidgeCV` elige su alfa con validación interna sin agrupar por IES dentro del entrenamiento de cada pliegue (sin fuga hacia la prueba).
- **Dependencia entre pliegues:** la corrección de Nadeau y Bengio es una aproximación; con 5 × 5 pliegues los p de Wilcoxon sobre pares dependientes son optimistas.
- El sesgo de selección, supervivencia y geografía de `reports/05_sesgo_dataset.md` aplica igual a estos resultados.
"""
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    Path(ruta).write_text(texto, encoding="utf-8")
    return texto
