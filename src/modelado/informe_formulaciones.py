"""Informe de las formulaciones: `reports/resultados_v2.md` y la figura `mae_formulaciones.png`.

Todo el texto con cifras se genera a partir de `resultados` y `resumen` (ver `formulaciones.py`); no se escribe ninguna cifra a mano.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.modelado.experimentos import BASE_INGENUA, BASE_MEDIANA  # noqa: E402
from src.modelado.formulaciones import FORMULACIONES  # noqa: E402
from src.modelado.informe_experimentos import _mas_menos, _num, _pct  # noqa: E402

CORTA = {"A_contexto": "A\ncontexto", "B_historia_contexto": "B\nhistoria\n+ contexto", "B_solo_historia": "B solo\nhistoria",
         "C_correccion": "C\ncorrección", "C_correccion_solo_historia": "C solo\nhistoria"}
COLOR = {"A_contexto": "#4c78a8", "B_historia_contexto": "#59a14f", "B_solo_historia": "#8cd17d", "C_correccion": "#e15759",
         "C_correccion_solo_historia": "#ff9d9a"}
CLAVE = ["modelo", "objetivo", "imputacion", "seleccion"]


def figura_formulaciones(resumen: pd.DataFrame, ruta: str | Path) -> None:
    """Tres paneles (todas las filas, con historia, sin historia): MAE de cada configuración por formulación (puntos),
    la mejor con su desviación entre pliegues, y las dos líneas base."""
    base = resumen.set_index("configuracion")
    modelos = resumen[resumen["tipo"] == "modelo"]
    formulaciones = [f for f in FORMULACIONES if f in set(modelos["formulacion"])]
    paneles = [("Todas las filas", "mae"), ("Con historia (hay tasa t-1)", "mae_con_historia"), ("Sin historia (primer año)", "mae_sin_historia")]
    fig, ejes = plt.subplots(1, 3, figsize=(15, 5.2), sharex=False)
    rng = np.random.default_rng(0)
    for ax, (titulo, pref) in zip(ejes, paneles):
        col, desv = f"{pref}_media", f"{pref}_desv"
        for i, f in enumerate(formulaciones):
            g = modelos[modelos["formulacion"] == f]
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(g)), g[col], s=16, color=COLOR[f], alpha=0.55, edgecolor="none")
            mejor = g.sort_values("mae_media").iloc[0]  # la mejor por MAE global, la misma en los tres paneles
            ax.errorbar(i, mejor[col], yerr=mejor[desv], fmt="D", color="black", mfc=COLOR[f], ms=8, capsize=4, zorder=5)
        ax.axhline(base.loc[BASE_INGENUA, col], color="black", lw=1.4, label=f"ingenua ({_num(base.loc[BASE_INGENUA, col])})")
        ax.axhline(base.loc[BASE_MEDIANA, col], color="#555", ls="--", lw=1.4, label=f"mediana ({_num(base.loc[BASE_MEDIANA, col])})")
        ax.set_xticks(range(len(formulaciones)), [CORTA[f] for f in formulaciones], fontsize=8)
        ax.set_title(titulo, fontsize=10)
        ax.set_ylabel("MAE (escala de tasa)" if ax is ejes[0] else "")
        ax.legend(fontsize=8, loc="upper right", frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("MAE por formulación: puntos = cada configuración; rombo = la mejor del MAE global (± desviación entre pliegues)", fontsize=10)
    fig.tight_layout()
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def _etiqueta(r) -> str:
    return f"{r['modelo']} \\| {r['objetivo']} \\| {r['imputacion']} \\| {r['seleccion']}"


def _tabla_base(resumen: pd.DataFrame, resultados: pd.DataFrame) -> str:
    filas = ["| Línea base | MAE (todas) | MAE con historia | MAE sin historia | Filas con historia | Filas sin historia |", "|---|---|---|---|---|---|"]
    for nombre in (BASE_INGENUA, BASE_MEDIANA):
        r = resumen.set_index("configuracion").loc[nombre]
        x = resultados[resultados["configuracion"] == nombre]
        filas.append(f"| **{nombre}** | {_mas_menos(r['mae_media'], r['mae_desv'])} | {_mas_menos(r['mae_con_historia_media'], r['mae_con_historia_desv'])} "
                     f"| {_mas_menos(r['mae_sin_historia_media'], r['mae_sin_historia_desv'])} | {int(x['n_con_historia'].sum())} | {int(x['n_sin_historia'].sum())} |")
    return "\n".join(filas)


def _tabla_formulacion(resumen: pd.DataFrame, nombre: str) -> str:
    g = resumen[resumen["formulacion"] == nombre]
    filas = ["| Configuración (modelo \\| objetivo \\| imputación \\| selección) | MAE | RMSE | R² | Mejora vs. ingenua | MAE con historia | MAE sin historia | Pliegues mejores que la ingenua |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in g.iterrows():
        filas.append(f"| {_etiqueta(r)} | {_mas_menos(r['mae_media'], r['mae_desv'])} | {_mas_menos(r['rmse_media'], r['rmse_desv'])} "
                     f"| {_mas_menos(r['r2_media'], r['r2_desv'], 3)} | {_pct(r['mejora_vs_ingenua_pct'])} "
                     f"| {_num(r['mae_con_historia_media'])} ({_pct(r['mejora_vs_ingenua_con_historia_pct'])}) "
                     f"| {_num(r['mae_sin_historia_media'])} ({_pct(r['mejora_vs_ingenua_sin_historia_pct'])}) | {r['pliegues_mejor_que_ingenua']} de {r['n_pliegues']} |")
    return "\n".join(filas)


def _mejores(resumen: pd.DataFrame) -> str:
    m = resumen[resumen["tipo"] == "modelo"]
    ing = resumen.set_index("configuracion").loc[BASE_INGENUA]
    filas = ["| Formulación | Mejor configuración (por MAE) | MAE | Mejora vs. ingenua | Pliegues mejores que la ingenua | MAE con historia | MAE sin historia | Configuraciones que superan a la ingenua |",
             "|---|---|---|---|---|---|---|---|"]
    for f in FORMULACIONES:
        g = m[m["formulacion"] == f]
        if g.empty:
            continue
        r = g.iloc[0]
        filas.append(f"| {f} | {_etiqueta(r)} | {_mas_menos(r['mae_media'], r['mae_desv'])} | {_pct(r['mejora_vs_ingenua_pct'])} | {r['pliegues_mejor_que_ingenua']} de {r['n_pliegues']} "
                     f"| {_num(r['mae_con_historia_media'])} ({_pct(r['mejora_vs_ingenua_con_historia_pct'])}) | {_num(r['mae_sin_historia_media'])} ({_pct(r['mejora_vs_ingenua_sin_historia_pct'])}) "
                     f"| {int((g['mae_media'] < ing['mae_media']).sum())} de {len(g)} |")
    return "\n".join(filas)


def comparar(resumen: pd.DataFrame, resultados: pd.DataFrame, a: str, b: str, col: str = "mae") -> dict | None:
    """Compara la formulación `a` con `b` (aporte de lo que `a` tiene de más): mejor contra mejor, pares con la misma
    configuración y pliegue a pliegue entre las dos mejores. `diferencia > 0` significa que `a` tiene menos error."""
    m = resumen[resumen["tipo"] == "modelo"]
    ga, gb = m[m["formulacion"] == a], m[m["formulacion"] == b]
    if ga.empty or gb.empty:
        return None
    media = f"{col}_media" if col == "mae" else f"{col}_media"
    ma, mb = ga.sort_values(media).iloc[0], gb.sort_values(media).iloc[0]
    pares = ga.merge(gb, on=CLAVE, suffixes=("_a", "_b"))
    dif_pares = pares[f"{media}_b"] - pares[f"{media}_a"]
    pa = resultados[resultados["configuracion"] == ma["configuracion"]].set_index("pliegue")[col]
    pb = resultados[resultados["configuracion"] == mb["configuracion"]].set_index("pliegue")[col]
    dif_pliegues = pb - pa
    return {
        "mejor_a": ma, "mejor_b": mb, "diferencia_mejores": float(mb[media] - ma[media]), "n_pares": len(pares),
        "pares_a_mejor": int((dif_pares > 0).sum()), "dif_pares_media": float(dif_pares.mean()) if len(pares) else float("nan"),
        "pliegues_a_mejor": int((dif_pliegues > 0).sum()), "n_pliegues": len(dif_pliegues),
        "dif_pliegues_media": float(dif_pliegues.mean()), "dif_pliegues_desv": float(dif_pliegues.std()),
    }


def _frase(c: dict | None, nombre_a: str, nombre_b: str, col: str = "mae") -> str:
    if c is None:
        return f"- {nombre_a} frente a {nombre_b}: no se corrieron ambas formulaciones."
    d = c["diferencia_mejores"]
    razon = abs(c["dif_pliegues_media"]) / c["dif_pliegues_desv"] if c["dif_pliegues_desv"] > 0 else float("inf")
    if c["pliegues_a_mejor"] in (0, c["n_pliegues"]):
        veredicto = f"el mismo signo en los {c['n_pliegues']} pliegues" + ("" if d > 0 else " (a favor de la otra)")
    else:
        veredicto = "no es consistente en todos los pliegues"
    veredicto += f"; la diferencia media equivale a {_num(razon, 2)} desviaciones entre pliegues"
    return (f"- **{nombre_a} frente a {nombre_b}:** mejor contra mejor ({'por MAE de las filas con historia' if col != 'mae' else 'por MAE global'}), {_num(c['mejor_a'][col + '_media'])} frente a {_num(c['mejor_b'][col + '_media'])} "
            f"(diferencia {_num(d, 4)} de MAE a favor de {nombre_a if d > 0 else nombre_b}); entre las dos mejores, {nombre_a} gana en "
            f"{c['pliegues_a_mejor']} de {c['n_pliegues']} pliegues (diferencia media {_num(c['dif_pliegues_media'], 4)} ± {_num(c['dif_pliegues_desv'], 4)}): {veredicto}. "
            f"Con la misma configuración, {nombre_a} tiene menor MAE en {c['pares_a_mejor']} de {c['n_pares']} pares (diferencia media {_num(c['dif_pares_media'], 4)}).")


SUBCONJUNTOS = [("todas las filas", "mae"), ("filas con historia", "mae_con_historia"), ("filas sin historia", "mae_sin_historia")]


def _por_subconjunto(resumen: pd.DataFrame, resultados: pd.DataFrame) -> str:
    """Mejor configuración de las 72 según el MAE de cada subconjunto, y cuántas superan a la ingenua en él."""
    m = resumen[resumen["tipo"] == "modelo"]
    ing = resumen.set_index("configuracion").loc[BASE_INGENUA]
    filas = ["| Subconjunto | Ingenua | Mejor configuración en ese subconjunto | MAE | Mejora vs. ingenua | Pliegues mejores que la ingenua | Configuraciones que superan a la ingenua |",
             "|---|---|---|---|---|---|---|"]
    for etiqueta, pref in SUBCONJUNTOS:
        col = f"{pref}_media"
        r = m.sort_values(col).iloc[0]
        por_pliegue = resultados.pivot(index="pliegue", columns="configuracion", values=pref)
        gana = int((por_pliegue[r["configuracion"]] < por_pliegue[BASE_INGENUA]).sum())
        filas.append(f"| {etiqueta} | {_num(ing[col])} | `{r['configuracion'].replace('|', chr(92) + '|')}` | {_mas_menos(r[col], r[f'{pref}_desv'])} "
                     f"| {_pct(100 * (ing[col] - r[col]) / ing[col])} | {gana} de {len(por_pliegue)} | {int((m[col] < ing[col]).sum())} de {len(m)} |")
    return "\n".join(filas)


def _descomposicion(resultados: pd.DataFrame, mejor: str) -> tuple[float, float]:
    """(aporte de las filas con historia, aporte de las filas sin historia) a la diferencia de MAE total ingenua − `mejor`,
    exacta por pliegue: MAE_total = (n_con · MAE_con + n_sin · MAE_sin) / n."""
    a = resultados[resultados["configuracion"] == mejor].set_index("pliegue")
    b = resultados[resultados["configuracion"] == BASE_INGENUA].set_index("pliegue")
    con = (a["n_con_historia"] / a["n_prueba"] * (b["mae_con_historia"] - a["mae_con_historia"])).mean()
    sin = (a["n_sin_historia"] / a["n_prueba"] * (b["mae_sin_historia"] - a["mae_sin_historia"])).mean()
    return float(con), float(sin)


def _conclusion(resumen: pd.DataFrame, resultados: pd.DataFrame) -> str:
    m = resumen[resumen["tipo"] == "modelo"]
    ing = resumen.set_index("configuracion").loc[BASE_INGENUA]
    mejor = m.iloc[0]
    lineas = []
    supera = m[m["mae_media"] < ing["mae_media"]]
    lineas.append(f"- **¿Alguna formulación supera a la ingenua?** " + (
        f"Sí: {len(supera)} de {len(m)} configuraciones tienen MAE medio menor que el de la ingenua ({_num(ing['mae_media'])}). " if len(supera) else
        f"No: ninguna de las {len(m)} configuraciones tiene MAE medio menor que el de la ingenua ({_num(ing['mae_media'])}). ")
        + f"La mejor es `{mejor['configuracion']}` con {_mas_menos(mejor['mae_media'], mejor['mae_desv'])} ({_pct(mejor['mejora_vs_ingenua_pct'])}; "
        f"mejor que la ingenua en {mejor['pliegues_mejor_que_ingenua']} de {mejor['n_pliegues']} pliegues).")
    pareado = resultados[resultados["configuracion"] == BASE_INGENUA].set_index("pliegue")["mae"] - resultados[
        resultados["configuracion"] == mejor["configuracion"]].set_index("pliegue")["mae"]
    con, sin = _descomposicion(resultados, mejor["configuracion"])
    n = resultados[resultados["configuracion"] == BASE_INGENUA]
    lineas.append(f"  - Matiz: la diferencia de la mejor frente a la ingenua es {_num(ing['mae_media'] - mejor['mae_media'], 4)} de MAE; por pliegue (ingenua − mejor) "
                  f"vale {_num(pareado.mean(), 4)} ± {_num(pareado.std(), 4)}, y la mejor se eligió entre {len(m)} configuraciones con esos mismos pliegues (la ingenua no se eligió), "
                  f"así que la ventaja está inflada por la selección. De esa diferencia, {_num(sin, 4)} viene de las filas sin historia ({int(n['n_sin_historia'].sum())} de "
                  f"{int(n['n_prueba'].sum())}) y {_num(con, 4)} de las filas con historia.")
    for f in FORMULACIONES:
        g = m[m["formulacion"] == f]
        if g.empty:
            continue
        r = g.iloc[0]
        lineas.append(f"  - {f}: mejor `{_etiqueta(r).replace(chr(92), '')}`, MAE {_num(r['mae_media'])} ({_pct(r['mejora_vs_ingenua_pct'])} frente a la ingenua; "
                      f"{int((g['mae_media'] < ing['mae_media']).sum())} de {len(g)} configuraciones la superan).")
    lineas.append("- **Por subconjunto de filas** (mejor de las configuraciones en cada uno):\n\n" + _por_subconjunto(resumen, resultados) + "\n")
    lineas.append(f"- **Filas con y sin historia (para la mejor global):** con historia, la ingenua tiene MAE {_num(ing['mae_con_historia_media'])}; la mejor configuración global llega a "
                  f"{_num(mejor['mae_con_historia_media'])} ({_pct(mejor['mejora_vs_ingenua_con_historia_pct'])}). Sin historia (primer año de la IES), la ingenua no tiene tasa t-1 y usa la "
                  f"mediana (MAE {_num(ing['mae_sin_historia_media'])}); la mejor configuración global llega a {_num(mejor['mae_sin_historia_media'])} "
                  f"({_pct(mejor['mejora_vs_ingenua_sin_historia_pct'])}).")
    lineas.append("- **Aporte del contexto sobre la historia** (MAE de todas las filas; positivo = el contexto reduce el error):")
    lineas.append("  " + _frase(comparar(resumen, resultados, "B_historia_contexto", "B_solo_historia"), "B (historia + contexto)", "B solo historia").replace("\n", " "))
    lineas.append("  " + _frase(comparar(resumen, resultados, "C_correccion", "C_correccion_solo_historia"), "C (corrección con contexto)", "C solo historia"))
    lineas.append("- **Lo mismo, solo en filas con historia:**")
    lineas.append("  " + _frase(comparar(resumen, resultados, "B_historia_contexto", "B_solo_historia", "mae_con_historia"), "B", "B solo historia", "mae_con_historia"))
    lineas.append("  " + _frase(comparar(resumen, resultados, "C_correccion", "C_correccion_solo_historia", "mae_con_historia"), "C", "C solo historia", "mae_con_historia"))
    lineas.append("- **Otras comparaciones:**")
    lineas.append("  " + _frase(comparar(resumen, resultados, "C_correccion", "B_historia_contexto"), "C (corrección)", "B (directa)"))
    lineas.append("  " + _frase(comparar(resumen, resultados, "B_historia_contexto", "A_contexto"), "B (con historia)", "A (solo contexto)"))
    return "\n".join(lineas)


def escribir_informe(resultados: pd.DataFrame, resumen: pd.DataFrame, meta: dict, ruta: str | Path) -> str:
    n_cfg = int((resumen["tipo"] == "modelo").sum())
    secciones = "\n\n".join(
        f"### {f}\n\n{_tabla_formulacion(resumen, f)}" for f in FORMULACIONES if (resumen["formulacion"] == f).any())
    texto = f"""# Resultados v2: historia de la IES y corrección sobre la ingenua

Generado por `python run_formulaciones.py` (motor en `src/modelado/formulaciones.py`). Todas las cifras salen de la corrida; no se editan a mano. Continúa `reports/resultados_v1.md`.

## Diseño

- **Datos:** `data/processed/dataset_ies.csv`, {f"{meta['filas']:,}".replace(',', '.')} filas y {meta['ies']} IES (ventana {meta['ventana'][0]}–{meta['ventana'][1]}). Variables de historia (solo años anteriores a t, ver `src/integrate/historia.py`): `tasa_t1`, `tasa_t2`, `tasa_media_previa`, `n_anios_previos`, `falta_historia`.
- **Formulaciones** (todas con GroupKFold por `cod_ies`, {meta['n_pliegues']} pliegues, **los mismos para todas las configuraciones, para las líneas base y para la v1**):
  - **A_contexto:** las variables de la v1 ({meta['n_contexto']} variables), sin historia.
  - **B_historia_contexto:** historia + todas las de A ({meta['n_contexto'] + 5} variables).
  - **B_solo_historia:** solo las 5 variables de historia (aísla el aporte del contexto).
  - **C_correccion:** el objetivo es `tasa_t − ingenua` (ingenua = `tasa_t1`, o la mediana del entrenamiento si no hay); el modelo lo predice con historia + contexto y la predicción final es ingenua + cambio predicho. Sin logit.
  - **C_correccion_solo_historia** (adición, para aislar el aporte del contexto dentro de la corrección): como C con solo las 5 variables de historia.
- **Grilla ({n_cfg} configuraciones):** imputación mediana; selección (ninguna | información mutua k = {meta['k_info_mutua']} | PCA {int(round(100 * meta['varianza_pca']))} %) × modelo (RidgeCV | Random Forest | XGBoost, hiperparámetros por defecto de la v1) × objetivo (ninguna | logit; en C solo ninguna). Semilla {meta['semilla']}.
- **Métricas:** MAE, RMSE y R² por pliegue en escala de tasa, con las predicciones finales recortadas a [0, 1]; media ± desviación entre pliegues. «Con historia» son las filas que tienen `tasa_t1`; «sin historia», las que no (primer año de la IES en el dataset o sin fila del año anterior). La mejora es el cambio porcentual del MAE medio frente a la ingenua (positiva = menos error).
- **Tiempo total de la corrida:** {meta['tiempo_total_s']:.0f} s ({_num(meta['tiempo_total_s'] / 60, 1)} min).

## Líneas base

{_tabla_base(resumen, resultados)}

En las filas sin historia la ingenua no tiene tasa t-1 y usa la mediana del entrenamiento, así que su MAE ahí es el de la mediana: la comparación frente a la ingenua solo es una comparación de persistencia en las filas con historia.

## Conclusión principal

{_conclusion(resumen, resultados)}

## Mejor configuración por formulación

{_mejores(resumen)}

![MAE por formulación](figuras/mae_formulaciones.png)

## Tablas por formulación (ordenadas por MAE)

{secciones}

## Supuestos y limitaciones

- **Disponibilidad de `tasa_t1`:** se asume que la tasa anual de t-1 está disponible al inicio del año t. La fórmula (desertores de t / matriculados t-2 semestres) y la definición de desertor (dos o más períodos consecutivos sin matrícula) son del glosario de SPADIES; el glosario y las preguntas frecuentes **no dicen cuándo se publican los datos** (ver la bitácora). Si en la práctica el último año completo estuviera disponible con más rezago, las formulaciones B y C (y la propia línea base ingenua) estarían usando información que no se tendría al predecir.
- **Primer año sin historia:** el dataset empieza en 2015, así que las filas de 2015 no tienen `tasa_t1` por construcción (la mayor parte de las filas «sin historia»). SPADIES trae la serie desde 1998: ampliar la historia hacia atrás con años previos a 2015 es posible y no se hizo aquí.
- **Persistencia:** no se analizó cuánto de la persistencia de la tasa entre años es mecánica por cómo SPADIES clasifica a los desertores (dos o más períodos sin matrícula).
- **Solo historia con información mutua:** con 5 variables, `k = 15` se reduce a 5 (conserva todas), de modo que `info_mutua` equivale a `ninguna` en `B_solo_historia` y `C_correccion_solo_historia`.
- **Hiperparámetros por defecto** (no es el modelo final); `RidgeCV` elige su alfa con validación interna sin agrupar por IES dentro del entrenamiento (sin fuga hacia el pliegue de prueba).
- **Una sola partición agrupada** ({meta['n_pliegues']} pliegues): diferencias menores que la desviación entre pliegues no son concluyentes. Las comparaciones por pliegue de arriba usan los mismos pliegues para todas las configuraciones.
- El sesgo de selección, supervivencia y geografía de `reports/analisis_sesgo.md` aplica igual a estos resultados.
"""
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    Path(ruta).write_text(texto, encoding="utf-8")
    return texto
