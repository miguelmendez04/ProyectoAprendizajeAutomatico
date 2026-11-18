"""Informe de los experimentos: `reports/resultados_v1.md` y la figura del MAE por configuración.

Todo el texto con cifras se genera a partir de `resultados` y `resumen` (ver `experimentos.py`): ninguna cifra se escribe a mano.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from src.modelado.experimentos import BASE_INGENUA, BASE_MEDIANA  # noqa: E402

COLORES_MODELO = {"ridge": "#4c78a8", "random_forest": "#59a14f", "xgboost": "#e15759"}
FACTORES = [("modelo", "Modelo"), ("objetivo", "Objetivo"), ("imputacion", "Imputación"), ("seleccion", "Selección / reducción")]


def _pct(x: float) -> str:
    return f"{x:+.1f} %".replace(".", ",")


def _num(x: float, d: int = 4) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def _mas_menos(m: float, s: float, d: int = 4) -> str:
    return f"{_num(m, d)} ± {_num(s, d)}"


def figura_mae(resumen: pd.DataFrame, ruta: str | Path) -> None:
    """MAE medio ± desviación entre pliegues por configuración, con las dos líneas base marcadas."""
    modelos = resumen[resumen["tipo"] == "modelo"].sort_values("mae_media", ascending=False)  # el mejor queda arriba
    base = resumen.set_index("configuracion")
    fig, ax = plt.subplots(figsize=(9, 0.28 * len(modelos) + 2.6))
    ax.barh(modelos["configuracion"], modelos["mae_media"], xerr=modelos["mae_desv"],
            color=[COLORES_MODELO.get(m, "#888") for m in modelos["modelo"]], error_kw={"elinewidth": 1, "capsize": 2, "ecolor": "#333"})
    ax.axvline(base.loc[BASE_MEDIANA, "mae_media"], color="#555", ls="--", lw=1.4)
    ax.axvline(base.loc[BASE_INGENUA, "mae_media"], color="#000", ls="-", lw=1.4)
    ax.set_xlabel("MAE en escala de tasa (media ± desviación entre 5 pliegues, GroupKFold por IES)")
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylim(-0.7, len(modelos) - 0.3)
    leyenda = [Line2D([0], [0], color=c, lw=6, label=m) for m, c in COLORES_MODELO.items()]
    leyenda += [Line2D([0], [0], color="#555", ls="--", label=f"línea base mediana ({_num(base.loc[BASE_MEDIANA, 'mae_media'])})"),
                Line2D([0], [0], color="#000", ls="-", label=f"línea base ingenua t-1 ({_num(base.loc[BASE_INGENUA, 'mae_media'])})")]
    ax.legend(handles=leyenda, loc="upper center", bbox_to_anchor=(0.4, -0.06), ncol=2, fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def _tabla(resumen: pd.DataFrame) -> str:
    bases = resumen[resumen["tipo"] == "linea_base"].set_index("configuracion").loc[[BASE_INGENUA, BASE_MEDIANA]].reset_index()
    modelos = resumen[resumen["tipo"] == "modelo"]
    filas = ["| Configuración (modelo \\| objetivo \\| imputación \\| selección) | MAE | RMSE | R² | Mejora vs. ingenua | Pliegues mejores que la ingenua | Ajuste (s) |",
             "|---|---|---|---|---|---|---|"]
    for _, r in pd.concat([bases, modelos]).iterrows():
        negrita = "**" if r["tipo"] == "linea_base" else ""
        filas.append(
            f"| {negrita}{r['configuracion']}{negrita} | {_mas_menos(r['mae_media'], r['mae_desv'])} | {_mas_menos(r['rmse_media'], r['rmse_desv'])} "
            f"| {_mas_menos(r['r2_media'], r['r2_desv'], 3)} | {'—' if r['configuracion'] == BASE_INGENUA else _pct(r['mejora_vs_ingenua_pct'])} "
            f"| {'—' if r['configuracion'] == BASE_INGENUA else str(r['pliegues_mejor_que_ingenua']) + ' de ' + str(r['n_pliegues'])} "
            f"| {_num(r['t_ajuste_media_s'], 2)} |")
    return "\n".join(filas)


def _marginales(resumen: pd.DataFrame) -> str:
    m = resumen[resumen["tipo"] == "modelo"]
    bloques = []
    for col, titulo in FACTORES:
        g = m.groupby(col).agg(mae=("mae_media", "mean"), mejor=("mae_media", "min"), t=("t_ajuste_media_s", "mean"), n=("mae_media", "size"))
        g = g.sort_values("mae")
        filas = [f"| {titulo} | MAE medio de sus configuraciones | Mejor MAE | Ajuste medio (s) | Configuraciones |", "|---|---|---|---|---|"]
        filas += [f"| {k} | {_num(r['mae'])} | {_num(r['mejor'])} | {_num(r['t'], 2)} | {int(r['n'])} |" for k, r in g.iterrows()]
        bloques.append("\n".join(filas))
    return "\n\n".join(bloques)


def _analisis(resumen: pd.DataFrame, resultados: pd.DataFrame) -> str:
    base = resumen.set_index("configuracion")
    ing, med = base.loc[BASE_INGENUA], base.loc[BASE_MEDIANA]
    m = resumen[resumen["tipo"] == "modelo"].reset_index(drop=True)
    mejor = m.iloc[0]
    supera_ing = m[m["mae_media"] < ing["mae_media"]]
    supera_med = m[m["mae_media"] < med["mae_media"]]
    lineas = []
    lineas.append(
        f"- **Mejor configuración:** `{mejor['configuracion']}`, con MAE {_mas_menos(mejor['mae_media'], mejor['mae_desv'])} "
        f"({_pct(mejor['mejora_vs_ingenua_pct'])} frente a la ingenua; mejor que ella en {mejor['pliegues_mejor_que_ingenua']} de {mejor['n_pliegues']} pliegues).")
    if len(supera_ing):
        lineas.append(f"- **Frente a la línea base ingenua:** {len(supera_ing)} de {len(m)} configuraciones tienen MAE medio menor que la ingenua "
                      f"({_num(ing['mae_media'])}).")
    else:
        lineas.append(f"- **Frente a la línea base ingenua:** ninguna de las {len(m)} configuraciones tiene MAE medio menor que el de la ingenua "
                      f"({_num(ing['mae_media'])}); la mejor llega a {_num(mejor['mae_media'])}.")
    lineas.append(f"- **Frente a la mediana:** {len(supera_med)} de {len(m)} configuraciones tienen MAE medio menor que el de la mediana ({_num(med['mae_media'])}); "
                  f"la mejora de la mejor configuración sobre la mediana es {_pct(100 * (med['mae_media'] - mejor['mae_media']) / med['mae_media'])}.")
    if med["mae_media"] > ing["mae_media"]:
        lineas.append(f"- **Las dos líneas base:** la ingenua ({_num(ing['mae_media'])}) mejora a la mediana ({_num(med['mae_media'])}) en "
                      f"{_pct(100 * (med['mae_media'] - ing['mae_media']) / med['mae_media'])}: la tasa de una IES persiste de un año al siguiente.")
    t1, tm = int(resultados.loc[resultados["configuracion"] == BASE_INGENUA, "n_ingenua_t1"].sum()), int(
        resultados.loc[resultados["configuracion"] == BASE_INGENUA, "n_ingenua_mediana"].sum())
    lineas.append(f"- **Cobertura de la ingenua:** usó la tasa t-1 de la misma IES en {t1} filas de prueba y la mediana del entrenamiento en {tm} "
                  f"(de {t1 + tm}); la mediana aplica en 2015 y cuando la IES no tiene fila del año anterior en el dataset.")
    porm = m.groupby("modelo")["mae_media"].mean().sort_values()
    lineas.append(f"- **Modelo:** por MAE medio de sus configuraciones, el orden es " + ", ".join(f"{k} ({_num(v)})" for k, v in porm.items()) + ".")
    for col, titulo in FACTORES[1:]:
        porf = m.groupby(col)["mae_media"].mean().sort_values()
        dif = porf.iloc[-1] - porf.iloc[0]
        lineas.append(f"- **{titulo}:** " + ", ".join(f"{k} ({_num(v)})" for k, v in porf.items())
                      + f"; la diferencia entre el mejor y el peor nivel es {_num(dif)} de MAE.")
    mejores_por_modelo = m.sort_values("mae_media").groupby("modelo").head(1).sort_values("mae_media")
    lineas.append("- **Mejor combinación por modelo:** " + "; ".join(
        f"`{r['configuracion']}` ({_num(r['mae_media'])})" for _, r in mejores_por_modelo.iterrows()) + ".")
    return "\n".join(lineas)


def escribir_informe(resultados: pd.DataFrame, resumen: pd.DataFrame, meta: dict, ruta: str | Path) -> str:
    """Escribe el informe en markdown y lo devuelve. `meta`: filas, ies, semilla, versiones, tiempo_total_s, rutas."""
    n_cfg = int((resumen["tipo"] == "modelo").sum())
    texto = f"""# Resultados v1: primera tabla de experimentos

Generado por `python run_experimentos.py` (motor en `src/modelado/experimentos.py`). Todas las cifras salen de la corrida; no se editan a mano.

## Diseño

- **Datos:** `data/processed/dataset_ies.csv`, {f"{meta['filas']:,}".replace(',', '.')} filas y {meta['ies']} IES (ventana {meta['ventana'][0]}–{meta['ventana'][1]}); {meta['n_numericas']} variables numéricas y {meta['n_categoricas']} categóricas (sector y carácter), sin las excluidas por defecto ni `NUNCA_EXPLICATIVAS`.
- **Validación:** GroupKFold por `cod_ies`, {meta['n_pliegues']} pliegues, **los mismos para las {n_cfg} configuraciones y las dos líneas base** (la huella de las IES de prueba de cada pliegue está en `hash_prueba`). Preprocesamiento, selección y modelo se ajustan solo con el entrenamiento de cada pliegue.
- **Grilla ({n_cfg} configuraciones, hiperparámetros por defecto):** objetivo (ninguna | logit) × imputación (mediana | KNN, k = {meta['k_vecinos']}) × selección (ninguna | información mutua con k = {meta['k_info_mutua']} | PCA con {int(round(100 * meta['varianza_pca']))} % de varianza) × modelo (RidgeCV con alfas logspace(-3, 3, 13) | Random Forest de {meta['n_estimators_rf']} árboles | XGBoost de {meta['n_estimators_xgb']} árboles, tasa de aprendizaje {meta['lr_xgb']}, profundidad {meta['depth_xgb']}).
- **Métricas:** MAE, RMSE y R² por pliegue, siempre en escala de tasa (0–1), aunque se entrene en logit; en la tabla, media ± desviación entre pliegues. Las predicciones no se recortan a [0, 1].
- **Líneas base:** `base_mediana` (mediana de la tasa del entrenamiento) y `base_ingenua` (tasa del año anterior de la misma IES; si no hay fila t-1, la mediana del entrenamiento).
- **Semilla:** {meta['semilla']}. scikit-learn {meta['sklearn']}, xgboost {meta['xgboost']}. Tiempo total de la corrida: {meta['tiempo_total_s']:.0f} s ({_num(meta['tiempo_total_s'] / 60, 1)} min).

## Tabla de resultados (ordenada por MAE; líneas base arriba)

La mejora es el cambio porcentual del MAE medio frente a la línea base ingenua (positiva = menor error que la ingenua). «Pliegues mejores» compara cada pliegue con el de la ingenua.

{_tabla(resumen)}

![MAE por configuración](figuras/mae_configuraciones.png)

## Análisis

{_analisis(resumen, resultados)}

### Efecto de cada decisión (promedio sobre las demás)

{_marginales(resumen)}

## Limitaciones de esta primera tabla

- Hiperparámetros por defecto: es una comparación de configuraciones, no el modelo final. El ajuste viene después.
- `RidgeCV` elige su alfa con validación cruzada interna sin agrupar por IES (dentro del entrenamiento de cada pliegue), así que puede elegir un alfa algo pequeño; no hay fuga hacia el pliegue de prueba.
- La línea base ingenua usa la tasa observada del año anterior de la IES, que es un dato disponible al predecir el año t; los modelos de la grilla no la reciben como variable. Es la comparación que dice cuánto aporta el contexto territorial frente a la persistencia de cada IES.
- Con {meta['n_pliegues']} pliegues, las diferencias menores que la desviación entre pliegues no son concluyentes.
- Una sola partición agrupada (GroupKFold determinista): no se repite con otras semillas de pliegues.
- El sesgo de selección, supervivencia y geografía descrito en `reports/analisis_sesgo.md` aplica igual a estos resultados.
"""
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    Path(ruta).write_text(texto, encoding="utf-8")
    return texto
