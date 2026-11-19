"""Informe de complejidad: `reports/07_complejidad.md` y las dos figuras.

Todas las cifras salen de los CSV que escribe `desercion evaluar complejidad` (y de los resultados anteriores para el error y el costo del protocolo); ninguna se escribe a mano.
La tabla teórica (notación O) es la de `complejidad.teoria()`.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from desercion_ies.analysis.complejidad import _teoria_de, teoria  # noqa: E402
from desercion_ies.reporting.informe_experimentos import _num, _pct  # noqa: E402
from desercion_ies.reporting.informe_v3 import _c, _p  # noqa: E402

COLORES = [plt.cm.tab20(i) for i in range(20)]


def _ms(x_s: float) -> str:
    """Segundos → milisegundos con una cantidad de decimales adecuada."""
    ms = x_s * 1000.0
    d = 3 if ms < 1 else (2 if ms < 10 else (1 if ms < 100 else 0))
    return _num(ms, d)


def _t(ms: float) -> str:
    d = 3 if ms < 1 else (2 if ms < 10 else (1 if ms < 100 else 0))
    return _num(ms, d)


def _legible(s: float) -> str:
    """Segundos en una unidad legible (s, min, h, días)."""
    if s < 1:
        return f"{_num(s * 1000, 0)} ms"
    if s < 120:
        return f"{_num(s, 1)} s"
    if s < 7200:
        return f"{_num(s / 60, 1)} min"
    if s < 172800:
        return f"{_num(s / 3600, 1)} h"
    return f"{_num(s / 86400, 1)} días"


def _bytes(b: float) -> str:
    return f"{_num(b / 2**20, 1)} MiB" if b < 2**30 else f"{_num(b / 2**30, 1)} GiB" if b < 2**40 else f"{_num(b / 2**40, 1)} TiB"


# ---------- error de cada modelo (de los resultados anteriores) ----------
def mapa_error(ajuste_resumen: pd.DataFrame, componentes: pd.Series) -> pd.DataFrame:
    """MAE medio (ajuste anidado, 25 ajustes externos) de cada componente que es un modelo o una línea base, buscado en `ajuste_resumen.csv`."""
    a = ajuste_resumen.set_index(["configuracion", "variante"])
    filas = []
    for c in componentes:
        if c.startswith("T1 naive"):
            clave = ("T1 base_ingenua", "linea_base")
        elif c.startswith("T2 mediana"):
            clave = ("T2 base_mediana_grupo", "linea_base")
        elif c.startswith("T1 "):
            clave = (c, "ajustado")
        elif c.startswith("T2 "):
            modelo = c.split(" ")[1]
            con = "con clusters" in c
            variante = "defecto" if "(defecto)" in c else "ajustado"
            nombre = next((n for n, v in a.index if n.startswith(f"T2 | {modelo} |") and n.endswith(f"| {'con' if con else 'sin'} clusters") and v == variante), None)
            clave = (nombre, variante)
        else:
            continue
        if clave in a.index:
            f = a.loc[clave]
            filas.append({"componente": c, "mae": float(f["mae_media"]), "mae_ic_inf": float(f["mae_ic_inf"]), "mae_ic_sup": float(f["mae_ic_sup"]),
                          "mejora_base_pct": float(f["mejora_base_pct"]) if "mejora_base_pct" in f and not pd.isna(f["mejora_base_pct"]) else 0.0,
                          "p_t_base": float(f["p_t_base"]) if "p_t_base" in f and not pd.isna(f["p_t_base"]) else np.nan})
    return pd.DataFrame(filas)


def frontera_pareto(t: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Índices de los puntos no dominados (ninguno es a la vez más barato y con menos error)."""
    return np.array([i for i in range(len(t)) if not any((t[j] <= t[i] and e[j] <= e[i]) and (t[j] < t[i] or e[j] < e[i]) for j in range(len(t)))], dtype=int)


# ---------- tablas ----------
def _tabla_entorno(env: dict) -> str:
    filas = [("CPU", env["cpu"]), ("Núcleos físicos / hilos lógicos", f"{env['nucleos_fisicos']} / {env['hilos_logicos']}"), ("RAM", f"{_num(env['ram_gib'], 1)} GiB"), ("Sistema operativo", env["sistema_operativo"]),
             ("Python", env["python"]), ("scikit-learn", env["scikit_learn"]), ("xgboost", env["xgboost"]), ("numpy", env["numpy"]), ("pandas / scipy", f"{env['pandas']} / {env['scipy']}"),
             ("Bibliotecas de hilos", env["bibliotecas_de_hilos"]), ("Hilos usados en la medición", "1 (`n_jobs=1`, xgboost `nthread=1`, BLAS y OpenMP limitados a 1)"), ("Afinidad", env.get("afinidad", "—")),
             ("Calentamiento y repeticiones", f"{env.get('calentamiento', 1)} y {env.get('repeticiones', 5)}; se reporta la mediana y el rango intercuartílico")]
    return "| Característica | Valor |\n|---|---|\n" + "\n".join(f"| {a} | {b} |" for a, b in filas)


def _tabla_teorica(nombres: list[str]) -> str:
    out = ["| Componente | Ajuste | Predicción | Memoria | Exponente teórico en n |", "|---|---|---|---|---|"]
    vistos = set()
    for n in nombres:
        t = _teoria_de(n)
        familia = n if n in teoria() else next(f for f in ("random_forest", "xgboost", "ridge") if f in n)
        if familia in vistos:
            continue
        vistos.add(familia)
        lo, hi = t["exp"]
        exp = _num(lo, 1) if lo == hi else f"{_num(lo, 1)}–{_num(hi, 1)}"
        out.append(f"| {familia if familia in ('random_forest', 'xgboost', 'ridge') else n} | {_c(t['ajuste'])} | {_c(t['prediccion'])} | {_c(t['memoria'])} | {exp} |")
    return "\n".join(out)


def _tabla_medida(c: pd.DataFrame) -> str:
    out = ["| Componente | Tipo | Filas | Ajuste, ms: mediana [RIQ] | Predicción por 1.000 filas, ms: mediana [RIQ] | Memoria pico (tracemalloc), MiB | Memoria residente añadida (RSS), MiB |", "|---|---|---|---|---|---|---|"]
    for _, f in c.iterrows():
        out.append(f"| {_c(f['componente'])} | {f['tipo'].replace('linea_base', 'línea base').replace('preproceso', 'preprocesamiento')} | {int(f['n_filas'])} | {_ms(f['ajuste_mediana_s'])} [{_ms(f['ajuste_riq_s'])}] "
                   f"| {_t(f['pred_1000_mediana_ms'])} [{_t(f['pred_1000_riq_ms'])}] | {_num(f['memoria_pico_mib'], 1)} | {_num(f['memoria_rss_mib'], 1)} |")
    return "\n".join(out)


def _tabla_exponentes(e: pd.DataFrame) -> str:
    out = ["| Componente | Filas (mín.–máx.) | Ajuste (mín. → máx.), ms | Exponente empírico ± error estándar | R² | Exponente teórico | Comparación |", "|---|---|---|---|---|---|---|"]
    for _, f in e.iterrows():
        lo, hi = f["exponente_teorico_min"], f["exponente_teorico_max"]
        teo = _num(lo, 2) if abs(hi - lo) < 1e-9 else f"{_num(lo, 2)}–{_num(hi, 2)}"
        out.append(f"| {_c(f['componente'])} | {int(f['n_min'])}–{int(f['n_max'])} | {_ms(f['t_min_s'])} → {_ms(f['t_max_s'])} | {_num(f['exponente_empirico'], 2)} ± {_num(f['error_estandar'], 2)} | {_num(f['r2'], 3)} | {teo} | {f['comparacion']} |")
    return "\n".join(out)


def _entero(x) -> str:
    """Entero con punto como separador de miles (estilo español)."""
    return f"{int(x):,}".replace(",", ".")


def _tabla_protocolo(p: pd.DataFrame) -> str:
    out = ["| Etapa | Configuraciones | Ajustes | Tiempo de pared | Paralelismo | Suma de tiempos de ajuste medidos (referencia) | Detalle |", "|---|---|---|---|---|---|---|"]
    for _, f in p.iterrows():
        detalle = f["detalle"] if isinstance(f["detalle"], str) else ""
        out.append(f"| {f['etapa']} | {int(f['configuraciones'])} | {_entero(f['ajustes'])} | {_legible(f['tiempo_pared_s'])} | {f['paralelismo']} | {_legible(f['suma_t_ajuste_s'])} | {detalle} |")
    out.append(f"| **Total** | — | **{_entero(p['ajustes'].sum())}** | **{_legible(p['tiempo_pared_s'].sum())}** | — | **{_legible(p['suma_t_ajuste_s'].sum())}** | |")
    return "\n".join(out)


def _tabla_costo_error(ce: pd.DataFrame, tarea: str) -> str:
    g = ce[ce["tarea"] == tarea].sort_values("ajuste_mediana_s")
    base = g[g["tipo"] == "linea_base"]["mae"].iloc[0]
    par = set(g.iloc[frontera_pareto(g["ajuste_mediana_s"].to_numpy(), g["mae"].to_numpy())]["componente"])
    out = ["| Modelo | Ajuste, ms (mediana) | MAE [IC 95 %] | Mejora frente a la línea base | p t corregida | ¿En la frontera costo/error? |", "|---|---|---|---|---|---|"]
    for _, f in g.iterrows():
        es_base = f["tipo"] == "linea_base"
        out.append(f"| {'**' if es_base else ''}{_c(f['componente'])}{'**' if es_base else ''} | {_ms(f['ajuste_mediana_s'])} | {_num(f['mae'], 4)} [{_num(f['mae_ic_inf'], 4)}; {_num(f['mae_ic_sup'], 4)}] "
                   f"| {'—' if es_base else _pct(100 * (base - f['mae']) / base)} | {'—' if es_base else _p(f['p_t_base'])} | {'sí' if f['componente'] in par else 'no'} |")
    return "\n".join(out)


def _tabla_escala(e: pd.DataFrame, c: pd.DataFrame) -> str:
    n0 = int(c["n_filas"].max())
    out = [f"| Componente | Exponente empírico | Ajuste hoy ({n0:,} filas) | Estimado con 10× filas | Estimado con 100× filas | Memoria de la matriz n × n (teórica; solo KernelPCA) |".replace(",", "."), "|---|---|---|---|---|---|"]
    for _, f in e.iterrows():
        b, t = f["exponente_empirico"], f["t_max_s"]
        mem = f"10×: {_bytes(8 * (10 * n0) ** 2)}; 100×: {_bytes(8 * (100 * n0) ** 2)}" if f["componente"].startswith("KernelPCA") else "—"
        out.append(f"| {_c(f['componente'])} | {_num(b, 2)} | {_legible(t)} | {_legible(t * 10 ** b)} | {_legible(t * 100 ** b)} | {mem} |")
    return "\n".join(out)


# ---------- conclusión ----------
def _mejor_relacion(ce: pd.DataFrame, tarea: str) -> str:
    g = ce[ce["tarea"] == tarea]
    base = g[g["tipo"] == "linea_base"].iloc[0]
    m = g[g["tipo"] == "modelo"].copy()
    m["mejora"] = base["mae"] - m["mae"]
    mejor = m.sort_values("mae").iloc[0]
    par = g.iloc[frontera_pareto(g["ajuste_mediana_s"].to_numpy(), g["mae"].to_numpy())]
    if mejor["mejora"] <= 0:
        return (f"- **Tarea {tarea[1]}:** ningún modelo mejora a la línea base (`{base['componente']}`, MAE {_num(base['mae'], 4)}, ajuste {_ms(base['ajuste_mediana_s'])} ms); el mejor modelo, `{mejor['componente']}`, "
                f"tiene MAE {_num(mejor['mae'], 4)} y cuesta {_ms(mejor['ajuste_mediana_s'])} ms. La mejor relación costo/error es la de la línea base.")
    candidatos = m[m["mejora"] >= 0.9 * mejor["mejora"]].sort_values("ajuste_mediana_s")
    elegido = candidatos.iloc[0]
    firme = "sin firmeza estadística (p t corregida " + _p(elegido["p_t_base"]) + ")" if not np.isnan(elegido["p_t_base"]) and elegido["p_t_base"] >= 0.05 else "con firmeza (p t corregida " + _p(elegido["p_t_base"]) + ")"
    caros = m[(m["ajuste_mediana_s"] > elegido["ajuste_mediana_s"] * 3) & (m["mae"] >= elegido["mae"])]
    return (f"- **Tarea {tarea[1]}:** la mejor relación costo/error es `{elegido['componente']}`: MAE {_num(elegido['mae'], 4)} ({_pct(100 * elegido['mejora'] / base['mae'])} frente a la línea base, {firme}) "
            f"con un ajuste de {_ms(elegido['ajuste_mediana_s'])} ms, el más barato que logra al menos 90 % de la mejora del mejor modelo (`{mejor['componente']}`, MAE {_num(mejor['mae'], 4)}, {_ms(mejor['ajuste_mediana_s'])} ms). "
            f"La línea base cuesta {_ms(base['ajuste_mediana_s'])} ms (MAE {_num(base['mae'], 4)}). "
            + (f"Cuestan más de 3 veces su ajuste sin tener menos error: {', '.join('`' + x + '`' for x in caros['componente'])}. " if len(caros) else "")
            + f"Frontera costo/error: {', '.join('`' + x + '`' for x in par.sort_values('ajuste_mediana_s')['componente'])}."
            + (f" Como esa mejora no es firme, la línea base (`{base['componente']}`, {_ms(base['ajuste_mediana_s'])} ms) es la alternativa razonable: cuesta {_num(elegido['ajuste_mediana_s'] / base['ajuste_mediana_s'], 0)} veces menos."
               if not np.isnan(elegido["p_t_base"]) and elegido["p_t_base"] >= 0.05 else ""))


def _conclusion(c, e, ce) -> str:
    L = ["**1. ¿Qué modelo ofrece la mejor relación costo/error en cada tarea?** (costo = tiempo de ajuste con un hilo sobre las 1.928 filas; error = MAE del ajuste anidado de `04_ajuste_temporal.md`)"]
    L += [_mejor_relacion(ce, "T1"), _mejor_relacion(ce, "T2")]
    L.append("\n**2. ¿Algún método escalaría mal con 10 o 100 veces más filas (por ejemplo, datos por estudiante)?** Con 4 puntos el exponente es aproximado; las extrapolaciones suponen que el exponente medido se mantiene.")
    malos = e[e["exponente_empirico"] >= 1.5].sort_values("exponente_empirico", ascending=False)
    n0 = int(c["n_filas"].max())
    if malos.empty:
        L.append("- Ninguno de los componentes medidos tiene un exponente empírico de 1,5 o más.")
    for _, f in malos.iterrows():
        extra = f" y una matriz n × n de {_bytes(8 * (10 * n0) ** 2)} con 10× filas ({_bytes(8 * (100 * n0) ** 2)} con 100×)" if f["componente"].startswith("KernelPCA") else ""
        L.append(f"- `{f['componente']}`: exponente empírico {_num(f['exponente_empirico'], 2)} ± {_num(f['error_estandar'], 2)} (teórico {_num(f['exponente_teorico_min'], 1)}–{_num(f['exponente_teorico_max'], 1)}): de {_legible(f['t_max_s'])} hoy pasaría a "
                 f"{_legible(f['t_max_s'] * 10 ** f['exponente_empirico'])} con 10× filas y a {_legible(f['t_max_s'] * 100 ** f['exponente_empirico'])} con 100×{extra}.")
    teorico_alto = e[(e["exponente_empirico"] < 1.5) & (e["exponente_teorico_max"] >= 1.5)]
    for _, f in teorico_alto.iterrows():
        L.append(f"- `{f['componente']}`: en teoría es de orden n² ({_num(f['exponente_teorico_min'], 1)}–{_num(f['exponente_teorico_max'], 1)}) pero con estos tamaños mide {_num(f['exponente_empirico'], 2)} ± {_num(f['error_estandar'], 2)}, porque todavía domina el costo de "
                 f"K-means; con muchas más filas el término n² (la silueta compara cada fila con todas las demás) pasaría a dominar. Como cota superior teórica, con exponente 2 pasaría de {_legible(f['t_max_s'])} a {_legible(f['t_max_s'] * 100)} con 10× filas y a {_legible(f['t_max_s'] * 10000)} con 100×; "
                 "es teoría, no una medición.")
    lineales = e[(e["exponente_empirico"] < 1.5) & (e["t_max_s"] > 0.5)].sort_values("t_max_s", ascending=False)
    if len(lineales):
        f = lineales.iloc[0]
        L.append(f"- Los demás escalan aproximadamente de forma lineal o casi lineal; el más caro hoy es `{f['componente']}` ({_legible(f['t_max_s'])}; exponente {_num(f['exponente_empirico'], 2)}): con 100× filas serían unos {_legible(f['t_max_s'] * 100 ** f['exponente_empirico'])}.")
    L.append("- Con datos por estudiante (cientos de miles o millones de filas) los métodos de orden n² (KernelPCA, imputación KNN y la selección de k por silueta) dejarían de ser viables tal como están implementados; las alternativas habituales "
             "son aproximar el kernel (Nyström o características aleatorias de Fourier), imputar con un método lineal o por submuestra, K-means por lotes (`MiniBatchKMeans`) y calcular la silueta sobre una submuestra. Esto último es una recomendación general, no se midió aquí.")
    return "\n".join(L)


# ---------- figuras ----------
def figura_costo_error(ce: pd.DataFrame, ruta: str | Path) -> None:
    fig, ejes = plt.subplots(1, 2, figsize=(14, 5.6))
    for ax, tarea, titulo in zip(ejes, ("T1", "T2"), ("Tarea 1: con historia", "Tarea 2: sin historia")):
        g = ce[ce["tarea"] == tarea].reset_index(drop=True)
        t, e = g["ajuste_mediana_s"].to_numpy() * 1000, g["mae"].to_numpy()
        par = frontera_pareto(t, e)
        orden = par[np.argsort(t[par])]
        ax.plot(t[orden], e[orden], color="#999", lw=1, ls="--", zorder=1, label="frontera costo/error")
        for i, f in g.iterrows():
            base = f["tipo"] == "linea_base"
            ax.errorbar(t[i], e[i], xerr=[[max(t[i] - f["ajuste_q1_s"] * 1000, 0)], [max(f["ajuste_q3_s"] * 1000 - t[i], 0)]], fmt="D" if base else "o",
                        color="black" if base else COLORES[i % len(COLORES)], ecolor="#999", capsize=2, markersize=8, zorder=3)
            ax.annotate(f["componente"].replace("T1 ", "").replace("T2 ", "").replace(" (ajustado)", "").replace(" sin clusters", ""), (t[i], e[i]), textcoords="offset points",
                        xytext=(7, 7 if i % 2 == 0 else -12), fontsize=7)  # etiquetas alternadas arriba y abajo para que no se encimen
        ax.set_xscale("log")
        ax.set_xlabel("tiempo de ajuste, ms (mediana y RIQ; un hilo; escala log)")
        ax.set_ylabel("MAE medio del ajuste anidado (los IC están en la tabla)")
        ax.set_title(titulo, fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Costo de ajuste frente a error por modelo (rombo negro = línea base; barra horizontal = RIQ del tiempo; eje vertical ampliado: las diferencias de MAE son de milésimas)", fontsize=9)
    fig.tight_layout()
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def figura_escalado(esc: pd.DataFrame, exp: pd.DataFrame, ruta: str | Path) -> None:
    fig, ejes = plt.subplots(1, 2, figsize=(14, 7.2))
    e = exp.set_index("componente")
    for ax, tipos, titulo in zip(ejes, (("modelo", "linea_base"), ("preproceso",)), ("Modelos y líneas base", "Métodos de preprocesamiento")):
        g = esc[esc["tipo"].isin(tipos)]
        n0 = g["n_filas"].min()
        t0 = float(np.median(g[g["n_filas"] == n0]["ajuste_mediana_s"]))
        for i, (nombre, d) in enumerate(g.groupby("componente", sort=False)):
            d = d.sort_values("n_filas")
            ax.plot(d["n_filas"], d["ajuste_mediana_s"], marker="o", color=COLORES[i % len(COLORES)], lw=1.4,
                    label=f"{nombre.replace(' (ajustado)', '')} (b = {_num(e.loc[nombre, 'exponente_empirico'], 2)})")
            ax.fill_between(d["n_filas"], d["ajuste_q1_s"], d["ajuste_q3_s"], color=COLORES[i % len(COLORES)], alpha=0.12)
        nn = np.array([n0, g["n_filas"].max()], dtype=float)
        for b, estilo in ((1, ":"), (2, "--")):
            ax.plot(nn, t0 * (nn / n0) ** b, color="#555", ls=estilo, lw=1, label=f"referencia: pendiente {b}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("filas (submuestras por IES del 25, 50, 75 y 100 %; escala log)")
        ax.set_ylabel("tiempo de ajuste, s (mediana y RIQ; escala log)")
        ax.set_title(titulo, fontsize=10)
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Escalado empírico del tiempo de ajuste (b = pendiente log-log; con 4 puntos es aproximada)", fontsize=10)
    fig.tight_layout()
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


# ---------- informe ----------
def generar_informe(cfg: dict, s: dict[str, Path]) -> str:
    env = json.loads(Path(s["entorno"]).read_text(encoding="utf-8"))
    c, esc, exp, proto = (pd.read_csv(s[k]) for k in ("complejidad", "escalado", "exponentes", "protocolo"))
    ajuste = pd.read_csv(cfg["modelado"]["ajuste_temporal"]["salidas"]["ajuste_resumen"])
    err = mapa_error(ajuste, c["componente"])
    ce = c.merge(err, on="componente")
    figura_costo_error(ce, s["figura_costo_error"])
    figura_escalado(esc, exp, s["figura_escalado"])
    pc = cfg["modelado"]["complejidad"]
    medicion = Path(s["complejidad"]).with_name("complejidad_tiempo.json")
    t_med = json.loads(medicion.read_text())["t_medicion_s"] if medicion.exists() else None
    texto = f"""# Complejidad computacional: teórica y medida

Generado por `desercion evaluar complejidad` (módulos `src/desercion_ies/analysis/complejidad.py`, `costo_protocolo.py` e `informe_complejidad.py`). Todas las cifras salen de la corrida y de los resultados anteriores; no se editan a mano. No cambia ningún modelo, dato ni resultado anterior.

## 1. Cómo se midió

- **Un solo hilo y calentamiento.** Todas las mediciones con un hilo (`n_jobs=1`, xgboost `nthread=1`, BLAS y OpenMP limitados a 1 con `threadpoolctl` y con las variables de entorno fijadas antes de importar numpy). Cada medición tiene una ejecución de calentamiento (no se cuenta) y {pc['repeticiones']} repeticiones; se reporta la **mediana** y el **rango intercuartílico (RIQ)**.
- **Qué se cronometra.** *Ajuste:* `fit` en los modelos y `fit_transform` en los métodos de preprocesamiento (lo que cuesta entrenar). *Predicción:* `predict` o `transform` de todas las filas, expresada por 1.000 filas. Los datos se preparan fuera del cronómetro.
- **Memoria.** Pico de `tracemalloc` durante un ajuste, en una ejecución aparte (tracemalloc enlentece). Ve las asignaciones de Python y numpy pero **no** la memoria nativa de los árboles ni de xgboost, así que es un piso; como medida complementaria se muestrea la memoria residente (RSS) cada 2 ms y se reporta cuánto sube durante el ajuste (aproximada: puede perder picos breves y no baja de inmediato al liberar).
- **Modelos.** Los finales del análisis de sesgo, con sus hiperparámetros finales (consenso de los 25 pliegues del ajuste anidado): naive (tasa de t-1) y mediana por carácter × sector; ridge, random forest y xgboost de la Tarea 1 (C solo historia; B con xgboost) y de la Tarea 2 (con información mutua y con clusters). Los modelos de la Tarea 2 con clusters incluyen el costo de K-means con selección de k por silueta. Cada modelo incluye su preprocesamiento (imputación, escalado, one-hot).
- **Escalado.** Submuestras **por IES** (IES completas, semilla {pc['semilla_submuestra']}, anidadas) del {', '.join(f'{int(100 * f)} %' for f in pc['fracciones'])}; el exponente empírico es la pendiente log-log del tiempo de ajuste contra el número de filas.
- **Tiempo de la medición completa:** {_legible(t_med) if t_med else '—'}.

{_tabla_entorno(env)}

## Conclusión

{_conclusion(c, exp, ce)}

## 2. Complejidad teórica

n filas, p variables, T árboles, d profundidad, m variables que se prueban por división, k componentes, E épocas, h neuronas, I iteraciones de Lloyd, c clusters, K_c candidatos de k. Son las complejidades estándar de las implementaciones usadas (scikit-learn; xgboost con histogramas). Los modelos de la Tarea 2 con clusters suman el costo de «K-means + silueta».

{_tabla_teorica(list(c['componente']))}

## 3. Costo medido (todas las filas, un hilo)

Ajuste: mediana [RIQ] de {pc['repeticiones']} repeticiones; predicción por 1.000 filas. El tiempo de predicción de la imputación KNN y de KernelPCA depende del número de filas de entrenamiento (comparan cada fila nueva con todas las de entrenamiento).

{_tabla_medida(c)}

![Costo de ajuste frente a error](figures/complejidad_costo_error.png)

### Costo y error por tarea

**Tarea 1** (línea base: ingenua)

{_tabla_costo_error(ce, 'T1')}

**Tarea 2** (línea base: mediana por carácter × sector)

{_tabla_costo_error(ce, 'T2')}

## 4. Escalado empírico y exponentes

Con 4 puntos (25, 50, 75 y 100 % de las IES) el exponente es **aproximado**: se reporta su error estándar y se considera «consistente» con el teórico si cae dentro del rango teórico ± (0,3 + 2 errores estándar). Para random forest, la mediana por grupo y la información mutua el teórico es el de n·log n en el rango medido. Un exponente **por debajo del teórico** indica que, con estos tamaños (480 a 1.928 filas), dominan costos fijos (construir el pipeline, cada árbol, las llamadas a las librerías) y no el crecimiento con n; en los componentes que tardan unos pocos milisegundos el exponente está dominado además por el ruido.

{_tabla_exponentes(exp)}

![Escalado](figures/complejidad_escalado.png)

### Extrapolación a 10 y 100 veces más filas

Es una **extrapolación** con el exponente empírico (no se midió); sirve para ver el orden de magnitud, no para planificar.

{_tabla_escala(exp, c)}

## 5. Costo del protocolo de evaluación

Número de ajustes de cada etapa (contados en los CSV de resultados; en la búsqueda interna, {cfg['modelado']['ajuste_temporal']['n_iter']} iteraciones × {cfg['modelado']['ajuste_temporal']['n_splits_interno']} pliegues internos + el reajuste) y tiempo de pared de cada corrida, tomado de los informes anteriores. Las etapas v1–v3 corrieron en un solo proceso; las demás, con 16 procesos en paralelo. La suma de tiempos de ajuste medidos viene de las columnas de tiempo de esos CSV y es solo una referencia (se midió con paralelismo y, en el ajuste anidado, incluye la búsqueda).

{_tabla_protocolo(proto)}

## Supuestos y limitaciones

- **Una máquina, un entorno:** los tiempos son de este equipo (tabla de entorno) y sirven para comparar componentes entre sí, no para predecir tiempos en otra máquina. La CPU es híbrida (núcleos de rendimiento y de eficiencia); el proceso se midió con la siguiente afinidad: {env.get('afinidad', 'sin dato')}; no se verificó de qué tipo de núcleo se trata.
- **Hiperparámetros finales:** el costo de random forest y xgboost depende de ellos (número de árboles, profundidad); son el consenso del ajuste anidado, no los que habría elegido cada pliegue.
- **Memoria:** `tracemalloc` es un piso (no ve la memoria nativa) y la memoria residente muestreada es aproximada; no hay una medición exacta del pico total.
- **Predicción sobre las filas de entrenamiento:** el costo de predicción se mide sobre las mismas 1.928 filas con las que se ajustó (para árboles y modelos lineales no cambia con filas nuevas; para imputación KNN y KernelPCA sí depende del tamaño del entrenamiento).
- **Exponentes con 4 puntos:** aproximados. El tamaño pequeño (≤ 1.928 filas) hace que los costos fijos de las librerías pesen en los componentes rápidos; en los lentos (autoencoder, KernelPCA, KNN) el rango es más informativo.
- **Autoencoder:** entrena hasta el máximo de iteraciones (la parada temprana no se activó en la Tarea 2), así que su costo crece de forma casi lineal con n pero con un tiempo alto por la constante.
"""
    s["informe"].parent.mkdir(parents=True, exist_ok=True)
    s["informe"].write_text(texto, encoding="utf-8")
    return texto
