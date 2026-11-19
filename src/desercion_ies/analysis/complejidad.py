"""Costo computacional medido de forma controlada: modelos finales, métodos de preprocesamiento y escalado con el número de filas.

**Control de la medición**
- **Un solo hilo:** `threadpoolctl.threadpool_limits(1)` (BLAS, OpenMP) y `n_jobs=1` en los modelos (xgboost con un hilo, `nthread=1`). El ejecutor además fija las
  variables de entorno `OMP_NUM_THREADS`, `MKL_NUM_THREADS` y `OPENBLAS_NUM_THREADS` en 1 antes de importar numpy, y puede fijar la afinidad a una CPU lógica
  (en CPU híbridas, Windows puede mover el proceso entre núcleos rápidos y lentos).
- **Calentamiento y repeticiones:** una ejecución de calentamiento (no se cuenta) y luego `repeticiones` (5) mediciones; se reporta la mediana y el rango intercuartílico (RIQ).
- **Qué se mide.** *Ajuste:* `fit` en los modelos y `fit_transform` en los transformers (es lo que cuesta entrenar). *Predicción:* `predict` o `transform` sobre todas las filas, expresada
  por 1.000 filas. *Memoria:* pico de `tracemalloc` durante el ajuste, en una ejecución aparte (tracemalloc enlentece, así que nunca se mezcla con los tiempos). `tracemalloc`
  ve las asignaciones de Python y de numpy pero **no** la memoria nativa de xgboost ni de otras librerías en C: es un piso, no el total.
- **Escalado:** submuestras por IES (se toman IES completas, con una semilla fija, anidadas: la de 25 % está dentro de la de 50 %, etc.), y el exponente empírico es la pendiente
  del tiempo de ajuste contra el número de filas en escala log-log. Con 4 puntos es aproximado (se reporta su error estándar).

Los modelos y las líneas base son los finales del análisis de sesgo (hiperparámetros por consenso de los 25 pliegues del ajuste anidado); no se cambia ninguno.
"""
import json
import os
import platform
import subprocess
import sys
import time
import tracemalloc
from functools import partial

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.impute import KNNImputer, SimpleImputer
from threadpoolctl import threadpool_limits

from desercion_ies.evaluation import ajuste as aj
from desercion_ies.evaluation import evaluacion_final as ef
from desercion_ies.evaluation import tarea2 as t2
from desercion_ies.evaluation.experimentos import parametros_experimentos
from desercion_ies.models.metodos import AgregarClusters, AutoencoderPequeno, KernelPCAComoPCA
from desercion_ies.features.particion import construir_preprocesador

_DEFECTO = {"repeticiones": 5, "calentamiento": 1, "fracciones": [0.25, 0.5, 0.75, 1.0], "semilla_submuestra": 7, "kmeans_k": 6, "afinidad_cpu": 0}


def parametros_complejidad(cfg: dict) -> dict:
    p = dict(_DEFECTO)
    p.update(cfg.get("modelado", {}).get("complejidad") or {})
    if p["repeticiones"] < 1 or any(not 0 < f <= 1 for f in p["fracciones"]):
        raise ValueError("complejidad: repeticiones >= 1 y fracciones en (0, 1]")
    return p


# ---------- entorno ----------
def _powershell(comando: str) -> str | None:
    try:
        return subprocess.run(["powershell", "-NoProfile", "-Command", comando], capture_output=True, text=True, timeout=60).stdout.strip() or None
    except Exception:  # noqa: BLE001  (el entorno es informativo: cualquier fallo se reporta como ausente)
        return None


def entorno() -> dict:
    """CPU, núcleos, RAM, sistema operativo y versiones de Python y de las librerías usadas."""
    import numpy
    import pandas
    import scipy
    import sklearn
    import xgboost

    cpu, fisicos = platform.processor(), None
    if os.name == "nt":
        j = _powershell("Get-CimInstance Win32_Processor | Select-Object -First 1 Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed | ConvertTo-Json -Compress")
        if j:
            d = json.loads(j)
            cpu, fisicos = d["Name"].strip(), int(d["NumberOfCores"])
    try:
        import psutil

        ram = psutil.virtual_memory().total
        fisicos = fisicos or psutil.cpu_count(logical=False)
    except ImportError:
        ram = None
    blas = [f"{b.get('internal_api')} {b.get('version')} ({b.get('num_threads')} hilos disponibles)" for b in __import__("threadpoolctl").threadpool_info()]
    return {"cpu": cpu, "nucleos_fisicos": fisicos, "hilos_logicos": os.cpu_count(), "ram_gib": round(ram / 2**30, 1) if ram else None, "sistema_operativo": platform.platform(),
            "python": sys.version.split()[0], "numpy": numpy.__version__, "pandas": pandas.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__, "bibliotecas_de_hilos": "; ".join(blas), "hilos_en_la_medicion": 1}


def fijar_afinidad(cpu: int | None) -> str:
    """Fija el proceso a una CPU lógica (reduce la variabilidad en CPU híbridas). Devuelve una descripción; no falla si no se puede."""
    if cpu is None:
        return "sin afinidad fija"
    try:
        import psutil

        psutil.Process().cpu_affinity([cpu])
        return f"afinidad fija a la CPU lógica {cpu}"
    except Exception as e:  # noqa: BLE001
        return f"no se pudo fijar la afinidad ({e.__class__.__name__})"


# ---------- estadística de las mediciones ----------
def resumen_tiempos(valores) -> dict:
    """Mediana, cuartiles y rango intercuartílico (RIQ) de una lista de mediciones."""
    v = np.asarray(valores, dtype=float)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return {"mediana": float(med), "q1": float(q1), "q3": float(q3), "riq": float(q3 - q1), "n": int(len(v))}


def exponente_log_log(n, t) -> dict:
    """Pendiente de log t contra log n (mínimos cuadrados), su error estándar y R². Con pocos puntos es aproximado."""
    x, y = np.log(np.asarray(n, dtype=float)), np.log(np.asarray(t, dtype=float))
    if len(x) < 3:
        raise ValueError("se necesitan al menos 3 puntos para el exponente")
    pend, orig = np.polyfit(x, y, 1)
    res = y - (pend * x + orig)
    se = float(np.sqrt((res @ res) / (len(x) - 2) / ((x - x.mean()) @ (x - x.mean()))))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"exponente": float(pend), "error_estandar": se, "r2": float(1 - (res @ res) / ss_tot) if ss_tot > 0 else float("nan")}


def exponente_teorico_n_log_n(n_min: float, n_max: float) -> float:
    """Exponente local de n·log n entre `n_min` y `n_max` (para comparar con la pendiente medida en ese rango)."""
    return float(np.log((n_max * np.log(n_max)) / (n_min * np.log(n_min))) / np.log(n_max / n_min))


# ---------- submuestras por IES ----------
def orden_ies(df: pd.DataFrame, semilla: int) -> np.ndarray:
    """Las IES en un orden aleatorio fijo: la submuestra de fracción f son las primeras ceil(f · n_ies), así que las submuestras están anidadas."""
    ies = np.sort(df["cod_ies"].astype(str).unique())
    return ies[np.random.default_rng(semilla).permutation(len(ies))]


def submuestra_por_ies(df: pd.DataFrame, fraccion: float, semilla: int) -> pd.DataFrame:
    """Todas las filas de una fracción de las IES (nunca se parte una IES)."""
    ies = orden_ies(df, semilla)
    elegidas = set(ies[: max(1, int(np.ceil(fraccion * len(ies))))])
    return df[df["cod_ies"].astype(str).isin(elegidas)].reset_index(drop=True)


# ---------- líneas base con fit y predict separados (mismo resultado que las usadas en los experimentos) ----------
class NaiveT1:
    """Ingenua de la Tarea 1: tasa de t-1 (`tasa_t1`) o, si falta, la mediana de la tasa del entrenamiento."""

    def fit(self, tasa_t1: np.ndarray, y: np.ndarray):
        self.mediana_ = float(np.median(y))
        return self

    def predict(self, tasa_t1: np.ndarray) -> np.ndarray:
        return np.where(np.isnan(tasa_t1), self.mediana_, tasa_t1)


class MedianaGrupo:
    """Mediana de la tasa por carácter × sector del entrenamiento; si el grupo no está, la mediana global."""

    def fit(self, grupos: pd.Series, y: np.ndarray):
        self.medianas_ = pd.Series(y).groupby(grupos.to_numpy()).median()
        self.global_ = float(np.median(y))
        return self

    def predict(self, grupos: pd.Series) -> np.ndarray:
        return grupos.map(self.medianas_).fillna(self.global_).to_numpy(dtype=float)


# ---------- componentes ----------
def teoria() -> dict[str, dict]:
    """Complejidad teórica de cada componente: notación O del ajuste y de la predicción por fila, memoria y rango del exponente en n (p, T, d, k, E, h, I fijos).

    n filas, p variables, T árboles, d profundidad, m variables que se prueban por división (`max_features`), k componentes, E épocas, h neuronas, I iteraciones de Lloyd,
    K_c candidatos de k, c clusters. Son complejidades estándar de las implementaciones usadas (sklearn, xgboost con histogramas), no una derivación propia.
    """
    return {
        "T1 naive (tasa de t-1)": {"ajuste": "O(n)", "prediccion": "O(1)", "memoria": "O(n)", "exp": (1.0, 1.0)},
        "T2 mediana carácter × sector": {"ajuste": "O(n log n)", "prediccion": "O(1)", "memoria": "O(n)", "exp": (1.0, 1.2)},
        "ridge": {"ajuste": "O(n p² + p³)", "prediccion": "O(p)", "memoria": "O(n p + p²)", "exp": (1.0, 1.0)},
        "random_forest": {"ajuste": "O(T · m · n log n)", "prediccion": "O(T · d)", "memoria": "O(n p + T · nodos)", "exp": (1.0, 1.3)},
        "xgboost": {"ajuste": "O(T · d · n · p) (histogramas)", "prediccion": "O(T · d)", "memoria": "O(n p + T · 2^d)", "exp": (0.9, 1.1)},
        "Imputación mediana": {"ajuste": "O(n p)", "prediccion": "O(n p)", "memoria": "O(n p)", "exp": (1.0, 1.0)},
        "Imputación KNN (k=5)": {"ajuste": "O(n² p) al imputar el entrenamiento", "prediccion": "O(n_nuevas · n · p)", "memoria": "O(n p) (distancias por bloques)", "exp": (1.8, 2.0)},
        "Información mutua (k=15)": {"ajuste": "O(p · n log n)", "prediccion": "O(n k)", "memoria": "O(n p)", "exp": (1.0, 1.3)},
        "PCA (90 % varianza)": {"ajuste": "O(n p² + p³)", "prediccion": "O(n p k)", "memoria": "O(n p + p²)", "exp": (1.0, 1.0)},
        "KernelPCA (RBF)": {"ajuste": "O(n² p + n³) (o O(k n²) con eigensolver parcial)", "prediccion": "O(n_nuevas · n · (p + k))", "memoria": "O(n²)", "exp": (2.0, 3.0)},
        "Autoencoder (16-8-16)": {"ajuste": "O(E · n · p · h)", "prediccion": "O(p · h) por fila", "memoria": "O(n p + p h)", "exp": (1.0, 1.0)},
        "K-means (k=6, n_init=10)": {"ajuste": "O(n_init · I · n · c · p)", "prediccion": "O(c · p) por fila", "memoria": "O(n p + c p)", "exp": (1.0, 1.0)},
        "K-means + silueta (k=3–8)": {"ajuste": "O(K_c · (n_init · I · n · c · p + n² p))", "prediccion": "O(c · p) por fila", "memoria": "O(n p) (silueta por bloques)", "exp": (1.5, 2.0)},
    }


def _familia(nombre: str) -> str:
    for f in ("random_forest", "xgboost", "ridge"):
        if f in nombre:
            return f
    return nombre


def _teoria_de(nombre: str) -> dict:
    t = teoria()
    return t[nombre] if nombre in t else t[_familia(nombre)]


def componentes(cfg: dict, ajuste: pd.DataFrame | None = None) -> list[dict]:
    """Lista de componentes medibles: `nombre`, `tipo` (linea_base | modelo | preproceso), `tarea` y `construir(d)` → (`ajustar`, `predecir`).

    `ajustar()` hace un ajuste desde cero y devuelve el objeto ajustado; `predecir(obj)` predice/transforma todas las filas de `d`. La preparación de los datos queda
    dentro de `construir`, fuera de lo que se cronometra.
    """
    pc = parametros_complejidad(cfg)
    p_exp, p_t2 = parametros_experimentos(cfg), t2.parametros_tarea2(cfg)
    ajuste = pd.read_csv(cfg["modelado"]["sesgo_sensibilidad"]["ajuste"]) if ajuste is None else ajuste
    hp = ef.hiperparametros_finales(ajuste)
    specs = aj.configuraciones_ajuste(cfg)
    comps: list[dict] = []

    def base_naive(d):
        y, t1 = d["tasa_desercion"].to_numpy(float), d["tasa_t1"].to_numpy(float)
        return (lambda: NaiveT1().fit(t1, y)), (lambda obj: obj.predict(t1))

    def base_grupo(d):
        y, g = d["tasa_desercion"].to_numpy(float), d["caracter"].astype(str) + "|" + d["sector"].astype(str)
        return (lambda: MedianaGrupo().fit(g, y)), (lambda obj: obj.predict(g))

    comps += [{"nombre": "T1 naive (tasa de t-1)", "tipo": "linea_base", "tarea": "T1", "construir": base_naive},
              {"nombre": "T2 mediana carácter × sector", "tipo": "linea_base", "tarea": "T2", "construir": base_grupo}]

    def modelo(m):
        def construir(d):
            X, num, cat = aj.preparar(m["spec"], cfg, d)
            y = d["tasa_desercion"].astype(float).to_numpy()

            def ajustar():
                est = ef.construir_final(m, num, cat, p_exp, p_t2)
                est.fit(X, y)
                return est
            return ajustar, (lambda est: est.predict(X))
        return construir

    def nombre_modelo(s, variante):
        if s["tarea"] == "T1":
            return s["nombre"]
        sufijo = "con clusters" if s["clusters"] else "sin clusters"
        red = f" {s['reduccion']}" if s["reduccion"] != "ninguna" else ""
        return f"T2 {s['modelo']}{red} {sufijo} ({variante})"

    for s in specs:
        m = {"nombre": s["nombre"], "tarea": s["tarea"], "spec": s, "variante": "ajustado", "hiperparametros": hp[s["nombre"]]}
        comps.append({"nombre": nombre_modelo(s, "ajustado"), "tipo": "modelo", "tarea": s["tarea"], "construir": modelo(m)})
        if s["tarea"] == "T2" and s["modelo"] == "ridge":  # el ridge de la Tarea 2 también por defecto (RidgeCV)
            d_ = {**m, "variante": "defecto", "hiperparametros": {}}
            comps.append({"nombre": nombre_modelo(s, "defecto"), "tipo": "modelo", "tarea": s["tarea"], "construir": modelo(d_)})

    # --- métodos de preprocesamiento (contexto de la Tarea 2) ---
    def contexto(d):
        _, X, y, num, cat = t2.preparar_datos(cfg, d)
        return X, y.to_numpy(float), num, cat

    def transformer(crear, entrada):
        """`entrada(d)` → (datos que recibe el transformer, y). `crear()` → un transformer nuevo con `fit_transform` y `transform`."""
        def construir(d):
            A, y = entrada(d)

            def ajustar():
                t = crear()
                t.fit_transform(A, y)
                return t
            return ajustar, (lambda t: t.transform(A))
        return construir

    def en_numericas(d):
        X, y, num, _ = contexto(d)
        return X[num], y

    def en_z(d):
        X, y, num, cat = contexto(d)
        return construir_preprocesador(num, cat, "mediana", p_exp["k_vecinos"]).fit_transform(X), y

    def en_x(d):
        X, y, _, _ = contexto(d)
        return X, y

    def clusters(d):
        X, y, num, cat = contexto(d)
        k = p_t2["clusters"]
        return (lambda: AgregarClusters(num, cat, k["k_min"], k["k_max"], k["n_init"], p_exp["semilla"])), X

    class _KMeansFijo:
        """K-means con k fijo sobre Z; `transform` devuelve la etiqueta de cada fila."""

        def __init__(self):
            self.km = KMeans(n_clusters=pc["kmeans_k"], n_init=p_t2["clusters"]["n_init"], random_state=p_exp["semilla"])

        def fit_transform(self, Z, y=None):
            self.km.fit(Z)
            return self

        def transform(self, Z):
            return self.km.predict(Z)

    def construir_clusters(d):
        crear, X = clusters(d)

        def ajustar():
            c = crear()
            c.fit(X)
            return c
        return ajustar, (lambda c: c.transform(X))

    mi = partial(mutual_info_regression, random_state=p_exp["semilla"])
    preproceso = [
        ("Imputación mediana", transformer(lambda: SimpleImputer(strategy="median"), en_numericas)),
        (f"Imputación KNN (k={p_exp['k_vecinos']})", transformer(lambda: KNNImputer(n_neighbors=p_exp["k_vecinos"]), en_numericas)),
        (f"Información mutua (k={p_exp['k_info_mutua']})", transformer(lambda: SelectKBest(mi, k=p_exp["k_info_mutua"]), en_z)),
        ("PCA (90 % varianza)", transformer(lambda: PCA(n_components=p_exp["varianza_pca"], svd_solver="full", random_state=p_exp["semilla"]), en_z)),
        ("KernelPCA (RBF)", transformer(lambda: KernelPCAComoPCA(p_exp["varianza_pca"], p_t2["kernel_pca"]["gamma"], p_exp["semilla"]), en_z)),
        ("Autoencoder (16-8-16)", transformer(lambda: AutoencoderPequeno(**p_t2["autoencoder"], random_state=p_exp["semilla"]), en_z)),
        (f"K-means (k={pc['kmeans_k']}, n_init=10)", transformer(_KMeansFijo, en_z)),
    ]
    for nombre, construir in preproceso:
        comps.append({"nombre": nombre, "tipo": "preproceso", "tarea": "T2", "construir": construir})
    comps.append({"nombre": "K-means + silueta (k=3–8)", "tipo": "preproceso", "tarea": "T2", "construir": construir_clusters})
    return comps


# ---------- memoria ----------
def rss_pico(funcion, intervalo: float = 0.002) -> float:
    """Aumento máximo de la memoria residente (RSS) del proceso durante `funcion()`, en MiB, muestreado cada `intervalo` segundos desde otro hilo.

    A diferencia de `tracemalloc`, incluye la memoria nativa (árboles de sklearn, xgboost). Es aproximado: el muestreo puede perder picos más cortos que el intervalo y el RSS
    no baja de inmediato al liberar memoria. Si `psutil` no está disponible devuelve NaN."""
    try:
        import psutil
    except ImportError:
        return float("nan")
    import threading

    proceso = psutil.Process()
    base = proceso.memory_info().rss
    pico = [base]
    parar = threading.Event()

    def muestrear():
        while not parar.is_set():
            pico[0] = max(pico[0], proceso.memory_info().rss)
            parar.wait(intervalo)

    hilo = threading.Thread(target=muestrear, daemon=True)
    hilo.start()
    try:
        funcion()
    finally:
        parar.set()
        hilo.join()
        pico[0] = max(pico[0], proceso.memory_info().rss)
    return (pico[0] - base) / 2**20


# ---------- medición ----------
def medir(ajustar, predecir, n_filas: int, repeticiones: int = 5, calentamiento: int = 1, memoria: bool = False) -> dict:
    """Cronometra `ajustar()` y `predecir(obj)` con un solo hilo: calentamiento, `repeticiones` mediciones y (opcional) el pico de memoria en una ejecución aparte."""
    with threadpool_limits(limits=1):
        for _ in range(calentamiento):
            predecir(ajustar())
        aj_t, pr_t = [], []
        for _ in range(repeticiones):
            t0 = time.perf_counter()
            obj = ajustar()
            aj_t.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            predecir(obj)
            pr_t.append((time.perf_counter() - t0) * 1000.0 * (1000.0 / n_filas))  # milisegundos por 1.000 filas
        pico, pico_rss = float("nan"), float("nan")
        if memoria:
            tracemalloc.start()
            try:
                ajustar()
                pico = tracemalloc.get_traced_memory()[1] / 2**20
            finally:
                tracemalloc.stop()
            pico_rss = rss_pico(ajustar)
    return {"ajuste": resumen_tiempos(aj_t), "prediccion_1000": resumen_tiempos(pr_t), "memoria_pico_mib": pico, "memoria_rss_mib": pico_rss, "ajuste_crudo": aj_t,
            "prediccion_1000_crudo": pr_t}


def medir_componente(comp: dict, df: pd.DataFrame, pc: dict, fracciones: list[float] | None = None) -> list[dict]:
    """Mide un componente en cada fracción de IES (memoria solo con todas las filas). Una fila por fracción."""
    filas = []
    for f in fracciones or pc["fracciones"]:
        d = submuestra_por_ies(df, f, pc["semilla_submuestra"]) if f < 1 else df.reset_index(drop=True)
        ajustar, predecir = comp["construir"](d)
        r = medir(ajustar, predecir, len(d), pc["repeticiones"], pc["calentamiento"], memoria=f >= 1)
        filas.append({"componente": comp["nombre"], "tipo": comp["tipo"], "tarea": comp["tarea"], "fraccion_ies": f, "n_filas": len(d), "n_ies": int(d["cod_ies"].nunique()),
                      "ajuste_mediana_s": r["ajuste"]["mediana"], "ajuste_q1_s": r["ajuste"]["q1"], "ajuste_q3_s": r["ajuste"]["q3"], "ajuste_riq_s": r["ajuste"]["riq"],
                      "pred_1000_mediana_ms": r["prediccion_1000"]["mediana"], "pred_1000_q1_ms": r["prediccion_1000"]["q1"], "pred_1000_q3_ms": r["prediccion_1000"]["q3"],
                      "pred_1000_riq_ms": r["prediccion_1000"]["riq"], "memoria_pico_mib": r["memoria_pico_mib"], "memoria_rss_mib": r["memoria_rss_mib"], "repeticiones": pc["repeticiones"], "hilos": 1})
    return filas


def exponentes(escalado: pd.DataFrame) -> pd.DataFrame:
    """Exponente empírico de cada componente frente al teórico, con su error estándar y R²; el rango teórico y si el medido es consistente (±0,3 y 2 errores estándar)."""
    filas = []
    for nombre, g in escalado.groupby("componente", sort=False):
        g = g.sort_values("n_filas")
        e = exponente_log_log(g["n_filas"], g["ajuste_mediana_s"])
        lo, hi = _teoria_de(nombre)["exp"]
        if _familia(nombre) == "random_forest" or nombre == "T2 mediana carácter × sector" or nombre.startswith("Información mutua"):
            ref = exponente_teorico_n_log_n(g["n_filas"].min(), g["n_filas"].max())  # n log n en el rango medido
            lo, hi = min(lo, ref), max(hi, ref)
        tol = 0.3 + 2 * e["error_estandar"]
        estado = "consistente" if lo - tol <= e["exponente"] <= hi + tol else ("por encima del teórico" if e["exponente"] > hi + tol else "por debajo del teórico")
        filas.append({"componente": nombre, "tipo": g["tipo"].iloc[0], "tarea": g["tarea"].iloc[0], "n_min": int(g["n_filas"].min()), "n_max": int(g["n_filas"].max()), "t_min_s": float(g["ajuste_mediana_s"].iloc[0]),
                      "t_max_s": float(g["ajuste_mediana_s"].iloc[-1]), "exponente_empirico": e["exponente"], "error_estandar": e["error_estandar"], "r2": e["r2"],
                      "exponente_teorico_min": lo, "exponente_teorico_max": hi, "comparacion": estado})
    return pd.DataFrame(filas)
