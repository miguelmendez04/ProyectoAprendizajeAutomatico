"""Utilidades de reproducibilidad: checksums SHA-256, configuración del modo rápido y comparación contra los CSV publicados.

No toca la lógica de modelado: el modo rápido solo cambia la *configuración* (menos semillas, grilla reducida) y redirige todas las salidas a otra carpeta,
de modo que los resultados publicados en `reports/` no se sobrescriben.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pandas as pd

CHECKSUMS = Path("data/checksums.json")


# ---------- checksums ----------
def sha256_archivo(ruta: str | Path, bloque: int = 1 << 20) -> str:
    """SHA-256 de un archivo, leído por bloques (los crudos pesan cientos de MB)."""
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        while chunk := f.read(bloque):
            h.update(chunk)
    return h.hexdigest()


def leer_checksums(ruta: str | Path = CHECKSUMS) -> dict:
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


def verificar_checksums(grupo: str, raiz: str | Path = ".", ruta: str | Path = CHECKSUMS) -> list[dict]:
    """Compara los archivos de un grupo (`crudos` o `procesado`) con el manifiesto.

    Devuelve una fila por archivo con `estado`: `ok`, `ausente` o `distinto` (y los hashes esperado y observado).
    """
    manifiesto = leer_checksums(ruta)[grupo]
    filas = []
    for rel, info in manifiesto.items():
        p = Path(raiz) / rel
        if not p.exists():
            filas.append({"archivo": rel, "estado": "ausente", "esperado": info["sha256"], "observado": None})
            continue
        obs = sha256_archivo(p)
        filas.append({"archivo": rel, "estado": "ok" if obs == info["sha256"] else "distinto", "esperado": info["sha256"], "observado": obs})
    return filas


def resumen_verificacion(filas: list[dict]) -> tuple[bool, str]:
    """(todo_ok, texto) para imprimir al inicio de una etapa."""
    malos = [f for f in filas if f["estado"] != "ok"]
    if not malos:
        return True, f"checksums verificados ({len(filas)} archivos)"
    return False, "; ".join(f"{f['archivo']}: {f['estado']}" for f in malos)


def generar_checksums(raiz: str | Path, crudos: dict[str, dict], procesado: list[str]) -> dict:
    """Arma el manifiesto. `crudos` mapea ruta relativa -> metadatos (fuente, url, fecha_descarga); se agregan tamaño y SHA-256."""
    raiz = Path(raiz)
    salida = {"crudos": {}, "procesado": {}}
    for rel, meta in crudos.items():
        p = raiz / rel
        salida["crudos"][rel] = {**meta, "bytes": p.stat().st_size, "sha256": sha256_archivo(p)}
    for rel in procesado:
        p = raiz / rel
        salida["procesado"][rel] = {"bytes": p.stat().st_size, "sha256": sha256_archivo(p)}
    return salida


# ---------- modo rápido ----------
def _redirigir(x, salida: str):
    """Reemplaza, en todo el árbol de configuración, las rutas `reports/...` por `<salida>/reports/...` (entradas y salidas por igual)."""
    if isinstance(x, str) and x.startswith("reports/"):
        return f"{salida}/{x}"
    if isinstance(x, dict):
        return {k: _redirigir(v, salida) for k, v in x.items()}
    if isinstance(x, list):
        return [_redirigir(v, salida) for v in x]
    return x


# configuraciones del protocolo v3 que se conservan en el modo rápido (subconjunto del conjunto publicado; mismos nombres)
V3_RAPIDO = ["A contexto | ridge logit info_mutua", "B historia+contexto | xgboost logit", "C corrección solo historia | ridge", "C corrección solo historia | ridge [rezagada]"]


def config_rapida(cfg: dict, salida: str) -> dict:
    """Copia de `cfg` con menos repeticiones y grillas reducidas, y todas las salidas en `<salida>/reports/`.

    Los pliegues (n_splits) y las semillas que se conservan son las MISMAS de la configuración completa, por eso las cifras por pliegue de las
    configuraciones comunes coinciden exactamente con las publicadas. Lo que cambia: menos semillas, grillas más chicas, `n_iter` de la búsqueda
    interna, réplicas del bootstrap y repeticiones de la medición de complejidad.
    """
    c = _redirigir(copy.deepcopy(cfg), salida)
    m = c["modelado"]  # cada sección se reduce solo si existe (las configuraciones de prueba son parciales)

    def reducir(seccion: str, **cambios):
        if seccion in m:
            m[seccion].update(cambios)

    reducir("experimentos", objetivo=["ninguna"], imputacion=["mediana"], seleccion=["ninguna", "info_mutua"], modelo=["ridge", "xgboost"])
    reducir("formulaciones", incluir=["A_contexto", "C_correccion_solo_historia"], objetivo=["ninguna"], seleccion=["ninguna"], modelo=["ridge", "xgboost"])
    if "protocolo_v3" in m:
        v3 = m["protocolo_v3"]
        v3["semillas"] = v3["semillas"][:2]
        v3["configuraciones"] = [x for x in v3["configuraciones"] if x["nombre"] in V3_RAPIDO]
    if "tarea2" in m:
        t2 = m["tarea2"]
        t2["semillas_confirmacion"] = t2["semillas_confirmacion"][:1]
        if "grilla" in t2:
            t2["grilla"].update(objetivo=["logit"], imputacion=["mediana"])
    if "ajuste_temporal" in m:
        a = m["ajuste_temporal"]
        a.update(semillas_externas=a["semillas_externas"][:1], n_iter=3)
        if "bootstrap" in a:
            a["bootstrap"] = {**a["bootstrap"], "replicas": 100}
    if "sesgo_sensibilidad" in m:
        s = m["sesgo_sensibilidad"]
        s["semillas"] = s["semillas"][:1]
        s["bootstrap"] = {**s["bootstrap"], "replicas": 100}
        s["permutacion"] = {**s["permutacion"], "n_repeticiones": 2}
    reducir("complejidad", repeticiones=2, fracciones=[0.5, 0.75, 1.0])
    return c


# ---------- comparación con lo publicado ----------
CLAVES = ["configuracion", "tarea", "variante", "tipo", "repeticion", "semilla", "pliegue", "escenario", "etapa"]
METRICAS = ["mae", "rmse", "r2", "n_entrenamiento", "n_prueba"]


def comparar_csv(nuevo: str | Path, publicado: str | Path, filtro: dict | None = None) -> dict:
    """Compara fila a fila (por clave) las métricas de dos CSV, con igualdad EXACTA.

    Solo se comparan las filas con la misma clave en ambos (en el modo rápido el nuevo es un subconjunto del publicado).
    `filtro` ({columna: valor}) restringe a una variante (p. ej. `{"variante": "defecto"}`).
    """
    a, b = pd.read_csv(nuevo), pd.read_csv(publicado)
    if filtro:
        for k, v in filtro.items():
            a, b = a[a[k] == v], b[b[k] == v]
    claves = [c for c in CLAVES if c in a.columns and c in b.columns]
    metricas = [m for m in METRICAS if m in a.columns and m in b.columns]
    j = a[claves + metricas].merge(b[claves + metricas], on=claves, suffixes=("_nuevo", "_pub"))
    out = {"filas_nuevo": len(a), "filas_comunes": len(j), "metricas": metricas, "iguales": 0, "diferencia_max": 0.0}
    if j.empty:
        return out
    dif = pd.concat([(j[f"{m}_nuevo"] - j[f"{m}_pub"]).abs() for m in metricas], axis=1)
    fila_igual = (dif.fillna(0) == 0).all(axis=1)
    out["iguales"] = int(fila_igual.sum())
    out["diferencia_max"] = float(dif.max().max())
    return out
