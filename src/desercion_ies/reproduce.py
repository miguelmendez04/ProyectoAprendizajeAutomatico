"""Punto de entrada único para reproducir el proyecto de deserción en educación superior (Colombia).

Uso:
  desercion reproducir --etapa todo                  # corrida completa (≈ 1,5 h; ver README)
  desercion reproducir --etapa todo --rapido         # verificación en < 10 min; no toca reports/ (escribe en reproduccion/)
  desercion reproducir --etapa tarea2 --reanudar     # una etapa; reutiliza lo ya calculado donde el script lo permite
  desercion reproducir --solo-verificar              # solo los checksums de los datos

Etapas: etl, eda, tarea1, tarea2, ajuste, sesgo, complejidad, todo. Cada una llama a los comandos existentes (`desercion etl`, `desercion evaluar v1`,
desercion evaluar v2, desercion evaluar tarea1, desercion evaluar tarea2, desercion evaluar ajuste, desercion evaluar sesgo, desercion evaluar complejidad) sin cambiar su
lógica ni sus semillas. El modo --rapido solo cambia la configuración (menos semillas y grillas más chicas) y redirige las salidas.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

from desercion_ies import paths
from desercion_ies import reproducibilidad as rp

ETAPAS = ["etl", "eda", "tarea1", "tarea2", "ajuste", "sesgo", "complejidad"]
RES = "reports/tables"
# etapa -> lista de pasos (subcomando de `desercion evaluar`, argumentos extra, acepta --reanudar, salidas que prueban que ya se hizo)
PASOS = {
    "tarea1": [
        ("v1", [], False, [f"{RES}/experimentos_v1.csv"]),
        ("v2", [], False, [f"{RES}/experimentos_v2.csv"]),
        ("tarea1", [], False, [f"{RES}/experimentos_v3.csv"]),
    ],
    "tarea2": [("tarea2", [], False, [f"{RES}/experimentos_tarea2_confirmacion.csv"])],
    "ajuste": [("ajuste", [], True, [])],
    "sesgo": [("sesgo", [], True, [])],
    "complejidad": [("complejidad", [], True, [])],
}
NOTEBOOKS = {
    "tarea2": "construir_tarea2.py",
    "ajuste": "construir_ajuste_temporal.py",
    "sesgo": "construir_sesgo_interpretabilidad.py",
    "complejidad": "construir_complejidad.py",
}  # en notebooks/construir/
# archivos publicados que algún informe escribe en una ruta fija (no configurable): en modo rápido se restauran al terminar cada etapa
PROTEGIDOS = ["reports/05_sesgo_dataset.md"]
# archivos de entrada que una etapa necesita de otra anterior; en modo rápido, si faltan, se copian de lo publicado
PRERREQUISITOS = {
    "ajuste": [f"{RES}/experimentos_tarea2_confirmacion_resumen.csv"],
    "sesgo": [f"{RES}/ajuste.csv", f"{RES}/ajuste_resumen.csv"],
    "complejidad": [f"{RES}/ajuste.csv", f"{RES}/ajuste_resumen.csv"],
}


log = logging.getLogger(__name__)


class ErrorEtapa(RuntimeError):
    pass


def _entorno() -> dict:
    """Entorno de los subprocesos; `src/` va en PYTHONPATH para que `python -m desercion_ies` funcione también sin `pip install -e`."""
    previo = os.environ.get("PYTHONPATH")
    ruta = os.pathsep.join([str(paths.PROJECT_ROOT / "src"), *([previo] if previo else [])])
    return {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "PYTHONWARNINGS": "ignore",
        "PYTHONPATH": ruta,
    }


def correr(cmd: list[str], log: Path) -> None:
    """Ejecuta un comando, guarda toda su salida en `log` y muestra la última línea; falla con el final del log si termina con error."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w", encoding="utf-8") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=_entorno())
    if r.returncode != 0:
        cola = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-15:])
        raise ErrorEtapa(f"falló `{' '.join(cmd)}` (código {r.returncode}); ver {log}\n{cola}")


def verificar(grupo: str, raiz: str = ".") -> tuple[bool, str]:
    return rp.resumen_verificacion(rp.verificar_checksums(grupo, raiz))


# ---------- cifras clave por etapa (se leen de los CSV de la corrida) ----------
def _leer(ruta: Path) -> pd.DataFrame:
    return pd.read_csv(ruta)


def cifras(etapa: str, base: Path, extra: str = "") -> str:
    r = base / RES
    try:
        if etapa == "etl":
            d = pd.read_csv(base / "data/processed/dataset_ies.csv", dtype={"cod_ies": str})
            return f"{len(d)} filas, {d['cod_ies'].nunique()} IES{extra}"
        if etapa == "eda":
            return extra
        if etapa == "tarea1":
            v3 = _leer(r / "experimentos_v3_resumen.csv")
            lb = v3[v3["tipo"] == "linea_base"]
            mod = v3[v3["tipo"] == "modelo"].sort_values("mae_media").iloc[0]
            ing = lb[lb["configuracion"].str.contains("ingenua", case=False)].iloc[0]
            return f"v3: ingenua MAE {ing['mae_media']:.4f}; mejor {mod['configuracion']} MAE {mod['mae_media']:.4f}"
        if etapa == "tarea2":
            c = _leer(r / "experimentos_tarea2_confirmacion_resumen.csv")
            b = c[c["configuracion"] == "base_mediana_grupo"].iloc[0]
            m = c[c["tipo"] == "modelo"].sort_values("mae_media").iloc[0]
            return f"mediana carácter×sector MAE {b['mae_media']:.4f}; mejor modelo {m['mae_media']:.4f} ({m['configuracion']})"
        if etapa == "ajuste":
            a = _leer(r / "ajuste_resumen.csv")
            t1 = (
                a[(a["tarea"] == "T1") & (a["variante"] == "ajustado")]
                .sort_values("mae_media")
                .iloc[0]
            )
            t2 = (
                a[(a["tarea"] == "T2") & (a["variante"] == "ajustado")]
                .sort_values("mae_media")
                .iloc[0]
            )
            return (
                f"anidado: mejor T1 MAE {t1['mae_media']:.4f}, mejor T2 MAE {t2['mae_media']:.4f}"
            )
        if etapa == "sesgo":
            s = _leer(r / "sensibilidad.csv")
            return f"{len(s)} filas de sensibilidad en {s['escenario'].nunique()} escenarios; {int(s['cambia_conclusion'].sum())} cambian la conclusión"
        if etapa == "complejidad":
            c = _leer(r / "complejidad.csv")
            return f"{len(c)} componentes medidos; ajuste mediano más lento {c['ajuste_mediana_s'].max() * 1000:.0f} ms"
    except Exception as e:  # el resumen nunca debe tumbar la corrida
        return f"(sin cifras: {type(e).__name__}: {e})"
    return ""


# ---------- etapas ----------
def etapa_etl(ctx: dict) -> str:
    ok, txt = verificar("crudos")
    if not ok:
        raise ErrorEtapa(
            f"faltan o difieren los datos crudos: {txt}. Ver docs/DATOS.md; sin ellos use el dataset procesado (--etapa tarea1, etc.)."
        )
    cfg = yaml.safe_load(Path(ctx["config"]).read_text(encoding="utf-8"))
    if ctx[
        "rapido"
    ]:  # el ETL rápido escribe su propio dataset para compararlo con el publicado, sin tocar data/processed
        sal = ctx["salida"]
        for k in ("interim", "processed", "reports"):
            cfg["rutas"][k] = f"{sal}/{cfg['rutas'][k]}"
        cfg_path = sal / "config_etl.yaml"
        cfg_path.write_text(
            yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
    else:
        cfg_path = Path(ctx["config"])
    correr(
        [
            sys.executable,
            "-m",
            "desercion_ies",
            "etl",
            "--config",
            str(cfg_path),
            "--saltar-extract",
        ],
        ctx["salida"] / "logs" / "etl.log",
    )
    destino = (
        ctx["salida"] / "data/processed/dataset_ies.csv"
        if ctx["rapido"]
        else Path("data/processed/dataset_ies.csv")
    )
    esperado = rp.leer_checksums()["procesado"]["data/processed/dataset_ies.csv"]["sha256"]
    igual = rp.sha256_archivo(destino) == esperado
    return f"; dataset {'idéntico' if igual else 'DISTINTO'} al publicado (SHA-256)"


def etapa_eda(ctx: dict) -> str:
    """Ejecuta los cuadernos de EDA (01 y 02) de punta a punta. En modo rápido se guardan copias en la carpeta de salida."""
    nbs = ["notebooks/01_eda.ipynb", "notebooks/02_sesgo.ipynb"]
    for nb in nbs:
        cmd = [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--ExecutePreprocessor.timeout=1800",
            nb,
        ]
        cmd += (
            ["--output-dir", str(ctx["salida"] / "notebooks")] if ctx["rapido"] else ["--inplace"]
        )
        correr(cmd, ctx["salida"] / "logs" / f"eda_{Path(nb).stem}.log")
    return f"{len(nbs)} cuadernos ejecutados sin errores"


def etapa_modelado(etapa: str, ctx: dict) -> str:
    ok, txt = verificar("procesado")
    if not ok:
        raise ErrorEtapa(
            f"el dataset procesado no coincide con el publicado: {txt}. Use `desercion reproducir --etapa etl` o restaure data/processed/dataset_ies.csv."
        )
    base = ctx["salida"]
    if ctx["rapido"]:
        for rel in PRERREQUISITOS.get(etapa, []):
            nuevo = base / rel
            if not nuevo.exists():
                nuevo.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(rel, nuevo)
                log.info(f"    (rápido) falta {rel} de una etapa anterior: se copia del publicado")
    copias = (
        {r: Path(r).read_bytes() for r in PROTEGIDOS if Path(r).exists()} if ctx["rapido"] else {}
    )
    try:
        _pasos(etapa, ctx, base)
    finally:
        for r, contenido in copias.items():
            Path(r).write_bytes(contenido)
    return ""


def _pasos(etapa: str, ctx: dict, base: Path) -> None:
    for script, extra, acepta, salidas in PASOS[etapa]:
        if ctx["reanudar"] and salidas and all((base / s).exists() for s in salidas):
            log.info(f"    evaluar {script}: salidas ya presentes, se omite (--reanudar)")
            continue
        cmd = [
            sys.executable,
            "-m",
            "desercion_ies",
            "evaluar",
            script,
            "--config",
            str(ctx["config"]),
            *extra,
        ]
        if ctx["reanudar"] and acepta:
            cmd.append("--reanudar")
        t = time.perf_counter()
        correr(cmd, base / "logs" / f"evaluar_{script}.log")
        log.info(f"    evaluar {script}: {time.perf_counter() - t:.0f} s")
    if etapa in NOTEBOOKS and not ctx["rapido"] and not ctx["sin_notebooks"]:
        correr(
            [sys.executable, f"notebooks/construir/{NOTEBOOKS[etapa]}"],
            base / "logs" / f"{Path(NOTEBOOKS[etapa]).stem}.log",
        )


def comparar_con_publicado(ctx: dict) -> list[dict]:
    """Compara, con igualdad exacta, las cifras por pliegue del modo rápido contra los CSV publicados (solo claves comunes)."""
    pares = [
        ("experimentos_v1", None, ""),
        ("experimentos_v2", {"formulacion": "A_contexto"}, ""),
        (
            "experimentos_v2",
            {"formulacion": "C_correccion_solo_historia"},
            "diferencia esperada: la v2 publicada usa la historia desde 2015 (ver nota en resultados_v2.md)",
        ),
        ("experimentos_v3", None, ""),
        ("experimentos_tarea2_cribado", None, ""),
        ("experimentos_tarea2_confirmacion", None, ""),
        ("ajuste", {"variante": "defecto"}, ""),
        (
            "ajuste",
            {"variante": "ajustado"},
            "diferencia esperada: la búsqueda interna usa n_iter reducido",
        ),
        ("arranque_frio_municipio", None, ""),
        (
            "sensibilidad_pliegues",
            None,
            "diferencia esperada solo en los modelos «ajustado» (sus hiperparámetros salen del ajuste rápido); líneas base y modelos por defecto coinciden",
        ),
    ]
    filas = []
    for nombre, filtro, nota in pares:
        nuevo, pub = ctx["salida"] / RES / f"{nombre}.csv", Path(RES) / f"{nombre}.csv"
        if not nuevo.exists() or not pub.exists():
            continue
        c = rp.comparar_csv(nuevo, pub, filtro)
        filas.append(
            {
                "archivo": nombre + (f" [{next(iter(filtro.values()))}]" if filtro else ""),
                "nota": nota,
                **c,
            }
        )
    return filas


# ---------- principal ----------
def configurar(ap: argparse.ArgumentParser) -> None:
    """Opciones de `desercion reproducir`."""
    ap.add_argument("--etapa", choices=[*ETAPAS, "todo"], default="todo")
    ap.add_argument(
        "--rapido",
        action="store_true",
        help="pocas repeticiones y grilla reducida; escribe en --salida y no toca reports/",
    )
    ap.add_argument("--reanudar", action="store_true")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--salida", default="reproduccion", help="carpeta de salida del modo rápido")
    ap.add_argument(
        "--sin-notebooks",
        action="store_true",
        help="en la corrida completa, no regenera los cuadernos 03–06",
    )
    ap.add_argument(
        "--sin-verificar",
        action="store_true",
        help="no verifica los checksums al inicio de cada etapa",
    )
    ap.add_argument(
        "--solo-verificar", action="store_true", help="solo verifica los checksums y termina"
    )
    ap.add_argument(
        "--comparar",
        action="store_true",
        help="tras el modo rápido, compara con los CSV publicados (activo por defecto en --rapido)",
    )


def main(argv: list[str] | None = None) -> int:
    """Interfaz para pruebas y para `python -m`: lee `argv` y ejecuta."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    configurar(ap)
    return ejecutar(ap.parse_args(argv))


def ejecutar(a: argparse.Namespace) -> int:
    """Corre las etapas pedidas y devuelve el código de salida (0 si todo salió bien o se omitió)."""
    if a.solo_verificar:
        for g in ("crudos", "procesado"):
            filas = rp.verificar_checksums(g)
            for f in filas:
                log.info(f"{g:9s} {f['estado']:8s} {f['archivo']}")
        return 0

    ctx = {
        "rapido": a.rapido,
        "reanudar": a.reanudar,
        "salida": Path(a.salida) if a.rapido else Path("."),
        "sin_notebooks": a.sin_notebooks,
        "config": Path(a.config),
    }
    if a.rapido:
        ctx["salida"].mkdir(parents=True, exist_ok=True)
        cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
        ctx["config"] = ctx["salida"] / "config_rapida.yaml"
        ctx["config"].write_text(
            yaml.safe_dump(rp.config_rapida(cfg, a.salida), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    etapas = ETAPAS if a.etapa == "todo" else [a.etapa]
    resumen, t_total = [], time.perf_counter()
    for e in etapas:
        log.info(f"\n== Etapa {e} {'(rápido)' if a.rapido else ''}")
        t = time.perf_counter()
        try:
            if not a.sin_verificar and e != "etl":
                ok, txt = verificar("procesado")
                log.info(f"    {txt}")
            if (
                e in ("etl", "eda")
            ):  # ambas leen data/raw (el EDA, además, data/interim): sin los crudos se omiten en `todo` y fallan si se piden solas
                raw_ok = all(f["estado"] == "ok" for f in rp.verificar_checksums("crudos"))
                if a.etapa == "todo" and not raw_ok:
                    resumen.append(
                        (
                            e,
                            "omitida",
                            0.0,
                            "datos crudos no disponibles (se usa el dataset procesado verificado; ver docs/DATOS.md)",
                        )
                    )
                    log.info("    datos crudos no disponibles: se omite")
                    continue
                if e == "eda" and not raw_ok:
                    raise ErrorEtapa(
                        "el EDA (cuadernos 01 y 02) necesita los datos crudos y el resultado del ETL; ver docs/DATOS.md"
                    )
            if e == "etl":
                extra = etapa_etl(ctx)
            elif e == "eda":
                extra = etapa_eda(ctx)
            else:
                etapa_modelado(e, ctx)
                extra = ""
            resumen.append((e, "ok", time.perf_counter() - t, cifras(e, ctx["salida"], extra)))
        except ErrorEtapa as err:
            resumen.append((e, "ERROR", time.perf_counter() - t, str(err).splitlines()[0]))
            log.info(f"    ERROR: {err}")
            break

    log.info(
        "\n"
        + "=" * 100
        + "\nResumen"
        + (
            " (modo rápido: semillas y grillas reducidas, NO son las cifras del informe)"
            if a.rapido
            else ""
        )
    )
    for e, est, seg, txt in resumen:
        log.info(f"  {e:12s} {est:8s} {seg:7.0f} s  {txt}")
    log.info(f"  {'total':12s} {'':8s} {time.perf_counter() - t_total:7.0f} s")
    if a.rapido and any(r[1] == "ok" for r in resumen):
        comp = comparar_con_publicado(ctx)
        if comp:
            log.info(
                "\nComparación con los CSV publicados (filas con la misma clave; igualdad exacta o hasta 1e-9):"
            )
            for c in comp:
                log.info(
                    f"  {c['archivo']:42s} comunes {c['filas_comunes']:5d}  idénticas {c['iguales']:5d}  ≤1e-9 {c['casi_iguales']:5d}  dif. máx. {c['diferencia_max']:.2e}  {c['nota']}"
                )
    return 0 if all(r[1] in ("ok", "omitida") for r in resumen) else 1
