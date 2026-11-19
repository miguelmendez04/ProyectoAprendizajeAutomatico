"""Interfaz de línea de comandos: `desercion` (o `python -m desercion_ies`).

  desercion etl [--saltar-extract]                       ETL: extract → transform → integrate → validate
  desercion eda                                          ejecuta los cuadernos de EDA (01 y 02; necesitan los datos crudos)
  desercion evaluar {v1,v2,tarea1,tarea2,ajuste,sesgo,complejidad}
  desercion reproducir [--etapa ...] [--rapido] [--reanudar]
  desercion verificar                                    checksums SHA-256 de los datos
  desercion entrenar                                     entrena los modelos finales y los guarda en models/
  desercion predecir --tarea {1,2} --entrada X.csv --salida Y.csv

Opciones globales: `-v/--verbose` (mensajes de depuración) y `-q/--quiet` (solo avisos y errores).

Las rutas de `configs/config.yaml` son relativas a la raíz del proyecto, así que el comando trabaja desde ahí aunque se lo invoque desde otra carpeta;
las rutas que se pasen como argumento (`--config`, `--salida`) se interpretan respecto del directorio desde el que se invocó.

Este módulo importa solo la biblioteca estándar: las etapas se cargan al usarlas, para que las variables de hilos de `evaluar complejidad` se fijen
antes de importar numpy.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
from pathlib import Path

from desercion_ies import __version__, paths

log = logging.getLogger("desercion_ies")

# subcomando de `evaluar` -> (módulo del runner, resumen)
EVALUAR = {
    "v1": ("desercion_ies.runners.v1", "grilla de 36 configuraciones (histórico)"),
    "v2": ("desercion_ies.runners.v2", "formulaciones con historia (histórico)"),
    "tarea1": (
        "desercion_ies.runners.tarea1",
        "protocolo v3: Tarea 1 con historia, GroupKFold repetido 5 × 5",
    ),
    "tarea2": ("desercion_ies.runners.tarea2", "Tarea 2: estimación sin historia"),
    "ajuste": (
        "desercion_ies.runners.ajuste",
        "ajuste anidado, particiones temporales y arranque en frío",
    ),
    "sesgo": ("desercion_ies.runners.sesgo", "sesgo medido, sensibilidad e interpretabilidad"),
    "complejidad": ("desercion_ies.runners.complejidad", "costo computacional medido con un hilo"),
}
HILOS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
ATRIBUTOS_RUTA = (
    "config",
    "salida",
    "ajuste",
    "entrada",
    "modelos_dir",
)  # rutas que puede pasar el usuario


def configurar_logging(verbose: int = 0, quiet: bool = False) -> None:
    """Un solo manejador a la salida estándar; el mensaje va sin adornos salvo en modo detallado."""
    nivel = logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
    formato = "%(asctime)s %(levelname)s %(name)s: %(message)s" if verbose else "%(message)s"
    for flujo in (sys.stdout, sys.stderr):
        if hasattr(flujo, "reconfigure"):
            flujo.reconfigure(
                errors="replace"
            )  # una consola que no sea UTF-8 no debe tumbar la corrida por un carácter
    manejador = logging.StreamHandler(sys.stdout)
    manejador.setFormatter(logging.Formatter(formato))
    raiz = logging.getLogger()
    raiz.handlers[:] = [manejador]
    raiz.setLevel(nivel)


def limitar_hilos() -> None:
    """Un solo hilo en BLAS/OpenMP; debe ejecutarse ANTES de importar numpy (mediciones de complejidad)."""
    for v in HILOS:
        os.environ[v] = "1"


def _extraer_globales(argv: list[str]) -> tuple[list[str], int, bool]:
    """Saca `-v/--verbose` y `-q/--quiet` de cualquier posición (los parsers de cada etapa no los conocen)."""
    resto, verbose, quiet = [], 0, False
    for a in argv:
        if a in ("-v", "--verbose"):
            verbose += 1
        elif a in ("-q", "--quiet"):
            quiet = True
        else:
            resto.append(a)
    return resto, verbose, quiet


def _a_absoluta(args: argparse.Namespace, origen: Path, defectos: dict | None = None) -> None:
    """Las rutas pasadas por el usuario son relativas a donde invocó el comando; las predeterminadas, a la raíz del proyecto."""
    defectos = defectos or {}
    for atributo in ATRIBUTOS_RUTA:
        valor = getattr(args, atributo, None)
        if (
            isinstance(valor, str)
            and valor != defectos.get(atributo)
            and not Path(valor).is_absolute()
        ):
            setattr(args, atributo, str((origen / valor).resolve()))


def _parser_principal() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="desercion",
        description="Deserción en educación superior en Colombia (IES × año): ETL, evaluación de modelos y análisis.",
        epilog=os.linesep.join(__doc__.splitlines()[2:15]),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--version", action="version", version=f"desercion-ies {__version__}")
    sub = ap.add_subparsers(dest="comando", required=True, metavar="comando")
    # add_help=False: «-h» después del subcomando le llega al parser de la etapa, que conoce sus opciones
    sub.add_parser(
        "etl", add_help=False, help="ETL completo (extract → transform → integrate → validate)"
    )
    sub.add_parser(
        "eda", add_help=False, help="ejecuta los cuadernos de EDA (necesitan los datos crudos)"
    )
    ev = sub.add_parser(
        "evaluar", add_help=False, help="evaluación de una etapa: " + ", ".join(EVALUAR)
    )
    ev.add_argument("etapa", choices=list(EVALUAR), nargs="?")
    sub.add_parser(
        "reproducir",
        add_help=False,
        help="reproduce el proyecto por etapas (--rapido para verificar en < 10 min)",
    )
    sub.add_parser("verificar", add_help=False, help="verifica los SHA-256 de los datos")
    sub.add_parser(
        "entrenar",
        add_help=False,
        help="entrena los modelos finales con todo el dataset y los guarda en models/",
    )
    sub.add_parser(
        "predecir",
        add_help=False,
        help="predice la tasa de deserción de un CSV con un modelo guardado",
    )
    return ap


def _modulo_y_parser(comando: str, etapa: str | None) -> tuple[object, argparse.ArgumentParser]:
    """Importa la etapa pedida y arma su parser (aquí, y solo aquí, se cargan las bibliotecas pesadas)."""
    if comando == "etl":
        mod, prog, desc = (
            importlib.import_module("desercion_ies.etl.pipeline"),
            "desercion etl",
            "ETL del dataset de deserción",
        )
    elif comando == "evaluar":
        nombre = EVALUAR[etapa][0]
        mod, prog, desc = (
            importlib.import_module(nombre),
            f"desercion evaluar {etapa}",
            EVALUAR[etapa][1],
        )
    elif comando == "reproducir":
        mod, prog, desc = (
            importlib.import_module("desercion_ies.reproduce"),
            "desercion reproducir",
            "Reproduce el proyecto por etapas",
        )
    elif comando in ("entrenar", "predecir"):
        mod, prog, desc = (
            importlib.import_module(f"desercion_ies.inference.{comando}"),
            f"desercion {comando}",
            "Modelos finales utilizables",
        )
    else:
        raise ValueError(comando)
    p = argparse.ArgumentParser(
        prog=prog,
        description=mod.__doc__ or desc,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mod.configurar(p)
    return mod, p


def _verificar(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="desercion verificar",
        description="Compara los SHA-256 de data/checksums.json con los archivos presentes.",
    )
    ap.add_argument(
        "--estricto",
        action="store_true",
        help="también falla si faltan los datos crudos (por defecto solo importan el dataset y los crudos presentes)",
    )
    a = ap.parse_args(argv)
    from desercion_ies import reproducibilidad as rp

    malo = False
    for grupo in ("crudos", "procesado"):
        for f in rp.verificar_checksums(grupo, paths.PROJECT_ROOT, paths.CHECKSUMS):
            log.info("%-9s %-8s %s", grupo, f["estado"], f["archivo"])
            malo |= f["estado"] == "distinto" or (
                f["estado"] == "ausente" and (grupo == "procesado" or a.estricto)
            )
    log.info("verificación %s", "FALLÓ" if malo else "correcta")
    return 1 if malo else 0


def _eda(argv: list[str]) -> int:
    argparse.ArgumentParser(
        prog="desercion eda",
        description="Ejecuta notebooks/01_eda.ipynb y 02_sesgo.ipynb de punta a punta (necesitan data/raw y el ETL).",
    ).parse_args(argv)
    from desercion_ies import reproduce

    ctx = {"rapido": False, "salida": paths.PROJECT_ROOT / "reproduccion"}
    try:
        log.info("%s", reproduce.etapa_eda(ctx))
    except reproduce.ErrorEtapa as e:
        log.error("%s", e)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada de `desercion`; devuelve el código de salida."""
    origen = Path.cwd()
    argv, verbose, quiet = _extraer_globales(list(sys.argv[1:] if argv is None else argv))
    configurar_logging(verbose, quiet)
    ap = _parser_principal()
    args, resto = ap.parse_known_args(argv)
    if args.comando == "evaluar" and args.etapa is None:
        ap.parse_args(["-h"])  # `desercion evaluar` sin etapa: muestra la ayuda general
    os.chdir(paths.PROJECT_ROOT)
    if args.comando == "verificar":
        return _verificar(resto)
    if args.comando == "eda":
        return _eda(resto)
    if args.comando == "evaluar" and args.etapa == "complejidad":
        limitar_hilos()
    mod, parser = _modulo_y_parser(args.comando, getattr(args, "etapa", None))
    opciones = parser.parse_args(resto)
    _a_absoluta(opciones, origen, {a: parser.get_default(a) for a in ATRIBUTOS_RUTA})
    resultado = mod.ejecutar(opciones)
    return resultado if isinstance(resultado, int) else 0
