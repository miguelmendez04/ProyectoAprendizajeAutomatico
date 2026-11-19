"""Genera requirements.txt a partir de pyproject.toml (dependencias + extra [dev]); `make requirements`."""
import tomllib
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
CABECERA = (
    "# GENERADO por tools/generar_requirements.py a partir de pyproject.toml: no editar a mano.\n"
    "# Versiones exactas con las que se generaron los resultados (Python 3.14.6, Windows 11, 16 hilos).\n"
    "# Instalación sin empaquetar: pip install -r requirements.txt; con el comando `desercion`: pip install -e \".[dev]\"\n"
)


def contenido() -> str:
    proyecto = tomllib.loads((RAIZ / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    deps = [*proyecto["dependencies"], *proyecto["optional-dependencies"]["dev"]]
    return CABECERA + "\n".join(deps) + "\n"


if __name__ == "__main__":
    (RAIZ / "requirements.txt").write_text(contenido(), encoding="utf-8", newline="\n")
    print("requirements.txt actualizado")
