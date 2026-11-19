"""Convierte el resumen de fallos de un log de pytest en anotaciones de GitHub Actions (`::error`).

Uso en el CI:  python tools/anotar_pytest.py pytest.log
Las anotaciones aparecen en la página de la ejecución y en la API pública de check-runs, aunque el log completo requiera permisos.
"""

import re
import sys
from pathlib import Path

MAX = 3500  # GitHub recorta los mensajes largos


def anotar(texto: str) -> list[str]:
    salida = []
    # un bloque por prueba fallida: encabezado `___ nombre ___` y las líneas `E ` que siguen
    partes = re.split(r"\n_{3,} (.+?) _{3,}\n", texto)
    for nombre, cuerpo in zip(partes[1::2], partes[2::2], strict=False):
        errores = [linea for linea in cuerpo.splitlines() if linea.startswith("E ")][:8]
        mensaje = " | ".join(linea.strip() for linea in errores) or cuerpo.strip().splitlines()[-1]
        salida.append(f"{nombre}: {mensaje}"[:MAX])
    for linea in texto.splitlines():
        if linea.startswith(("FAILED ", "ERROR ")):
            salida.append(linea[:MAX])
    return salida


def main(ruta: str) -> int:
    for m in anotar(Path(ruta).read_text(encoding="utf-8", errors="replace"))[:30]:
        print(
            "::error title=pytest::" + m.replace("%", "%25").replace("\r", "").replace("\n", "%0A")
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
