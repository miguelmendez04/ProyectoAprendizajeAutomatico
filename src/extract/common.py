"""Utilidades de extracción comunes: descarga Socrata y trazabilidad (metadata.json)."""
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_URL = "https://www.datos.gov.co/resource"


def write_metadata(path: Path, fuente: str, url: str, filas: int) -> Path:
    """Escribe `metadata.json` junto al archivo: fuente, URL, fecha, filas y hash SHA-256."""
    path = Path(path)
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            sha.update(bloque)
    meta = {
        "fuente": fuente,
        "url": url,
        "fecha_descarga": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "filas": filas,
        "sha256": sha.hexdigest(),
    }
    destino = path.parent / "metadata.json"
    destino.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return destino


def _get_con_reintentos(url: str, params: dict, reintentos: int, espera_s: float) -> list[dict]:
    """GET con reintentos y espera exponencial ante errores transitorios (5xx, 429, red).

    datos.gov.co devuelve 500/503 de forma intermitente; los 4xx restantes se propagan de inmediato.
    """
    for intento in range(reintentos):
        ultimo = intento == reintentos - 1
        try:
            respuesta = requests.get(url, params=params, timeout=300)
            respuesta.raise_for_status()
            return respuesta.json()
        except requests.HTTPError as e:
            codigo = getattr(e.response, "status_code", None)
            transitorio = codigo is not None and (codigo >= 500 or codigo == 429)
            if not transitorio or ultimo:
                raise
        except (requests.ConnectionError, requests.Timeout):
            if ultimo:
                raise
        time.sleep(espera_s * 2**intento)


def download_socrata(
    dataset_id: str,
    dest: Path,
    columnas: list[str] | None = None,
    where: str | None = None,
    page_size: int = 50000,
    base_url: str = BASE_URL,
    reintentos: int = 5,
    espera_s: float = 2.0,
) -> Path:
    """Descarga un dataset de datos.gov.co paginando con $limit/$offset hasta página vacía.

    Escribe el CSV de forma incremental (no acumula todo en memoria). Si no se pasan
    `columnas`, el encabezado sale de la primera página; conviene pasarlas porque
    Socrata omite los campos nulos en el JSON.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{base_url}/{dataset_id}.json"
    offset, filas = 0, 0
    with open(dest, "w", newline="", encoding="utf-8") as f:
        escritor = None
        while True:
            params = {"$limit": page_size, "$offset": offset, "$order": ":id"}
            if columnas:
                params["$select"] = ",".join(columnas)
            if where:
                params["$where"] = where  # filtro en el servidor (SoQL)
            pagina = _get_con_reintentos(url, params, reintentos, espera_s)
            if not pagina:
                break
            if escritor is None:
                encabezado = columnas or list(pagina[0].keys())
                escritor = csv.DictWriter(f, fieldnames=encabezado, extrasaction="ignore")
                escritor.writeheader()
            escritor.writerows(pagina)
            filas += len(pagina)
            offset += page_size
    write_metadata(dest, fuente=dataset_id, url=url, filas=filas)
    return dest
