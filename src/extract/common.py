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
    """Escribe `<archivo>.metadata.json` junto al archivo: fuente, URL, fecha, filas y SHA-256.

    Hay un metadata por archivo (varios archivos pueden compartir carpeta). Si el hash no cambió
    respecto al metadata previo, se conserva la fecha de descarga original.
    """
    path = Path(path)
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            sha.update(bloque)
    destino = path.parent / f"{path.stem}.metadata.json"
    fecha = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if destino.exists():
        previo = json.loads(destino.read_text(encoding="utf-8"))
        if previo.get("sha256") == sha.hexdigest():
            fecha = previo.get("fecha_descarga", fecha)
    meta = {"fuente": fuente, "url": url, "fecha_descarga": fecha, "filas": filas, "sha256": sha.hexdigest()}
    destino.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return destino


def _get_con_reintentos(url: str, params: dict, reintentos: int, espera_s: float) -> list[dict]:
    """GET con reintentos y espera exponencial ante errores transitorios.

    Se reintenta: 5xx y 429, caídas de red, cortes de flujo y JSON truncado. Los demás 4xx se propagan.
    datos.gov.co devuelve 500/503 de forma intermitente.
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
        except (requests.RequestException, ValueError):  # red, corte de flujo, JSON inválido
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

    Escribe a `<dest>.part` de forma incremental y solo lo renombra a `dest` al terminar, de modo
    que una descarga interrumpida nunca deja un CSV truncado. Socrata omite los campos nulos del
    JSON: si no se pasan `columnas`, el encabezado sale de la primera fila y, si luego aparece una
    clave que no está en él, se lanza ValueError (en vez de descartarla en silencio).
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    parcial = dest.with_name(dest.name + ".part")
    url = f"{base_url}/{dataset_id}.json"
    offset, filas = 0, 0
    try:
        with open(parcial, "w", newline="", encoding="utf-8") as f:
            escritor, encabezado = None, None
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
                    escritor = csv.DictWriter(f, fieldnames=encabezado, extrasaction="raise")
                    escritor.writeheader()
                nuevas = {k for fila in pagina for k in fila} - set(encabezado)
                if nuevas:
                    raise ValueError(
                        f"{dataset_id}: columnas {sorted(nuevas)} ausentes del encabezado; "
                        "pasa `columnas` explícitas a download_socrata"
                    )
                escritor.writerows(pagina)
                filas += len(pagina)
                offset += page_size
        parcial.replace(dest)
    except BaseException:
        parcial.unlink(missing_ok=True)
        raise
    write_metadata(dest, fuente=dataset_id, url=url, filas=filas)
    return dest
