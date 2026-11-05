"""Pruebas de la descarga genérica Socrata y del metadata.json."""
import json

import pandas as pd
import pytest

from src.extract import common


class _Respuesta:
    def __init__(self, filas):
        self._filas = filas

    def raise_for_status(self):
        pass

    def json(self):
        return self._filas


def _simular(monkeypatch, paginas):
    """Reemplaza requests.get devolviendo `paginas` en orden; registra los parámetros."""
    llamadas = []

    def falso_get(url, params=None, timeout=None):
        llamadas.append(dict(params))
        i = len(llamadas) - 1
        return _Respuesta(paginas[i] if i < len(paginas) else [])

    monkeypatch.setattr(common.requests, "get", falso_get)
    return llamadas


def test_pagina_hasta_vacio(monkeypatch, tmp_path):
    paginas = [[{"a": "1", "b": "x"}, {"a": "2", "b": "y"}], [{"a": "3", "b": "z"}], []]
    llamadas = _simular(monkeypatch, paginas)
    destino = common.download_socrata("abcd-1234", tmp_path / "datos.csv", page_size=2)
    assert len(pd.read_csv(destino)) == 3
    assert [c["$offset"] for c in llamadas] == [0, 2, 4][: len(llamadas)]
    assert len(llamadas) == 3 and all(c["$limit"] == 2 for c in llamadas)


def test_select_columnas(monkeypatch, tmp_path):
    llamadas = _simular(monkeypatch, [[{"a": "1"}], []])
    common.download_socrata("abcd-1234", tmp_path / "d.csv", columnas=["a", "b"], page_size=5)
    assert llamadas[0]["$select"] == "a,b"


def test_metadata_json(monkeypatch, tmp_path):
    _simular(monkeypatch, [[{"a": "1"}, {"a": "2"}], []])
    destino = common.download_socrata("abcd-1234", tmp_path / "d.csv", page_size=2)
    meta = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert meta["filas"] == 2
    assert len(meta["sha256"]) == 64
    assert meta["fuente"] == "abcd-1234" and "abcd-1234" in meta["url"]
    assert destino.exists()


def test_where_filtra_en_servidor(monkeypatch, tmp_path):
    llamadas = _simular(monkeypatch, [[{"a": "1"}], []])
    common.download_socrata("abcd-1234", tmp_path / "d.csv", where="periodo >= '20141'", page_size=5)
    assert llamadas[0]["$where"] == "periodo >= '20141'"


class _RespuestaError(_Respuesta):
    def __init__(self, codigo):
        super().__init__([])
        self.status_code = codigo

    def raise_for_status(self):
        raise common.requests.HTTPError(f"{self.status_code} Server Error", response=self)


def test_reintenta_ante_error_5xx_transitorio(monkeypatch, tmp_path):
    respuestas = [_RespuestaError(503), _RespuestaError(500), _Respuesta([{"a": "1"}]), _Respuesta([])]
    llamadas = []

    def falso_get(url, params=None, timeout=None):
        llamadas.append(dict(params))
        return respuestas[len(llamadas) - 1]

    monkeypatch.setattr(common.requests, "get", falso_get)
    monkeypatch.setattr(common.time, "sleep", lambda s: None)
    destino = common.download_socrata("abcd-1234", tmp_path / "d.csv", page_size=5)
    assert len(pd.read_csv(destino)) == 1
    assert [c["$offset"] for c in llamadas] == [0, 0, 0, 5]  # reintenta la misma página


def test_se_rinde_tras_agotar_reintentos(monkeypatch, tmp_path):
    monkeypatch.setattr(common.requests, "get", lambda url, params=None, timeout=None: _RespuestaError(503))
    monkeypatch.setattr(common.time, "sleep", lambda s: None)
    with pytest.raises(common.requests.HTTPError):
        common.download_socrata("abcd-1234", tmp_path / "d.csv", page_size=5, reintentos=3)


def test_no_reintenta_errores_4xx(monkeypatch, tmp_path):
    llamadas = []

    def falso_get(url, params=None, timeout=None):
        llamadas.append(1)
        return _RespuestaError(404)

    monkeypatch.setattr(common.requests, "get", falso_get)
    monkeypatch.setattr(common.time, "sleep", lambda s: None)
    with pytest.raises(common.requests.HTTPError):
        common.download_socrata("abcd-1234", tmp_path / "d.csv")
    assert len(llamadas) == 1
