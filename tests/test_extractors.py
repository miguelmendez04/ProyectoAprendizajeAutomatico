"""Los extractores solo enlazan la configuración con download_socrata."""
from pathlib import Path

from src.extract import dane, snies

CFG = {
    "rutas": {"raw": "data/raw"},
    "datasets": {
        "snies_matricula_mpio": "aaaa-1111",
        "snies_matricula_programa": "bbbb-2222",
        "divipola_municipios": "cccc-3333",
    },
}


def _registrar(monkeypatch, modulo):
    llamadas = []

    def falso(dataset_id, dest, **kwargs):
        llamadas.append((dataset_id, Path(dest).as_posix()))
        return Path(dest)

    monkeypatch.setattr(modulo, "download_socrata", falso)
    return llamadas


def test_extract_snies_descarga_los_dos_datasets(monkeypatch):
    llamadas = _registrar(monkeypatch, snies)
    rutas = snies.extract_snies(CFG)
    assert ("aaaa-1111", "data/raw/snies/matricula_mpio.csv") in llamadas
    assert ("bbbb-2222", "data/raw/snies/matricula_programa.csv") in llamadas
    assert set(rutas) == {"matricula_mpio", "matricula_programa"}


def test_extract_divipola(monkeypatch):
    llamadas = _registrar(monkeypatch, dane)
    assert dane.extract_divipola(CFG).as_posix() == "data/raw/dane/divipola.csv"
    assert llamadas == [("cccc-3333", "data/raw/dane/divipola.csv")]


def test_extractores_piden_columnas_explicitas(monkeypatch):
    """Socrata omite campos nulos: sin `columnas` el encabezado podría perder columnas."""
    pedidas = {}

    def falso(dataset_id, dest, **kwargs):
        pedidas[dataset_id] = kwargs.get("columnas")
        return Path(dest)

    monkeypatch.setattr(snies, "download_socrata", falso)
    monkeypatch.setattr(dane, "download_socrata", falso)
    snies.extract_snies(CFG)
    dane.extract_divipola(CFG)
    assert "a_o" in pedidas["aaaa-1111"] and "ies_con_oferta" in pedidas["aaaa-1111"]
    assert "matriculados_2015" in pedidas["bbbb-2222"] and "id_g_nero" in pedidas["bbbb-2222"]
    assert pedidas["cccc-3333"] == ["cod_dpto", "dpto", "cod_mpio", "nom_mpio", "tipo_municipio", "longitud", "latitud"]
