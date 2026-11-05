"""Selección de las IES cuyo export de SPADIES hay que descargar."""
import pandas as pd

from src.integrate.ies_descargar import CAPITALES_PRINCIPALES, COLUMNAS, seleccionar_ies


def _datos():
    """10 IES: 2 en el Valle (una pequeña), 3 en Bogotá, 1 en Medellín, 1 en Barranquilla, 3 en municipios menores."""
    sedes = {
        "1": "76001", "2": "76520", "3": "11001", "4": "11001", "5": "11001", "6": "05001",
        "7": "08001", "8": "25754", "9": "25754", "10": "25754",
    }
    matricula = {"1": 5000, "2": 10, "3": 90000, "4": 80000, "5": 70000, "6": 60000, "7": 50000,
                 "8": 40000, "9": 30000, "10": 20000}
    cat = pd.DataFrame({"cod_ies": list(sedes), "cod_divipola_mpio": list(sedes.values()),
                        "sector": "1", "caracter": "4", "cod_snies_programa": "x"})
    ies = pd.DataFrame({"c_digo_instituci_n": list(sedes), "nombre_instituci_n": [f"IES {i}" for i in sedes],
                        "sector": ["Privado", "Oficial"] + ["Privado"] * 8})
    prog = pd.DataFrame({"cod_ies": list(matricula), "cod_snies_programa": "x", "anio": 2021,
                         "matriculados": list(matricula.values())})
    divipola = pd.DataFrame({
        "cod_divipola_mpio": ["76001", "76520", "11001", "05001", "08001", "25754"],
        "cod_depto": ["76", "76", "11", "05", "08", "25"],
        "depto_norm": ["VALLE DEL CAUCA", "VALLE DEL CAUCA", "BOGOTA D.C.", "ANTIOQUIA", "ATLANTICO", "CUNDINAMARCA"],
        "municipio_norm": ["CALI", "PALMIRA", "BOGOTA D.C.", "MEDELLIN", "BARRANQUILLA", "SOACHA"],
    })
    return ies, cat, prog, divipola


def test_columnas_y_formato_de_salida():
    ies, cat, prog, div = _datos()
    out = seleccionar_ies(ies, cat, prog, div, anio=2021, n_objetivo=5)
    assert list(out.columns) == COLUMNAS == [
        "codigo_snies", "nombre_oficial", "municipio_sede", "departamento", "sector", "matricula"]
    fila = out.set_index("codigo_snies").loc["2"]
    assert fila["municipio_sede"] == "PALMIRA" and fila["departamento"] == "VALLE DEL CAUCA"
    assert fila["sector"] == "oficial" and out.set_index("codigo_snies").loc["1", "sector"] == "privada"


def test_incluye_todas_las_del_valle_aunque_sean_pequenas():
    ies, cat, prog, div = _datos()
    assert {"1", "2"} <= set(seleccionar_ies(ies, cat, prog, div, anio=2021, n_objetivo=3)["codigo_snies"])


def test_cubre_las_ciudades_principales_antes_de_completar_por_matricula():
    ies, cat, prog, div = _datos()
    # obligatorias: Valle (1, 2) + 3 mayores de Bogotá (3, 4, 5) + Medellín (6) + Barranquilla (7); falta 1 para completar 8
    out = seleccionar_ies(ies, cat, prog, div, anio=2021, n_objetivo=8)
    assert set(out["codigo_snies"]) == {"1", "2", "3", "4", "5", "6", "7", "8"}  # la 8ª es la mayor de las restantes


def test_ordena_por_matricula_descendente():
    ies, cat, prog, div = _datos()
    assert seleccionar_ies(ies, cat, prog, div, anio=2021, n_objetivo=8)["matricula"].is_monotonic_decreasing


def test_si_las_obligatorias_superan_n_objetivo_se_conservan_todas():
    ies, cat, prog, div = _datos()
    assert len(seleccionar_ies(ies, cat, prog, div, anio=2021, n_objetivo=1)) == 7


def test_ies_sin_matricula_en_el_anio_no_se_seleccionan():
    ies, cat, prog, div = _datos()
    out = seleccionar_ies(ies, cat, prog[prog["cod_ies"] != "3"], div, anio=2021, n_objetivo=10)
    assert "3" not in set(out["codigo_snies"])


def test_las_capitales_principales_incluyen_las_cinco_pedidas():
    assert {"11001", "05001", "76001", "08001", "68001"} <= set(CAPITALES_PRINCIPALES)
