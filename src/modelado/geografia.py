"""Variables geográficas y de tamaño derivadas del dataset IES x año, para el análisis de sesgo y las métricas por subgrupo.

Todo se calcula en memoria sobre una copia: el dataset de `data/processed/` no cambia.
"""
import numpy as np
import pandas as pd

# Región natural por departamento (código DANE de 2 dígitos). Convención usual: Cauca, Nariño, Chocó y Valle del Cauca en
# la Pacífica; Meta, Arauca, Casanare y Vichada en la Orinoquía; Caquetá, Putumayo, Amazonas, Guainía, Guaviare y Vaupés
# en la Amazonía; San Andrés con el Caribe. Bogotá (11) va aparte de la región Andina porque concentra la oferta.
REGION_POR_DEPTO = {
    "08": "Caribe", "13": "Caribe", "20": "Caribe", "23": "Caribe", "44": "Caribe", "47": "Caribe", "70": "Caribe", "88": "Caribe",
    "05": "Andina", "15": "Andina", "17": "Andina", "25": "Andina", "41": "Andina", "54": "Andina", "63": "Andina",
    "66": "Andina", "68": "Andina", "73": "Andina",
    "11": "Bogotá D.C.",
    "19": "Pacífica", "27": "Pacífica", "52": "Pacífica", "76": "Pacífica",
    "50": "Orinoquía", "81": "Orinoquía", "85": "Orinoquía", "99": "Orinoquía",
    "18": "Amazonía", "86": "Amazonía", "91": "Amazonía", "94": "Amazonía", "95": "Amazonía", "97": "Amazonía",
}
# Municipios de las tres ciudades que concentran la oferta
CIUDADES_PRINCIPALES = {"11001": "Bogotá", "05001": "Medellín", "76001": "Cali"}
RESTO = "Resto del país"

# Tramos del denominador de la tasa (matrícula SPADIES en t-2). El mínimo del pipeline es 100.
CORTES_TAMANO = [0, 200, 500, 1000, 5000, np.inf]
ETIQUETAS_TAMANO = ["menos de 200","200–499", "500–999", "1.000–4.999", "5.000 o más"]


def denominador(df: pd.DataFrame) -> pd.Series:
    """Matrícula SPADIES en t-2 (el denominador de la tasa), recuperada de `log_matriculados_spadies_t2` (log1p)."""
    return np.expm1(df["log_matriculados_spadies_t2"])


def agregar_geografia(df: pd.DataFrame) -> pd.DataFrame:
    """Copia con `cod_depto`, `region`, `ciudad_principal` y `tramo_tamano` (si hay denominador)."""
    out = df.copy()
    municipio = out["cod_divipola_mpio"].astype("string")
    out["cod_depto"] = municipio.str[:2]
    out["region"] = out["cod_depto"].map(REGION_POR_DEPTO).fillna("sin dato").astype(object)
    out["ciudad_principal"] = municipio.map(CIUDADES_PRINCIPALES).fillna(RESTO).astype(object)
    if "log_matriculados_spadies_t2" in out.columns:
        out["tramo_tamano"] = pd.cut(denominador(out), bins=CORTES_TAMANO, labels=ETIQUETAS_TAMANO, right=False)
    return out
