"""Técnicas de mitigación de sesgo para el modelado. Todas son opciones: ninguna cambia el dataset ni los valores por defecto.

- `pesos_muestrales`: pesos por fila (por tamaño, por grupo o por IES) para `sample_weight`.
- `agregar_panel_incompleto`: años observados por IES e indicador de panel incompleto.
- `filtrar_denominador_minimo`: umbral de denominador más alto que el del pipeline (sensibilidad).
- `metricas_por_subgrupo`: MAE, RMSE y sesgo medio por sector, carácter, región, ciudad o tamaño.

Las particiones alternativas (estratificada, por departamento, dejando fuera a Bogotá) están en `particion.py`.
"""
import numpy as np
import pandas as pd

from src.modelado.geografia import agregar_geografia, denominador
from src.modelado.particion import ETIQUETAS_CARACTER, ETIQUETAS_SECTOR

ESQUEMAS_PESOS = ("ninguno", "matriculados", "raiz_matriculados", "inverso_grupo", "inverso_ies")
GRUPOS_POR_DEFECTO = ("sector", "caracter")


def pesos_muestrales(df: pd.DataFrame, esquema: str = "ninguno", grupos: list[str] | tuple[str, ...] = GRUPOS_POR_DEFECTO) -> np.ndarray:
    """Pesos por fila con media 1 (así la escala de la pérdida no cambia).

    - `ninguno`: todos 1.
    - `matriculados`: proporcional al denominador de la tasa. Ataca la varianza de las IES chicas, pero deja que unas pocas
      IES enormes (el SENA) dominen el ajuste.
    - `raiz_matriculados`: proporcional a su raíz cuadrada; el mismo efecto con menos concentración.
    - `inverso_grupo`: inverso de la frecuencia (en filas) del grupo `grupos` (sector x carácter por defecto). Ataca el
      desbalance entre grupos; sube el peso de los grupos escasos y con él el de sus valores atípicos.
    - `inverso_ies`: inverso del número de años de la IES. Cada IES pesa igual sin importar cuántos años tenga: ataca el
      sesgo de supervivencia (las IES con panel completo aportan más filas).
    """
    if esquema not in ESQUEMAS_PESOS:
        raise ValueError(f"esquema de pesos inválido: {esquema!r}; debe ser uno de {list(ESQUEMAS_PESOS)}")
    n = len(df)
    if esquema == "ninguno":
        w = np.ones(n)
    elif esquema in ("matriculados", "raiz_matriculados"):
        base = denominador(df).to_numpy(dtype=float)
        if np.isnan(base).any():
            raise ValueError("hay filas sin denominador (log_matriculados_spadies_t2 nulo): no se pueden ponderar por tamaño")
        w = base if esquema == "matriculados" else np.sqrt(base)
    elif esquema == "inverso_grupo":
        etiqueta = df[list(grupos)].astype(str).agg("|".join, axis=1)
        w = 1.0 / etiqueta.map(etiqueta.value_counts()).to_numpy(dtype=float)
    else:
        w = 1.0 / df.groupby("cod_ies")["anio"].transform("size").to_numpy(dtype=float)
    return w / w.mean()


def pesos_desde_config(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    """Pesos según `config.yaml: modelado.pesos_muestrales` (`ninguno` por defecto)."""
    m = cfg.get("modelado", {})
    return pesos_muestrales(df, m.get("pesos_muestrales", "ninguno"), m.get("grupos_pesos", GRUPOS_POR_DEFECTO))


def agregar_panel_incompleto(df: pd.DataFrame, anios: tuple[int, int] | None = None) -> pd.DataFrame:
    """Copia con `n_anios_ies` (años con tasa de la IES) y `panel_incompleto` (True si no cubre todos los años de la ventana).

    La ventana es `anios=(inicial, final)` o, si no se da, el rango de años del propio dataframe.
    Aviso: `n_anios_ies` usa años posteriores al de cada fila (si la IES siguió reportando). Sirve para diagnosticar,
    estratificar o ponderar; como variable explicativa deja pasar información del futuro (ver la bitácora).
    """
    inicio, fin = anios if anios is not None else (int(df["anio"].min()), int(df["anio"].max()))
    esperados = fin - inicio + 1
    out = df.copy()
    dentro = out["anio"].between(inicio, fin)
    out["n_anios_ies"] = out["cod_ies"].map(out[dentro].groupby("cod_ies")["anio"].nunique()).fillna(0).astype(int)
    out["panel_incompleto"] = out["n_anios_ies"] < esperados
    return out


def filtrar_denominador_minimo(df: pd.DataFrame, minimo: float, minimo_pipeline: float = 100) -> pd.DataFrame:
    """Filas con denominador (matrícula SPADIES en t-2) de al menos `minimo`: sensibilidad al umbral del pipeline.

    Solo permite umbrales mayores o iguales al del pipeline (`min_matriculados_t2`): las filas por debajo ya no están en el
    dataset. Para un umbral menor hay que rehacer el dataset (ver `sesgo.sensibilidad_umbral`).
    """
    if minimo < minimo_pipeline:
        raise ValueError(
            f"umbral {minimo} menor que el del pipeline ({minimo_pipeline}): esas filas ya no están en el dataset; "
            "regenérelo con min_matriculados_t2 más bajo"
        )
    # el denominador se recupera de un log1p: la tolerancia evita perder las filas exactamente en el umbral por redondeo
    return df[denominador(df) >= minimo - 1e-6].reset_index(drop=True)


def metricas_por_subgrupo(
    df: pd.DataFrame, y_true, y_pred, columnas: list[str] | tuple[str, ...] = ("sector", "caracter", "region", "ciudad_principal", "tramo_tamano"),
    peso=None,
) -> pd.DataFrame:
    """MAE, RMSE y sesgo medio (predicho − observado) por subgrupo, más el total.

    `df` debe estar alineado por posición con `y_true` y `y_pred`. Las columnas geográficas y de tamaño se agregan solas
    si faltan. Devuelve (variable, grupo, filas, ies, mae, rmse, sesgo_medio); con `peso`, MAE y RMSE ponderados.
    """
    datos = agregar_geografia(df) if {"region", "ciudad_principal"} - set(df.columns) and "cod_divipola_mpio" in df.columns else df.copy()
    datos = datos.reset_index(drop=True)
    for col, etiquetas in (("sector", ETIQUETAS_SECTOR), ("caracter", ETIQUETAS_CARACTER)):  # ids de SNIES -> etiqueta legible
        if col in datos.columns and datos[col].astype(str).isin(etiquetas).all():
            datos[col] = datos[col].astype(str).map(etiquetas)
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if not (len(datos) == len(y_true) == len(y_pred)):
        raise ValueError("df, y_true y y_pred deben tener la misma longitud")
    datos["_error"] = y_pred - y_true
    datos["_peso"] = np.ones(len(datos)) if peso is None else np.asarray(peso, dtype=float)

    def resumen(parte: pd.DataFrame) -> dict:
        w, e = parte["_peso"].to_numpy(), parte["_error"].to_numpy()
        return {"filas": len(parte), "ies": parte["cod_ies"].nunique() if "cod_ies" in parte else np.nan,
                "mae": np.average(np.abs(e), weights=w), "rmse": float(np.sqrt(np.average(e**2, weights=w))),
                "sesgo_medio": np.average(e, weights=w)}

    filas = [{"variable": "total", "grupo": "todas", **resumen(datos)}]
    for col in columnas:
        if col not in datos.columns:
            continue
        for valor, parte in datos.groupby(col, observed=True, dropna=False):
            filas.append({"variable": col, "grupo": str(valor), **resumen(parte)})
    return pd.DataFrame(filas)
