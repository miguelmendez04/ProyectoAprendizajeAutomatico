"""Utilidades de partición del dataset IES x año para la fase de modelado.

Dos esquemas de validación, ambos pensados para no inflar el desempeño:
- `particion_group_kfold`: validación cruzada por IES (`cod_ies`); una institución nunca está a la vez en
  entrenamiento y prueba, porque sus años consecutivos son casi la misma observación.
- `particion_temporal`: entrena con los años iniciales y prueba con los finales (2015–2019 / 2020–2021 por
  defecto). Aquí las mismas IES aparecen en ambos lados: mide la capacidad de extrapolar en el tiempo.

Limitación: el dataset no trae variables de mercado laboral (ninguna serie municipal de TerriData cubre 2015–2021).
"""
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

COLUMNAS_CATEGORICAS = ["sector", "caracter"]  # ids de SNIES: son categorías, no cantidades
_TEXTO = ["cod_ies", "cod_divipola_mpio", *COLUMNAS_CATEGORICAS]
# id_sector de SNIES; coincide con el sector (Oficial / Privado) del catálogo oficial de IES
ETIQUETAS_SECTOR = {"1": "oficial", "2": "privada"}
# id_caracter de SNIES, verificado contra el carácter académico del catálogo oficial de IES (ver la bitácora)
ETIQUETAS_CARACTER = {"1": "técnica profesional", "2": "tecnológica", "3": "institución universitaria", "4": "universidad"}
# Indicadores de nulo por bloque de años (Saber 11, MDM y SNIES), que construye el ETL
INDICADORES_FALTA = ["falta_saber11", "falta_mdm", "falta_snies"]
VENTANA_POR_DEFECTO = (2015, 2021)


def cargar_dataset(cfg: dict, excluir_sena: bool | None = None, ventana_modelado: tuple[int, int] | None = None) -> pd.DataFrame:
    """Lee `data/processed/dataset_<unidad>.csv` con los códigos y las categorías como texto.

    - Sensibilidad del SENA: con `excluir_sena=True` (o `config.yaml: modelado.excluir_sena: true` si no se pasa el
      argumento) se quitan las filas con `es_sena`.
    - Ventana temporal: `ventana_modelado=(2015, 2021)` por defecto (o `modelado.ventana_modelado` de la configuración);
      `(2017, 2021)` es la sensibilidad que evita los años 2015–2016, donde faltan MDM y SNIES. Los argumentos mandan
      sobre la configuración.
    - Opciones de sesgo (apagadas por defecto): `modelado.umbral_denominador` (null) sube el denominador mínimo de la tasa
      (50 no es posible aquí: ver `sesgo.sensibilidad_umbral`) y `modelado.indicador_panel_incompleto` (false) agrega
      `n_anios_ies` y `panel_incompleto`.
    """
    ruta = Path(cfg["rutas"]["processed"]) / f"dataset_{cfg['unidad_analisis']}.csv"
    cabecera = pd.read_csv(ruta, nrows=0).columns
    df = pd.read_csv(ruta, dtype={c: str for c in _TEXTO if c in cabecera})
    modelado = cfg.get("modelado", {})
    if excluir_sena is None:
        excluir_sena = bool(modelado.get("excluir_sena", False))
    if excluir_sena and "es_sena" in df.columns:
        df = df[~df["es_sena"]]
    ventana = tuple(ventana_modelado if ventana_modelado is not None else modelado.get("ventana_modelado", VENTANA_POR_DEFECTO))
    if len(ventana) != 2 or ventana[0] > ventana[1]:
        raise ValueError(f"ventana_modelado inválida: {ventana}; debe ser (año_inicial, año_final) con inicial <= final")
    df = df[df["anio"].between(*ventana)].reset_index(drop=True)
    umbral = modelado.get("umbral_denominador")
    if umbral is not None or modelado.get("indicador_panel_incompleto", False):
        from src.modelado.mitigacion import agregar_panel_incompleto, filtrar_denominador_minimo  # evita el import circular

        if umbral is not None:
            df = filtrar_denominador_minimo(df, umbral, cfg.get("min_matriculados_t2", 100))
        if modelado.get("indicador_panel_incompleto", False):
            df = agregar_panel_incompleto(df, ventana)
    return df


# Variables que no entran como explicativas por defecto:
# - td_poblacion_total_t1: la proyección de población cambia de base (censo 2005 -> 2018) entre los años objetivo 2018 y
#   2019 (-3 % nacional, -10 % en Bogotá) y el nivel absoluto lee ese salto como una caída. Se conservan las variables
#   relativas (% rural, % por edad) hasta tener las retroproyecciones del DANE con base 2018.
# - ratio_spadies_snies: incluye el denominador de la propia tasa (SPADIES t-2); es una variable de diagnóstico.
# - matriculados_ies_t2: muy asimétrica (el SENA tiene ~430 mil frente a una mediana de ~3,7 mil); entra su logaritmo
#   (`log_matriculados_ies_t2`, log1p). La versión bruta sigue en el dataset.
# - log_matriculados_ies_t2: alternativa a `log_matriculados_spadies_t2` (tamaño según SNIES; 2015–2016 nulo).
#   La de SPADIES es la de tamaño por defecto: viene completa 2015–2021 y es del mismo desfase t-2 que el denominador de
#   la tasa, de modo que no es fuga (usa información de t-2). Sí guarda una relación mecánica con la tasa (es su denominador).
# Columnas que nunca son explicativas, ni siquiera con `excluir=[]`: `n_anios_ies` cuenta años posteriores al de cada fila
# (si la IES siguió reportando), es decir, información del futuro. `panel_incompleto` sí puede entrar (opción de sesgo).
NUNCA_EXPLICATIVAS = ("n_anios_ies",)
EXCLUIDAS_POR_DEFECTO = ["td_poblacion_total_t1", "ratio_spadies_snies", "matriculados_ies_t2", "log_matriculados_ies_t2"]


def columnas_explicativas(df: pd.DataFrame, objetivo: str = "tasa_desercion", excluir: list[str] | None = None) -> list[str]:
    """Variables numéricas que pueden entrar como explicativas (sin llaves, año, objetivo ni `excluir`).

    `excluir=None` aplica `EXCLUIDAS_POR_DEFECTO`; con `[]` no se excluye nada más. Las banderas (booleanas) nunca entran.
    """
    fuera = {objetivo, "anio", "semestre", *NUNCA_EXPLICATIVAS, *(EXCLUIDAS_POR_DEFECTO if excluir is None else excluir)}
    numericas = df.select_dtypes("number").columns
    return [c for c in numericas if c not in fuera and not c.startswith("cod_")]


def columnas_explicativas_desde_config(df: pd.DataFrame, cfg: dict, objetivo: str = "tasa_desercion") -> list[str]:
    """Como `columnas_explicativas`, con la lista `config.yaml: modelado.excluir_variables` (o la de por defecto)."""
    return columnas_explicativas(df, objetivo, cfg.get("modelado", {}).get("excluir_variables"))


def columnas_categoricas(df: pd.DataFrame) -> list[str]:
    """Variables categóricas del dataset (`sector`, `caracter`): ids de SNIES para pasar por one-hot."""
    return [c for c in COLUMNAS_CATEGORICAS if c in df.columns]


def etiquetar_categoricas(df: pd.DataFrame) -> pd.DataFrame:
    """Copia con `sector` y `caracter` como etiquetas legibles; un código ausente o desconocido queda 'desconocido'."""
    out = df.copy()
    for col, etiquetas in (("sector", ETIQUETAS_SECTOR), ("caracter", ETIQUETAS_CARACTER)):
        if col in out.columns:
            out[col] = out[col].astype("string").map(etiquetas).fillna("desconocido").astype(object)
    return out


def variables_modelado(
    df: pd.DataFrame, objetivo: str = "tasa_desercion", excluir: list[str] | None = None, incluir_indicadores_falta: bool = True,
    incluir_panel_incompleto: bool = False,
) -> tuple[list[str], list[str]]:
    """(numéricas, categóricas) por separado, para un ColumnTransformer (ver `construir_preprocesador`).

    Las numéricas son `columnas_explicativas` más, si se pide, los indicadores `falta_*` (0/1; marcan los bloques de años
    sin Saber 11, MDM o SNIES). Con `incluir_panel_incompleto` entra además `panel_incompleto` (0/1, opción de sesgo; ver
    `mitigacion.agregar_panel_incompleto`). Las categóricas son `sector` y `caracter`.
    """
    numericas = columnas_explicativas(df, objetivo, excluir)
    if incluir_indicadores_falta:
        numericas += [c for c in INDICADORES_FALTA if c in df.columns and c not in numericas]
    if incluir_panel_incompleto and "panel_incompleto" in df.columns:
        numericas.append("panel_incompleto")
    return numericas, columnas_categoricas(df)


def variables_modelado_desde_config(df: pd.DataFrame, cfg: dict, objetivo: str = "tasa_desercion") -> tuple[list[str], list[str]]:
    """Como `variables_modelado`, con `modelado.excluir_variables` e `incluir_indicadores_falta` de `config.yaml`."""
    m = cfg.get("modelado", {})
    return variables_modelado(df, objetivo, m.get("excluir_variables"), m.get("incluir_indicadores_falta", True),
                              m.get("indicador_panel_incompleto", False))


def preparar_x(df: pd.DataFrame, numericas: list[str], categoricas: list[str]) -> pd.DataFrame:
    """Matriz de entrada del preprocesador: numéricas (las booleanas pasan a 0/1) y categóricas con etiqueta."""
    x = etiquetar_categoricas(df)[numericas + categoricas].copy()
    for c in numericas:
        if x[c].dtype == bool:
            x[c] = x[c].astype(int)
    return x


def construir_preprocesador(numericas: list[str], categoricas: list[str]) -> ColumnTransformer:
    """ColumnTransformer: numéricas con imputación por mediana y estandarización; categóricas con one-hot.

    Las categorías que no se vieron en el entrenamiento se ignoran (todo ceros). La imputación y la estandarización
    se ajustan solo con los datos que reciba `fit`: dentro de un Pipeline y una validación cruzada no hay fuga.
    """
    return ColumnTransformer([
        ("num", Pipeline([("imputar", SimpleImputer(strategy="median")), ("escalar", StandardScaler())]), numericas),
        ("cat", Pipeline([("imputar", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categoricas),
    ])


def particion_group_kfold(df: pd.DataFrame, n_splits: int = 5, grupo: str = "cod_ies") -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Genera (índices de entrenamiento, índices de prueba) posicionales con GroupKFold por `grupo`."""
    n_grupos = df[grupo].nunique()
    if n_splits > n_grupos:
        raise ValueError(f"n_splits={n_splits} supera el número de IES distintas ({n_grupos})")
    yield from GroupKFold(n_splits=n_splits).split(df, groups=df[grupo])


def particion_temporal(df: pd.DataFrame, ultimo_anio_entrenamiento: int = 2019) -> tuple[np.ndarray, np.ndarray]:
    """(entrenamiento, prueba): años <= `ultimo_anio_entrenamiento` frente a los posteriores (índices posicionales)."""
    entrena = np.flatnonzero((df["anio"] <= ultimo_anio_entrenamiento).to_numpy())
    prueba = np.flatnonzero((df["anio"] > ultimo_anio_entrenamiento).to_numpy())
    if len(entrena) == 0 or len(prueba) == 0:
        raise ValueError(
            f"la partición temporal con corte {ultimo_anio_entrenamiento} deja vacío el conjunto de "
            f"{'entrenamiento' if len(entrena) == 0 else 'prueba'}"
        )
    return entrena, prueba


def _etiqueta_estrato(df: pd.DataFrame, estratos: list[str], grupo: str, minimo: int) -> pd.Series:
    """Etiqueta de estrato por IES (combinación de `estratos`); los estratos con menos de `minimo` IES pasan a 'otros'."""
    por_ies = df.drop_duplicates(grupo).set_index(grupo)[list(estratos)].astype(str).agg("|".join, axis=1)
    escasos = por_ies.value_counts()[lambda s: s < minimo].index
    por_ies = por_ies.where(~por_ies.isin(escasos), "otros")
    return df[grupo].map(por_ies)


def particion_group_kfold_estratificado(
    df: pd.DataFrame, n_splits: int = 5, estratos: list[str] | tuple[str, ...] = ("sector", "caracter"),
    grupo: str = "cod_ies", semilla: int = 0,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """GroupKFold por IES con los pliegues estratificados por `estratos` (sector y carácter por defecto).

    Cada IES queda entera en un pliegue (como `particion_group_kfold`) y la mezcla oficial/privada y de carácter de cada
    pliegue se parece a la del conjunto. Los estratos con menos de `n_splits` IES se agrupan en 'otros'. Es aleatoria con
    `semilla` fija (StratifiedGroupKFold mezcla): reproducible, pero distinta de la partición determinista de GroupKFold.
    """
    n_grupos = df[grupo].nunique()
    if n_splits > n_grupos:
        raise ValueError(f"n_splits={n_splits} supera el número de IES distintas ({n_grupos})")
    etiquetas = _etiqueta_estrato(df, list(estratos), grupo, n_splits)
    yield from StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=semilla).split(df, etiquetas, groups=df[grupo])


def particion_por_departamento(df: pd.DataFrame, n_splits: int = 5) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """GroupKFold por departamento sede: un departamento nunca está a la vez en entrenamiento y prueba.

    Mide la generalización a territorios no vistos. Bogotá pesa mucho y un solo pliegue queda desbalanceado.
    """
    dep = df.assign(cod_depto=df["cod_divipola_mpio"].astype("string").str[:2])
    if dep["cod_depto"].isna().any():
        raise ValueError("hay IES sin municipio sede (cod_divipola_mpio nulo): no se pueden agrupar por departamento")
    yield from particion_group_kfold(dep, n_splits=n_splits, grupo="cod_depto")


def particion_dejando_fuera(df: pd.DataFrame, prefijos: list[str] | tuple[str, ...] = ("11",)) -> tuple[np.ndarray, np.ndarray]:
    """(entrenamiento, prueba): la prueba son las filas cuyo municipio sede empieza por alguno de `prefijos`.

    `("11",)` deja fuera a Bogotá; `("11001", "05001", "76001")`, a las tres ciudades principales. Entrena con el resto del
    país y evalúa en esas ciudades: mide cuánto del desempeño depende de ellas.
    """
    municipio = df["cod_divipola_mpio"].astype("string")
    es_prueba = municipio.str.startswith(tuple(prefijos)).fillna(False).to_numpy(dtype=bool)
    if not es_prueba.any() or es_prueba.all():
        raise ValueError(f"los prefijos {tuple(prefijos)} dejan vacío el conjunto de {'prueba' if not es_prueba.any() else 'entrenamiento'}")
    return np.flatnonzero(~es_prueba), np.flatnonzero(es_prueba)


def kfold_desde_config(df: pd.DataFrame, cfg: dict) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Pliegues según `particion`: `agrupar_por` (ies por defecto | departamento) y `estratificar_por` ([] por defecto).

    Con los valores por defecto es exactamente `particion_group_kfold` por IES.
    """
    p = cfg.get("particion", {})
    n = p.get("n_splits", 5)
    agrupar = p.get("agrupar_por", "ies")
    estratos = p.get("estratificar_por") or []
    if agrupar not in ("ies", "departamento"):
        raise ValueError(f"particion.agrupar_por inválido: {agrupar!r}; debe ser 'ies' o 'departamento'")
    if agrupar == "departamento":
        if estratos:
            raise ValueError("particion.estratificar_por solo aplica con agrupar_por: ies")
        return particion_por_departamento(df, n_splits=n)
    if estratos:
        return particion_group_kfold_estratificado(df, n_splits=n, estratos=estratos)
    return particion_group_kfold(df, n_splits=n)


def particiones_desde_config(df: pd.DataFrame, cfg: dict):
    """(generador de pliegues, partición temporal) con los parámetros de `config.yaml: particion`."""
    p = cfg.get("particion", {})
    return kfold_desde_config(df, cfg), particion_temporal(df, p.get("ultimo_anio_entrenamiento", 2019))
