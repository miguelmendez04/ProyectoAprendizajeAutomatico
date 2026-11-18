"""Variables de historia de la tasa de cada entidad (IES): solo años anteriores al de la fila.

Para cada fila (entidad, año t) y usando únicamente tasas de años < t:
- `tasa_t1`: la tasa del año t-1 de la misma entidad (nula si no hay tasa de ese año exacto).
- `tasa_t2`: la tasa del año t-2.
- `tasa_media_previa`: la media de todas las tasas anteriores disponibles (años <= t-1; nula si no hay ninguna).
- `n_anios_previos`: cuántas tasas anteriores hay (0 si ninguna).
- `falta_historia`: verdadero si no hay `tasa_t1`.
Variante **rezagada** (escenario conservador: la tasa de t-1 aún no estaría publicada al inicio de t), sin usar t-1:
- `tasa_media_rezagada`: media de las tasas de los años <= t-2 (nula si no hay ninguna).
- `n_anios_rezagados`: cuántas tasas de años <= t-2 hay.
- `falta_historia_rezagada`: verdadero si no hay `tasa_t2`. (`tasa_t2` es compartida con la historia normal.)

La historia sale de las filas del dataset (ya depuradas, así que `tasa_t1` coincide con la línea base ingenua) y, para los
años anteriores a la ventana del dataset, de `previa`: las tasas anuales de SPADIES de esos años, calculadas con la misma
definición y los mismos filtros de calidad (denominador mínimo, ceros iniciales como faltantes) y usadas **solo** para alimentar
estas variables: las filas del dataset (y su objetivo) no cambian. «Anterior» se define por el año: la tasa de t, y cualquiera
posterior, nunca entra. A diferencia de `n_anios_ies` (que cuenta también años posteriores), `n_anios_previos` no usa el futuro.
"""
import pandas as pd

COLUMNAS_HISTORIA = ["tasa_t1", "tasa_t2", "tasa_media_previa", "n_anios_previos", "falta_historia"]
COLUMNAS_HISTORIA_REZAGADA = ["tasa_t2", "tasa_media_rezagada", "n_anios_rezagados", "falta_historia_rezagada"]
COLUMNAS_HISTORIA_TODAS = COLUMNAS_HISTORIA + [c for c in COLUMNAS_HISTORIA_REZAGADA if c not in COLUMNAS_HISTORIA]


def _panel(df: pd.DataFrame, llave: list[str], objetivo: str, previa: pd.DataFrame | None) -> pd.DataFrame:
    """Tasas por entidad y año (las filas de `df` más `previa` de años anteriores a `df`), con la suma y el conteo acumulados
    hasta cada año inclusive (acumulados secuenciales: el valor de un año no depende de los posteriores)."""
    panel = df[llave + ["anio", objetivo]]
    if previa is not None and len(previa):
        antes = previa.loc[previa["anio"] < df["anio"].min(), llave + ["anio", objetivo]]  # sin solapar con las filas del dataset
        panel = pd.concat([antes, panel], ignore_index=True)
    panel = panel.dropna(subset=[objetivo])
    if panel.duplicated(llave + ["anio"]).any():
        raise ValueError(f"{llave + ['anio']} no es único en la historia (previa y dataset)")
    panel = panel.sort_values(llave + ["anio"]).reset_index(drop=True)
    grupo = panel.groupby(llave, sort=False)
    panel["_cs"] = grupo[objetivo].cumsum()
    panel["_cn"] = grupo.cumcount() + 1
    return panel


def _hasta(df: pd.DataFrame, panel: pd.DataFrame, llave: list[str], k: int) -> tuple[pd.Series, pd.Series]:
    """(media, n) de las tasas con año <= t-k de la misma entidad: el acumulado del último año del panel que no pasa de t-k."""
    consulta = df[llave].copy()
    consulta["_y"] = df["anio"] - k
    consulta["_i"] = df.index
    acumulado = panel[llave + ["anio", "_cs", "_cn"]].rename(columns={"anio": "_y"})
    r = pd.merge_asof(consulta.sort_values("_y"), acumulado.sort_values("_y"), on="_y", by=llave, direction="backward")
    r = r.set_index("_i").reindex(df.index)
    return r["_cs"] / r["_cn"], r["_cn"].fillna(0).astype(int)


def agregar_historia(
    df: pd.DataFrame, llave_entidad: list[str], objetivo: str = "tasa_desercion", previa: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Copia de `df` con las columnas de historia (`COLUMNAS_HISTORIA_TODAS`) al final.

    `llave_entidad`: columnas que identifican a la entidad (p. ej. `["cod_ies"]`); con `anio` debe ser único.
    `previa`: tasas de años anteriores a la ventana (columnas `llave_entidad`, `anio`, `objetivo`), opcional.
    """
    if df.duplicated(llave_entidad + ["anio"]).any():
        raise ValueError(f"{llave_entidad + ['anio']} no es único: la historia no está definida")
    out = df.copy()
    panel = _panel(out, llave_entidad, objetivo, previa)
    tasas = panel[llave_entidad + ["anio", objetivo]]
    for k in (1, 2):  # la tasa del año t-k de la misma entidad: se desplaza su año k posiciones hacia adelante
        anterior = tasas.assign(anio=tasas["anio"] + k).rename(columns={objetivo: f"tasa_t{k}"})
        out = out.merge(anterior, on=llave_entidad + ["anio"], how="left", validate="one_to_one")
    out["tasa_media_previa"], out["n_anios_previos"] = _hasta(out, panel, llave_entidad, 1)
    out["falta_historia"] = out["tasa_t1"].isna()
    out["tasa_media_rezagada"], out["n_anios_rezagados"] = _hasta(out, panel, llave_entidad, 2)
    out["falta_historia_rezagada"] = out["tasa_t2"].isna()
    return out


def clasificar_sin_historia(df: pd.DataFrame, anual_sin_minimo: pd.DataFrame, minimo: float) -> pd.DataFrame:
    """Para cada fila sin `tasa_t1`, el motivo. `anual_sin_minimo`: tasas anuales de SPADIES sin denominador mínimo
    (columnas `cod_ies`, `anio`, `matriculados_t2_spadies`); `minimo`: el denominador mínimo del pipeline.

    - «IES nueva»: SPADIES no tiene ninguna tasa anual de esa IES en años anteriores (su serie empieza ese año).
    - «t-1 con denominador menor que el mínimo»: había tasa de t-1 pero con denominador < `minimo`.
    - «t-1 excluida por calidad»: la tasa de t-1 existe con denominador suficiente, pero esa fila no está en el dataset
      (posible cierre, inconsistencia entre fuentes, régimen especial) o no se usó.
    - «sin tasa anual en t-1»: hay años anteriores pero no el t-1 (falta un semestre o la IES dejó de reportar un año).
    """
    sin = df.loc[df["falta_historia"], ["cod_ies", "anio", "n_anios_previos"]].copy()
    primero = anual_sin_minimo.groupby("cod_ies")["anio"].min()
    den = anual_sin_minimo.set_index(["cod_ies", "anio"])["matriculados_t2_spadies"]

    def motivo(fila) -> str:
        primer = primero.get(fila["cod_ies"])
        if primer is None or primer >= fila["anio"]:
            return "IES nueva: su primera tasa anual de SPADIES es de este año"
        d = den.get((fila["cod_ies"], fila["anio"] - 1))
        if d is None:
            return "sin tasa anual en t-1 (falta un semestre o la IES no reportó)"
        if d < minimo:
            return "t-1 con denominador menor que el mínimo"
        return "t-1 excluida por calidad (posible cierre, inconsistencia o régimen especial)"

    sin["motivo"] = sin.apply(motivo, axis=1)
    sin["primer_anio_spadies"] = sin["cod_ies"].map(primero)
    return sin.sort_values(["motivo", "cod_ies", "anio"]).reset_index(drop=True)
