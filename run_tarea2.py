"""Corre la Tarea 2 (estimación sin historia): cribado, confirmación, clusters, informe y figuras.

Uso: python run_tarea2.py [--config RUTA]
Salidas (rutas en config.yaml: modelado.tarea2.salidas): reports/resultados/experimentos_tarea2_cribado.csv,
..._confirmacion.csv, reports/resultados_tarea2.md, reports/figuras/tarea2_metodologias.png y reports/figuras/clusters.png.
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import sklearn
import xgboost

from src.config import load_config
from src.modelado import tarea2 as t2
from src.modelado.experimentos import parametros_experimentos
from src.modelado.informe_tarea2 import describir_clusters, escribir_informe, figura_clusters, figura_metodologias
from src.modelado.particion import cargar_dataset

SALIDAS = {"cribado": "reports/resultados/experimentos_tarea2_cribado.csv", "confirmacion": "reports/resultados/experimentos_tarea2_confirmacion.csv",
           "resumen_confirmacion": "reports/resultados/experimentos_tarea2_confirmacion_resumen.csv", "clusters": "reports/resultados/clusters_tarea2.csv",
           "informe": "reports/resultados_tarea2.md", "figura_metodologias": "reports/figuras/tarea2_metodologias.png",
           "figura_clusters": "reports/figuras/clusters.png"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--solo-informe", action="store_true", help="no repite el cribado ni la confirmación: lee los CSV y los tiempos guardados")
    args = ap.parse_args()
    t0 = time.perf_counter()
    cfg = load_config(args.config)
    p_exp, p = parametros_experimentos(cfg), t2.parametros_tarea2(cfg)
    df, X, y, numericas, categoricas = t2.preparar_datos(cfg, cargar_dataset(cfg))
    salidas = {k: Path(v) for k, v in {**SALIDAS, **p.get("salidas", {})}.items()}
    salidas["cribado"].parent.mkdir(parents=True, exist_ok=True)

    tiempos = salidas["cribado"].with_name("experimentos_tarea2_tiempos.json")
    if args.solo_informe:
        cribado, conf = pd.read_csv(salidas["cribado"]), pd.read_csv(salidas["confirmacion"])
        guardados = json.loads(tiempos.read_text())
        t_cribado, t_conf, t_total_guardado = guardados["t_cribado_s"], guardados["t_confirmacion_s"], guardados.get("t_total_s")
        r = t2.resumen_cribado(cribado)
        nombres = list(dict.fromkeys(conf.loc[conf["tipo"] == "modelo", "configuracion"]))
    else:
        # a) cribado: grilla completa, una semilla
        t = time.perf_counter()
        cribado = t2.cribado(cfg, df, verbose=True)
        t_cribado = time.perf_counter() - t
        cribado.to_csv(salidas["cribado"], index=False)
        r = t2.resumen_cribado(cribado)

        # b) confirmación: línea base + mejor de cada modelo y de cada reducción, semillas distintas
        elegidas = t2.seleccion_confirmacion(r)
        nombres = [t2.nombre_configuracion(c) for c in elegidas]
        print("\nElegidas para la confirmación:", *nombres, sep="\n  ", flush=True)
        t = time.perf_counter()
        conf = t2.confirmacion(cfg, elegidas, df, verbose=True)
        t_conf = time.perf_counter() - t
        conf.to_csv(salidas["confirmacion"], index=False)
        tiempos.write_text(json.dumps({"t_cribado_s": t_cribado, "t_confirmacion_s": t_conf}), encoding="utf-8")
    cr = t2.resumen_confirmacion(conf, p["nivel_confianza"])
    cr.to_csv(salidas["resumen_confirmacion"], index=False)

    # clusters: descripción con todas las filas (solo para describir)
    ag, df_cl, etiquetas = t2.ajustar_clusters(cfg, df)
    desc, medias = describir_clusters(df_cl, X, etiquetas, numericas)
    desc.merge(medias.reset_index(), on="cluster").to_csv(salidas["clusters"], index=False)
    figura_clusters(ag, X, etiquetas, desc, salidas["figura_clusters"])
    figura_metodologias(r, salidas["figura_metodologias"])

    from sklearn.decomposition import PCA
    from src.modelado.particion import construir_preprocesador
    n_pca = int(PCA(n_components=p_exp["varianza_pca"], svd_solver="full").fit(construir_preprocesador(numericas, categoricas).fit_transform(X)).n_components_)
    total = (t_total_guardado or t_cribado + t_conf) if args.solo_informe else time.perf_counter() - t0
    if not args.solo_informe:
        tiempos.write_text(json.dumps({"t_cribado_s": t_cribado, "t_confirmacion_s": t_conf, "t_total_s": total}), encoding="utf-8")
    meta = {
        "filas": len(df), "ies": int(df["cod_ies"].nunique()), "n_numericas": len(numericas), "k_vecinos": p_exp["k_vecinos"],
        "k_info_mutua": p_exp["k_info_mutua"], "varianza_pca": p_exp["varianza_pca"], "n_pca": n_pca,
        "gamma_texto": "1 / n_variables" if p["kernel_pca"]["gamma"] is None else p["kernel_pca"]["gamma"], "ae_oculta": p["autoencoder"]["oculta"],
        "ae_cuello": p["autoencoder"]["cuello"], "ae_act": p["autoencoder"]["activacion"], "ae_val": p["autoencoder"]["validation_fraction"],
        "k_min": p["clusters"]["k_min"], "k_max": p["clusters"]["k_max"], "semilla_cribado": p["semilla_cribado"],
        "semillas_confirmacion": p["semillas_confirmacion"], "n_configuraciones": int((r["tipo"] == "modelo").sum()),
        "n_ajustes": p["n_splits"] * len(p["semillas_confirmacion"]), "t_cribado_s": t_cribado, "t_confirmacion_s": t_conf, "t_total_s": total,
        "sklearn": sklearn.__version__, "xgboost": xgboost.__version__, "k_clusters": ag.k_, "siluetas": ag.siluetas_,
    }
    diag_ae = t2.diagnostico_autoencoder(cfg, df)
    escribir_informe(cribado, r, conf, cr, desc, nombres, meta, salidas["informe"], diag_ae)

    cols = ["configuracion", "mae_media", "mae_desv"]
    print("\nLíneas base (cribado):\n" + r[r["tipo"] == "linea_base"][cols].to_string(index=False))
    print("\nTop 5 del cribado:\n" + r[r["tipo"] == "modelo"].head(5)[cols].to_string(index=False))
    print("\nConfirmación:\n" + cr[["configuracion", "mae_media", "mae_ic_inf", "mae_ic_sup", "mejora_grupo_pct", "p_wilcoxon_grupo", "p_t_grupo"]].to_string(index=False))
    print(f"\nClusters: k = {ag.k_}\n" + desc[["cluster", "nombre", "n_filas", "n_ies", "tasa_mediana"]].to_string(index=False))
    print(f"\nTiempo: cribado {t_cribado:.0f} s, confirmación {t_conf:.0f} s, total {total:.0f} s")


if __name__ == "__main__":
    main()
