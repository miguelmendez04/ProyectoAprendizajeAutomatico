"""Métodos del profesor: KernelPCA, autoencoder y clusters (ajuste solo con entrenamiento, dimensiones, semillas)."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.decomposition import PCA

from src.modelado.metodos import AgregarClusters, AutoencoderPequeno, KernelPCAComoPCA


def _matriz(n: int = 150, d: int = 12, semilla: int = 0) -> np.ndarray:
    rng = np.random.default_rng(semilla)
    base = rng.normal(size=(n, 4))
    return base @ rng.normal(size=(4, d)) + 0.1 * rng.normal(size=(n, d))  # rango efectivo ~4


def _blobs(n_por_grupo: int = 40, centros=((0, 0), (6, 0), (0, 6), (6, 6)), semilla: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    filas = [rng.normal(c, 0.5, size=(n_por_grupo, 2)) for c in centros]
    df = pd.DataFrame(np.vstack(filas), columns=["x1", "x2"])
    df["sector"] = np.where(df.index % 2 == 0, "oficial", "privada")
    df["falta_saber11"] = (np.arange(len(df)) % 3 == 0).astype(int)
    return df


# ---------- KernelPCA ----------
def test_kernel_pca_tiene_tantos_componentes_como_pca_al_90_del_entrenamiento():
    X = _matriz()
    entrena, prueba = X[:100], X[100:]
    k = KernelPCAComoPCA(varianza=0.90, random_state=0).fit(entrena)
    esperado = PCA(n_components=0.90, svd_solver="full").fit(entrena).n_components_
    assert k.n_componentes_ == esperado
    assert k.transform(entrena).shape == (100, esperado) and k.transform(prueba).shape == (50, esperado)


def test_kernel_pca_se_ajusta_solo_con_el_entrenamiento():
    X = _matriz()
    entrena = X[:100]
    prueba_distinta = X[100:] + 1000.0
    k = KernelPCAComoPCA(random_state=0).fit(entrena)
    assert k.kpca_.X_fit_.shape == entrena.shape and np.allclose(k.kpca_.X_fit_, entrena)  # el kernel solo ve el entrenamiento
    k.transform(prueba_distinta)  # transformar la prueba no cambia lo ajustado
    assert np.allclose(k.kpca_.X_fit_, entrena)


def test_kernel_pca_gamma_configurable_y_reproducible():
    X = _matriz()
    a = KernelPCAComoPCA(gamma=0.5, random_state=0).fit(X)
    b = KernelPCAComoPCA(gamma=0.5, random_state=0).fit(X)
    c = KernelPCAComoPCA(gamma=0.01, random_state=0).fit(X)
    assert a.kpca_.gamma == 0.5 and np.allclose(a.transform(X), b.transform(X))
    assert not np.allclose(a.transform(X)[:, :3], c.transform(X)[:, :3])


# ---------- autoencoder ----------
def test_autoencoder_devuelve_el_cuello_de_botella_y_su_arquitectura_es_n_16_8_16_n():
    X = _matriz(d=12)
    ae = AutoencoderPequeno(oculta=16, cuello=8, max_iter=200, random_state=0).fit(X)
    assert ae.transform(X).shape == (150, 8)
    formas = [w.shape for w in ae.mlp_.coefs_]
    assert formas == [(12, 16), (16, 8), (8, 16), (16, 12)]


def test_el_cuello_de_botella_es_coherente_con_la_reconstruccion_del_mlp():
    """Si se sigue el cálculo desde el cuello hasta la salida se obtiene la predicción (reconstrucción) del MLP."""
    X = _matriz()
    ae = AutoencoderPequeno(max_iter=100, random_state=0).fit(X)
    h = ae.transform(X)
    capa3 = np.tanh(h @ ae.mlp_.coefs_[2] + ae.mlp_.intercepts_[2])
    salida = capa3 @ ae.mlp_.coefs_[3] + ae.mlp_.intercepts_[3]
    assert np.allclose(salida, ae.mlp_.predict(X))


def test_el_autoencoder_usa_parada_temprana_con_validacion_interna_y_semilla():
    X = _matriz()
    ae = AutoencoderPequeno(max_iter=2000, validation_fraction=0.2, random_state=3).fit(X)
    assert ae.mlp_.early_stopping and ae.mlp_.validation_fraction == 0.2 and ae.mlp_.validation_scores_ is not None
    assert ae.paro_temprano_ and ae.n_iter_ < 2000
    otro = AutoencoderPequeno(max_iter=2000, validation_fraction=0.2, random_state=3).fit(X)
    assert np.allclose(ae.transform(X), otro.transform(X))  # misma semilla, mismo resultado
    distinta = AutoencoderPequeno(max_iter=2000, validation_fraction=0.2, random_state=4).fit(X)
    assert not np.allclose(ae.transform(X), distinta.transform(X))


def test_el_autoencoder_se_ajusta_solo_con_el_entrenamiento_y_se_puede_clonar():
    X = _matriz()
    entrena = X[:100]
    ae = AutoencoderPequeno(max_iter=50, random_state=0).fit(entrena)
    antes = [w.copy() for w in ae.mlp_.coefs_]
    ae.transform(X[100:] + 500.0)  # transformar la prueba no toca los pesos
    assert all(np.array_equal(a, b) for a, b in zip(antes, ae.mlp_.coefs_))
    assert ae.n_features_in_ == entrena.shape[1] and isinstance(clone(ae), AutoencoderPequeno)


def test_activacion_invalida_falla():
    with pytest.raises(ValueError, match="activacion"):
        AutoencoderPequeno(activacion="seno").fit(_matriz())


# ---------- clusters ----------
def test_k_se_elige_por_silueta_entre_3_y_8():
    df = _blobs()
    ag = AgregarClusters(["x1", "x2", "falta_saber11"], [], k_min=3, k_max=8, random_state=0).fit(df)  # solo numéricas: el sector alterno no agrupa
    assert ag.k_ == 4  # los datos tienen 4 grupos bien separados
    assert set(ag.siluetas_) == set(range(3, 9)) and max(ag.siluetas_, key=ag.siluetas_.get) == 4


def test_los_clusters_se_agregan_como_one_hot_y_distancias():
    df = _blobs()
    ag = AgregarClusters(["x1", "x2", "falta_saber11"], ["sector"], random_state=0).fit(df)
    salida = ag.transform(df)
    k = ag.k_
    assert salida.shape == (len(df), 2 * k)
    onehot, distancias = salida[:, :k], salida[:, k:]
    assert np.array_equal(onehot.sum(axis=1), np.ones(len(df))) and (distancias >= 0).all()
    assert np.array_equal(distancias.argmin(axis=1), onehot.argmax(axis=1))  # el cluster es el del centroide más cercano


def test_el_clustering_no_usa_los_indicadores_falta_y_si_la_categoria():
    df = _blobs()
    ag = AgregarClusters(["x1", "x2", "falta_saber11"], ["sector"], random_state=0).fit(df)
    numericas_usadas = ag.pre_.transformers_[0][2]
    assert "falta_saber11" not in numericas_usadas and numericas_usadas == ["x1", "x2"]
    assert ag.pre_.transformers_[1][2] == ["sector"]


def test_los_clusters_se_ajustan_solo_con_el_entrenamiento():
    df = _blobs()
    entrena, prueba = df.iloc[:120], df.iloc[120:].copy()
    prueba[["x1", "x2"]] = 1000.0  # prueba extrema: si participara, movería los centroides
    ag = AgregarClusters(["x1", "x2"], ["sector"], random_state=0).fit(entrena)
    assert ag.n_entrenamiento_ == 120 and ag.kmeans_.labels_.shape[0] == 120
    centros = ag.kmeans_.cluster_centers_.copy()
    ag.transform(prueba)
    assert np.array_equal(centros, ag.kmeans_.cluster_centers_)
    assert ag.pre_.transformers_[0][1].named_steps["escalar"].mean_ == pytest.approx(entrena[["x1", "x2"]].mean().to_numpy())


def test_los_clusters_son_reproducibles_con_semilla_y_toleran_categorias_nuevas():
    df = _blobs()
    a = AgregarClusters(["x1", "x2"], ["sector"], random_state=5).fit(df).transform(df)
    b = AgregarClusters(["x1", "x2"], ["sector"], random_state=5).fit(df).transform(df)
    assert np.allclose(a, b)
    ag = AgregarClusters(["x1", "x2"], ["sector"], random_state=5).fit(df)
    nueva = df.iloc[:5].assign(sector="mixta")
    assert ag.transform(nueva).shape == (5, 2 * ag.k_)
