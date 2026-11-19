"""Métodos de reducción y de estructura recomendados por el profesor, como transformers de scikit-learn.

Todos se ajustan **solo con los datos que reciba `fit`** (dentro de un Pipeline y una validación cruzada, el entrenamiento de
cada pliegue), tienen semilla y se pueden clonar:

- `KernelPCAComoPCA`: PCA con kernel RBF con el mismo número de componentes que un PCA que retiene `varianza` (90 %) de la
  varianza del entrenamiento; `gamma` configurable (`None` = 1 / n_variables).
- `AutoencoderPequeno`: autoencoder n → 16 → 8 → 16 → n, entrenado para reconstruir su entrada con parada temprana sobre una
  validación interna; devuelve el cuello de botella (8 unidades). Se construye con `MLPRegressor` de scikit-learn para no agregar
  una dependencia (PyTorch o Keras); calcula el cuello de botella con los pesos aprendidos. La validación interna de la parada
  temprana es una partición aleatoria de las filas de entrenamiento (no agrupa por IES); solo decide cuándo parar.
- `AgregarClusters`: K-means sobre las variables territoriales e institucionales imputadas y escaladas, con k elegido por silueta
  entre `k_min` y `k_max`. Devuelve el cluster (one-hot) y las distancias a los centroides, para **agregarlos** como variables nuevas
  (no reemplazan a las demás). Recibe la tabla cruda (DataFrame) y hace su propia imputación y escalado.

LDA no se implementa: es un método lineal y supervisado para **clases** (maximiza la separación entre clases), y aquí el objetivo
es una tasa continua. Solo aplicaría discretizando la tasa (p. ej., en terciles), lo que pierde información del objetivo y cambia el
problema de regresión en uno de clasificación.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, KernelPCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_array

PREFIJO_INDICADORES = "falta_"  # los indicadores de bloques de años no describen a la institución: no entran al clustering


class KernelPCAComoPCA(TransformerMixin, BaseEstimator):
    """KernelPCA (RBF) con tantos componentes como un PCA que retiene `varianza` de la varianza del entrenamiento."""

    def __init__(self, varianza: float = 0.90, gamma: float | None = None, random_state: int = 0):
        self.varianza = varianza
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None):
        X = check_array(X, dtype=float)
        self.n_componentes_ = int(
            PCA(n_components=self.varianza, svd_solver="full").fit(X).n_components_
        )
        self.kpca_ = KernelPCA(
            n_components=self.n_componentes_,
            kernel="rbf",
            gamma=self.gamma,
            random_state=self.random_state,
        ).fit(X)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return self.kpca_.transform(check_array(X, dtype=float))


def _activar(h: np.ndarray, nombre: str) -> np.ndarray:
    if nombre == "tanh":
        return np.tanh(h)
    if nombre == "relu":
        return np.maximum(h, 0.0)
    if nombre == "logistic":
        return 1.0 / (1.0 + np.exp(-h))
    raise ValueError(f"activacion inválida: {nombre!r}; debe ser tanh, relu o logistic")


class AutoencoderPequeno(TransformerMixin, BaseEstimator):
    """Autoencoder n → `oculta` → `cuello` → `oculta` → n con parada temprana; `transform` devuelve el cuello de botella."""

    def __init__(
        self,
        oculta: int = 16,
        cuello: int = 8,
        activacion: str = "tanh",
        max_iter: int = 500,
        validation_fraction: float = 0.15,
        n_iter_no_change: int = 15,
        alpha: float = 1e-4,
        random_state: int = 0,
    ):
        self.oculta = oculta
        self.cuello = cuello
        self.activacion = activacion
        self.max_iter = max_iter
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, X, y=None):
        X = check_array(X, dtype=float)
        _activar(X[:1], self.activacion)  # valida el nombre
        self.mlp_ = MLPRegressor(
            hidden_layer_sizes=(self.oculta, self.cuello, self.oculta),
            activation=self.activacion,
            solver="adam",
            alpha=self.alpha,
            early_stopping=True,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", ConvergenceWarning
            )  # llegar a max_iter sin parar antes no es un error
            self.mlp_.fit(X, X)
        self.n_iter_ = int(self.mlp_.n_iter_)
        self.paro_temprano_ = self.n_iter_ < self.max_iter
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        h = check_array(X, dtype=float)
        for peso, sesgo in zip(
            self.mlp_.coefs_[:2], self.mlp_.intercepts_[:2], strict=False
        ):  # n → oculta → cuello
            h = _activar(h @ peso + sesgo, self.activacion)
        return h


class AgregarClusters(TransformerMixin, BaseEstimator):
    """K-means (k por silueta) sobre las variables numéricas (sin los indicadores `falta_*`) y categóricas, imputadas y escaladas.

    `transform` devuelve `[cluster en one-hot (k columnas), distancias a los k centroides]` (2k columnas) para agregarlas a las demás.
    """

    def __init__(
        self,
        numericas: list[str],
        categoricas: list[str],
        k_min: int = 3,
        k_max: int = 8,
        n_init: int = 10,
        random_state: int = 0,
    ):
        self.numericas = numericas
        self.categoricas = categoricas
        self.k_min = k_min
        self.k_max = k_max
        self.n_init = n_init
        self.random_state = random_state

    def _preprocesador(self) -> ColumnTransformer:
        num = [c for c in self.numericas if not c.startswith(PREFIJO_INDICADORES)]
        pasos = [
            (
                "num",
                Pipeline(
                    [("imputar", SimpleImputer(strategy="median")), ("escalar", StandardScaler())]
                ),
                num,
            )
        ]
        if self.categoricas:
            pasos.append(
                (
                    "cat",
                    Pipeline(
                        [
                            ("imputar", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                        ]
                    ),
                    list(self.categoricas),
                )
            )
        return ColumnTransformer(pasos)

    def fit(self, X: pd.DataFrame, y=None):
        self.pre_ = self._preprocesador()
        Z = self.pre_.fit_transform(X)
        self.siluetas_ = {}
        mejor = None
        for k in range(self.k_min, self.k_max + 1):
            km = KMeans(n_clusters=k, n_init=self.n_init, random_state=self.random_state).fit(Z)
            self.siluetas_[k] = float(silhouette_score(Z, km.labels_))
            if (
                mejor is None or self.siluetas_[k] > self.siluetas_[mejor[0]]
            ):  # empate: gana el k menor
                mejor = (k, km)
        self.k_, self.kmeans_ = mejor
        self.n_entrenamiento_ = len(Z)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        Z = self.pre_.transform(X)
        etiquetas = self.kmeans_.predict(Z)
        return np.hstack([np.eye(self.k_)[etiquetas], self.kmeans_.transform(Z)])

    def etiquetas(self, X: pd.DataFrame) -> np.ndarray:
        """Cluster asignado a cada fila de `X` (para describir los clusters)."""
        return self.kmeans_.predict(self.pre_.transform(X))
