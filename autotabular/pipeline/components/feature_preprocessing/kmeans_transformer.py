import numpy as np
from autotabular.pipeline.components.base import AutotabularPreprocessingAlgorithm
from autotabular.pipeline.constants import DENSE, UNSIGNED_DATA
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler


class KMeansTransformer(AutotabularPreprocessingAlgorithm):

    def __init__(self, k_fold, random_state=None):
        self.k_fold = k_fold

        self.random_state = random_state

    def fit(self, X, Y=None):

        self.preprocessor = KMeansTransformerOriginal(k_fold=self.k_fold)
        self.preprocessor.fit(X, Y)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            'shortname': 'KMeansTransformer',
            'name': 'KMeans Transformer To generate new features',
            'handles_regression': True,
            'handles_classification': True,
            'handles_multiclass': True,
            'handles_multilabel': True,
            'handles_multioutput': True,
            # TODO document that we have to be very careful
            'is_deterministic': False,
            'input': (DENSE, UNSIGNED_DATA),
            'output': (DENSE, UNSIGNED_DATA)
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        k_fold = UniformIntegerHyperparameter(
            'k_fold', lower=2, upper=8, default_value=6)
        cs = ConfigurationSpace()
        cs.add_hyperparameters([k_fold])
        return cs


class KMeansTransformerOriginal(object):
    """KMeansTransformerOriginal.

    reference:
        https://github.com/mljar/mljar-supervised
    """

    def __init__(self, k_fold=None):
        self._new_features = []
        self._input_columns = []
        self._error = None
        self._kmeans = None
        self._scale = None
        self._k_fold = k_fold

    def fit(self, X, y):
        if self._new_features:
            return
        if self._error is not None and self._error:
            raise Exception(
                'KMeans Features not created due to error (please check errors.md). '
                + self._error)
            return
        if X.shape[1] == 0:
            self._error = f'KMeans not created. No continous features. Input data shape: {X.shape}, {y.shape}'
            raise Exception(
                'KMeans Features not created. No continous features.')

        n_clusters = int(np.log10(X.shape[0]) * 8)
        n_clusters = max(8, n_clusters)
        n_clusters = min(n_clusters, X.shape[1])

        self._input_columns = X.columns.tolist()
        # scale data
        self._scale = StandardScaler(copy=True, with_mean=True, with_std=True)
        X = self._scale.fit_transform(X)

        # Kmeans
        self._kmeans = MiniBatchKMeans(n_clusters=n_clusters, init='k-means++')
        self._kmeans.fit(X)
        self._create_new_features_names()

    def _create_new_features_names(self):
        n_clusters = self._kmeans.cluster_centers_.shape[0]
        self._new_features = [f'Dist_Cluster_{i}' for i in range(n_clusters)]
        self._new_features += ['Cluster']

    def transform(self, X):
        if self._kmeans is None:
            raise Exception('KMeans not fitted')

        # scale
        X_scaled = self._scale.transform(X[self._input_columns])

        # kmeans
        distances = self._kmeans.transform(X_scaled)
        clusters = self._kmeans.predict(X_scaled)

        X[self._new_features[:-1]] = distances
        X[self._new_features[-1]] = clusters

        return X
