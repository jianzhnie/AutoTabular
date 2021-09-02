from typing import Optional

import numpy as np
from autotabular.pipeline.base import DATASET_PROPERTIES_TYPE, PIPELINE_DATA_DTYPE
from autotabular.pipeline.components.base import AutotabularPreprocessingAlgorithm
from ConfigSpace.configuration_space import ConfigurationSpace
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError


class Rescaling(object):
    # Rescaling does not support fit_transform (as of 0.19.1)!
    def __init__(self, random_state: Optional[np.random.RandomState] = None):
        self.preprocessor: Optional[BaseEstimator] = None

    def fit(
        self,
        X: PIPELINE_DATA_DTYPE,
        y: Optional[PIPELINE_DATA_DTYPE] = None
    ) -> 'AutotabularPreprocessingAlgorithm':
        if self.preprocessor is None:
            raise NotFittedError()
        self.preprocessor.fit(X)
        return self

    def transform(self, X: PIPELINE_DATA_DTYPE) -> PIPELINE_DATA_DTYPE:
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        return cs
