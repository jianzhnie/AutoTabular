import os
import json
import warnings
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Union
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

import numpy as np
from autotabular.pipeline.base import DATASET_PROPERTIES_TYPE, PIPELINE_DATA_DTYPE
from autotabular.pipeline.components.base import AutotabularPreprocessingAlgorithm
from autotabular.pipeline.constants import DENSE, SPARSE, UNSIGNED_DATA, INPUT



class NumericalImputation(AutotabularPreprocessingAlgorithm):

    def __init__(self, strategy: str = 'mean',
                 random_state: Optional[np.random.RandomState] = None):
        self.strategy = strategy
        self.random_state = random_state

    def fit(self, X: PIPELINE_DATA_DTYPE,
            y: Optional[PIPELINE_DATA_DTYPE] = None) -> 'NumericalImputation':
        import sklearn.impute

        self.preprocessor = sklearn.impute.SimpleImputer(
            strategy=self.strategy, copy=False)
        self.preprocessor.fit(X)
        return self

    def transform(self, X: PIPELINE_DATA_DTYPE) -> PIPELINE_DATA_DTYPE:
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)



class ExcludeRowsMissingTarget(object):
    @staticmethod
    def transform(X=None, y=None, sample_weight=None, warn=False):
        if y is None:
            return X, y, sample_weight
        y_missing = pd.isnull(y)
        if np.sum(np.array(y_missing)) == 0:
            return X, y, sample_weight
        logger.debug("Exclude rows with missing target values")
        if warn:
            warnings.warn(
                "There are samples with missing target values in the data which will be excluded for further analysis"
            )
        y = y.drop(y.index[y_missing])
        y.reset_index(drop=True, inplace=True)

        if X is not None:
            X = X.drop(X.index[y_missing])
            X.reset_index(drop=True, inplace=True)

        if sample_weight is not None:
            sample_weight = sample_weight.drop(sample_weight.index[y_missing])
            sample_weight.reset_index(drop=True, inplace=True)

        return X, y, sample_weight
