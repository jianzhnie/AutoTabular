import warnings
from typing import Dict, Optional, Tuple, Union

import numpy as np
from autotabular.pipeline.base import DATASET_PROPERTIES_TYPE, PIPELINE_DATA_DTYPE
from autotabular.pipeline.components.base import AutotabularPreprocessingAlgorithm
from autotabular.pipeline.constants import DENSE, INPUT, SPARSE, UNSIGNED_DATA
from category_encoders import TargetEncoder
from category_encoders.wrapper import PolynomialWrapper
from ConfigSpace.configuration_space import ConfigurationSpace


class PolynomialWrapperTransformer(AutotabularPreprocessingAlgorithm):

    def __init__(
        self,
        feature_encoder=TargetEncoder(),
        random_state: Optional[np.random.RandomState] = None,
    ):
        self.feature_encoder = feature_encoder
        self.random_state = random_state

    def fit(self,
            X: PIPELINE_DATA_DTYPE,
            y: Optional[PIPELINE_DATA_DTYPE] = None
            ) -> 'PolynomialWrapperTransformer':

        self.preprocessor = PWrapper(feature_encoder=self.feature_encoder)
        self.preprocessor.fit(X, y)
        return self

    def transform(self, X: PIPELINE_DATA_DTYPE) -> PIPELINE_DATA_DTYPE:
        if self.preprocessor is None:
            raise NotImplementedError()

        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None
    ) -> Dict[str, Optional[Union[str, int, bool, Tuple]]]:
        return {
            'shortname': 'PolynomialWrapperTransformer',
            'name': 'PolynomialWrapper Transformer',
            'handles_regression': False,
            'handles_classification': True,
            'handles_multiclass': True,
            'handles_multilabel': True,
            'handles_multioutput': True,
            # TODO find out of this is right!
            'handles_sparse': True,
            'handles_dense': True,
            'input': (DENSE, SPARSE, UNSIGNED_DATA),
            'output': (INPUT, ),
        }

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
    ) -> ConfigurationSpace:
        return ConfigurationSpace()


class PWrapper(object):

    def __init__(self, feature_encoder=TargetEncoder()):
        self.wp = PolynomialWrapper(feature_encoder=feature_encoder)

    def fit(self, X, y):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.wp.fit(X, y)

    def transform(self, X):
        return self.wp.transform(X)
