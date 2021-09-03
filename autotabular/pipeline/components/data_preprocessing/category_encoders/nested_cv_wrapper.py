import warnings
from typing import Dict, Optional, Tuple, Union

import numpy as np
from autotabular.pipeline.base import DATASET_PROPERTIES_TYPE, PIPELINE_DATA_DTYPE
from autotabular.pipeline.components.base import AutotabularPreprocessingAlgorithm
from autotabular.pipeline.constants import DENSE, INPUT, SPARSE, UNSIGNED_DATA
from category_encoders import TargetEncoder
from category_encoders.wrapper import NestedCVWrapper
from ConfigSpace.configuration_space import ConfigurationSpace


class NestedCVWrapperTransformer(AutotabularPreprocessingAlgorithm):

    def __init__(
        self,
        feature_encoder=TargetEncoder(),
        cv=5,
        shuffle=True,
        random_state: Optional[np.random.RandomState] = None,
    ):
        self.feature_encoder = feature_encoder
        self.random_state = random_state
        self.cv = cv
        self.shuffle = shuffle

    def fit(self,
            X: PIPELINE_DATA_DTYPE,
            y: Optional[PIPELINE_DATA_DTYPE] = None
            ) -> 'NestedCVWrapperTransformer':

        self.preprocessor = NCVWrapper(
            feature_encoder=self.feature_encoder,
            cv=self.cv,
            shuffle=self.shuffle,
            random_state=self.random_state)
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
            'shortname': 'NestedCVWrapperTransformer',
            'name': 'NestedCVWrapper Transformer',
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


class NCVWrapper(object):

    def __init__(self,
                 random_state,
                 feature_encoder=TargetEncoder(),
                 cv=5,
                 shuffle=True):
        self.wp = NestedCVWrapper(
            feature_encoder=feature_encoder,
            cv=cv,
            shuffle=True,
            random_state=random_state)

    def fit(self, X, y):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.wp.fit(X, y)

    def transform(self, X):
        return self.wp.transform(X)
