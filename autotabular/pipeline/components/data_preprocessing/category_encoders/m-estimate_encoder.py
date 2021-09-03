import warnings
from typing import Dict, Optional, Tuple, Union

import numpy as np
from autotabular.pipeline.base import DATASET_PROPERTIES_TYPE, PIPELINE_DATA_DTYPE
from autotabular.pipeline.components.base import AutotabularPreprocessingAlgorithm
from autotabular.pipeline.constants import DENSE, INPUT, SPARSE, UNSIGNED_DATA
from category_encoders import MEstimateEncoder
from ConfigSpace.configuration_space import ConfigurationSpace


class MEstimateEncoderTransformer(AutotabularPreprocessingAlgorithm):

    def __init__(
        self,
        cols=None,
        random_state: Optional[np.random.RandomState] = None,
    ):
        self.cols = cols
        self.random_state = random_state

    def fit(self,
            X: PIPELINE_DATA_DTYPE,
            y: Optional[PIPELINE_DATA_DTYPE] = None
            ) -> 'MEstimateEncoderTransformer':

        self.preprocessor = MEEncoder(cols=self.cols)
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
            'shortname': 'MEstimateEncoderTransformer',
            'name': 'MEstimateEncoder Transformer',
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


class MEEncoder(object):

    def __init__(self, cols=None):
        self.enc = MEstimateEncoder(
            cols=cols,
            verbose=1,
            drop_invariant=False,
            return_df=True,
            handle_unknown='value',
            handle_missing='value',
            random_state=None,
            randomized=False,
            sigma=0.05,
            m=1.0)

    def fit(self, X, y):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.enc.fit(X, y)

    def transform(self, X):
        return self.enc.transform(X)
