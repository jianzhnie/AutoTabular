from typing import Dict, Optional, Tuple, Union

import numpy as np
import scipy.sparse
from autotabular.pipeline.base import DATASET_PROPERTIES_TYPE, PIPELINE_DATA_DTYPE
from autotabular.pipeline.components.base import AutotabularPreprocessingAlgorithm
from autotabular.pipeline.constants import DENSE, INPUT, SPARSE, UNSIGNED_DATA
from autotabular.pipeline.implementations.SparseOneHotEncoder import SparseOneHotEncoder
from ConfigSpace.configuration_space import ConfigurationSpace
from sklearn.preprocessing import OneHotEncoder as DenseOneHotEncoder


class OneHotEncoder(AutotabularPreprocessingAlgorithm):

    def __init__(self, random_state: Optional[np.random.RandomState] = None):
        self.random_state = random_state

    def fit(self,
            X: PIPELINE_DATA_DTYPE,
            y: Optional[PIPELINE_DATA_DTYPE] = None) -> 'OneHotEncoder':
        if scipy.sparse.issparse(X):
            self.preprocessor = SparseOneHotEncoder()
        else:
            self.preprocessor = DenseOneHotEncoder(
                sparse=False, categories='auto', handle_unknown='ignore')
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
            'shortname': '1Hot',
            'name': 'One Hot Encoder',
            'handles_regression': True,
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
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None
    ) -> ConfigurationSpace:
        return ConfigurationSpace()
