from typing import Dict, Optional, Tuple, Union

import numpy as np
from autotabular.pipeline.base import DATASET_PROPERTIES_TYPE
from autotabular.pipeline.components.base import AutotabularPreprocessingAlgorithm
from autotabular.pipeline.components.data_preprocessing.rescaling.abstract_rescaling import Rescaling
from autotabular.pipeline.constants import DENSE, INPUT, SIGNED_DATA, UNSIGNED_DATA


class MinMaxScalerComponent(Rescaling, AutotabularPreprocessingAlgorithm):

    def __init__(self, random_state: Optional[np.random.RandomState] = None):
        from sklearn.preprocessing import MinMaxScaler
        self.preprocessor = MinMaxScaler(copy=False)

    @staticmethod
    def get_properties(
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None
    ) -> Dict[str, Optional[Union[str, int, bool, Tuple]]]:
        return {
            'shortname': 'MinMaxScaler',
            'name': 'MinMaxScaler',
            'handles_missing_values': False,
            'handles_nominal_values': False,
            'handles_numerical_features': True,
            'prefers_data_scaled': False,
            'prefers_data_normalized': False,
            'handles_regression': True,
            'handles_classification': True,
            'handles_multiclass': True,
            'handles_multilabel': True,
            'handles_multioutput': True,
            'is_deterministic': True,
            # TODO find out if this is right!
            'handles_sparse': False,
            'handles_dense': True,
            'input': (DENSE, UNSIGNED_DATA),
            'output': (INPUT, SIGNED_DATA),
            'preferred_dtype': None
        }
