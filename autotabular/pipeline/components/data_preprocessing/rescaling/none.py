from typing import Dict, Optional, Tuple, Union

from autotabular.pipeline.base import DATASET_PROPERTIES_TYPE, PIPELINE_DATA_DTYPE
from autotabular.pipeline.components.base import AutotabularPreprocessingAlgorithm
from autotabular.pipeline.components.data_preprocessing.rescaling.abstract_rescaling import Rescaling
from autotabular.pipeline.constants import DENSE, INPUT, SPARSE, UNSIGNED_DATA


class NoRescalingComponent(Rescaling, AutotabularPreprocessingAlgorithm):

    def fit(
        self,
        X: PIPELINE_DATA_DTYPE,
        y: Optional[PIPELINE_DATA_DTYPE] = None
    ) -> 'AutotabularPreprocessingAlgorithm':
        self.preprocessor = 'passthrough'
        return self

    def transform(self, X: PIPELINE_DATA_DTYPE) -> PIPELINE_DATA_DTYPE:
        return X

    @staticmethod
    def get_properties(
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None
    ) -> Dict[str, Optional[Union[str, int, bool, Tuple]]]:
        return {
            'shortname': 'NoRescaling',
            'name': 'NoRescaling',
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
            'handles_sparse': True,
            'handles_dense': True,
            'input': (SPARSE, DENSE, UNSIGNED_DATA),
            'output': (INPUT, ),
            'preferred_dtype': None
        }
