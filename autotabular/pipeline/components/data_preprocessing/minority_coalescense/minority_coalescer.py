from typing import Dict, Optional, Tuple, Union

import autotabular.pipeline.implementations.MinorityCoalescer
import numpy as np
from autotabular.pipeline.base import DATASET_PROPERTIES_TYPE, PIPELINE_DATA_DTYPE
from autotabular.pipeline.components.base import AutotabularPreprocessingAlgorithm
from autotabular.pipeline.constants import DENSE, INPUT, SPARSE, UNSIGNED_DATA
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter


class MinorityCoalescer(AutotabularPreprocessingAlgorithm):
    """Group together categories which occurence is less than a specified
    minimum fraction."""

    def __init__(self,
                 minimum_fraction: float = 0.01,
                 random_state: Optional[np.random.RandomState] = None):
        self.minimum_fraction = minimum_fraction

    def fit(self,
            X: PIPELINE_DATA_DTYPE,
            y: Optional[PIPELINE_DATA_DTYPE] = None) -> 'MinorityCoalescer':
        self.minimum_fraction = float(self.minimum_fraction)

        self.preprocessor = autotabular.pipeline.implementations.MinorityCoalescer\
            .MinorityCoalescer(minimum_fraction=self.minimum_fraction)
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
            'shortname': 'coalescer',
            'name': 'Categorical minority coalescer',
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
        cs = ConfigurationSpace()
        minimum_fraction = UniformFloatHyperparameter(
            'minimum_fraction',
            lower=.0001,
            upper=0.5,
            default_value=0.01,
            log=True)
        cs.add_hyperparameter(minimum_fraction)
        return cs
