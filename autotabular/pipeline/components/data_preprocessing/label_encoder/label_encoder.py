from decimal import Decimal
from typing import Dict, Optional, Tuple, Union

import numpy as np
from autotabular.pipeline.base import DATASET_PROPERTIES_TYPE, PIPELINE_DATA_DTYPE
from autotabular.pipeline.components.base import AutotabularPreprocessingAlgorithm
from autotabular.pipeline.constants import DENSE, INPUT, SPARSE, UNSIGNED_DATA
from ConfigSpace.configuration_space import ConfigurationSpace


class LabelEncoder(object):

    def __init__(self, try_to_fit_numeric=False):
        from sklearn.preprocessing import LabelEncoder
        self.lbl = LabelEncoder()
        self._try_to_fit_numeric = try_to_fit_numeric

    def fit(self, x):
        self.lbl.fit(x)  # list(x.values))
        if self._try_to_fit_numeric:
            try:
                arr = {Decimal(c): c for c in self.lbl.classes_}
                sorted_arr = dict(sorted(arr.items()))
                self.lbl.classes_ = np.array(
                    list(sorted_arr.values()), dtype=self.lbl.classes_.dtype)
            except Exception as e:
                print(e)
                pass

    def transform(self, x):
        try:
            return self.lbl.transform(x)  # list(x.values))
        except ValueError as ve:
            print(ve)
            # rescue
            classes = np.unique(x)  # list(x.values))
            diff = np.setdiff1d(classes, self.lbl.classes_)
            self.lbl.classes_ = np.concatenate((self.lbl.classes_, diff))
            return self.lbl.transform(x)  # list(x.values))

    def inverse_transform(self, x):
        return self.lbl.inverse_transform(x)  # (list(x.values))


class LabelEncoderTransformer(AutotabularPreprocessingAlgorithm):

    def __init__(
        self,
        random_state: Optional[np.random.RandomState] = None,
    ):
        self.random_state = random_state

    def fit(
        self,
        X: Optional[PIPELINE_DATA_DTYPE] = None,
        y: PIPELINE_DATA_DTYPE = None,
    ) -> 'LabelEncoderTransformer':
        self.preprocessor = LabelEncoder(try_to_fit_numeric=False)
        self.preprocessor.fit(y)
        return self

    def transform(self, X: PIPELINE_DATA_DTYPE) -> PIPELINE_DATA_DTYPE:
        if self.preprocessor is None:
            raise NotImplementedError()
        # Notice we are shifting the unseen categories during fit to 1
        # from -1, 0, ... to 0,..., cat + 1
        # This is done because Category shift requires non negative integers
        # Consider removing this if that step is removed
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None
    ) -> Dict[str, Optional[Union[str, int, bool, Tuple]]]:
        return {
            'shortname': 'LabelEncoderTransformer',
            'name': 'LabelEncoder Transformer',
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
