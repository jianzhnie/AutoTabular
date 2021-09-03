from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from autotabular.pipeline.base import DATASET_PROPERTIES_TYPE, PIPELINE_DATA_DTYPE
from autotabular.pipeline.components.base import AutotabularPreprocessingAlgorithm
from autotabular.pipeline.constants import DENSE, INPUT, SPARSE, UNSIGNED_DATA
from ConfigSpace.configuration_space import ConfigurationSpace


class DateTimeTransformer(AutotabularPreprocessingAlgorithm):
    """DateTimeEncoding to handle date and time data."""

    def __init__(
        self,
        random_state: Optional[np.random.RandomState] = None,
    ):

        self.random_state = random_state

    def fit(self,
            X: PIPELINE_DATA_DTYPE,
            y: Optional[PIPELINE_DATA_DTYPE] = None) -> 'DateTimeTransformer':

        self.preprocessor = DateTimeTransformerOriginal()

        self.preprocessor.fit(X, y)

        return self

    def transform(self, X: PIPELINE_DATA_DTYPE) -> PIPELINE_DATA_DTYPE:
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None
    ) -> Dict[str, Optional[Union[str, int, bool, Tuple]]]:
        return {
            'shortname': 'DateTimeEncoding',
            'name': 'DateTime Encoding',
            'handles_regression': True,
            'handles_classification': True,
            'handles_multiclass': True,
            'handles_multilabel': True,
            'handles_multioutput': True,
            # TODO find out of this is right!
            'handles_sparse': False,
            'handles_dense': True,
            'input': (DENSE, SPARSE, UNSIGNED_DATA),
            'output': (INPUT, ),
        }

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
    ) -> ConfigurationSpace:
        return ConfigurationSpace()


class DateTimeTransformerOriginal(object):
    """DataTime Transformer.

    reference:
        https://github.com/mljar/mljar-supervised
    """

    def __init__(self):
        self._new_columns = []
        self._old_column = None
        self._min_datetime = None
        self._transforms = []

    def fit(self, X, column):
        self._old_column = column
        self._min_datetime = np.min(X[column])

        values = X[column].dt.year
        if len(np.unique(values)) > 1:
            self._transforms += ['year']
            new_column = column + '_Year'
            self._new_columns += [new_column]

        values = X[column].dt.month
        if len(np.unique(values)) > 1:
            self._transforms += ['month']
            new_column = column + '_Month'
            self._new_columns += [new_column]

        values = X[column].dt.day
        if len(np.unique(values)) > 1:
            self._transforms += ['day']
            new_column = column + '_Day'
            self._new_columns += [new_column]

        values = X[column].dt.weekday
        if len(np.unique(values)) > 1:
            self._transforms += ['weekday']
            new_column = column + '_WeekDay'
            self._new_columns += [new_column]

        values = X[column].dt.dayofyear
        if len(np.unique(values)) > 1:
            self._transforms += ['dayofyear']
            new_column = column + '_DayOfYear'
            self._new_columns += [new_column]

        values = X[column].dt.hour
        if len(np.unique(values)) > 1:
            self._transforms += ['hour']
            new_column = column + '_Hour'
            self._new_columns += [new_column]

        values = (X[column] - self._min_datetime).dt.days
        if len(np.unique(values)) > 1:
            self._transforms += ['days_diff']
            new_column = column + '_Days_Diff_To_Min'
            self._new_columns += [new_column]

    def transform(self, X):
        column = self._old_column

        if 'year' in self._transforms:
            new_column = column + '_Year'
            X[new_column] = X[column].dt.year

        if 'month' in self._transforms:
            new_column = column + '_Month'
            X[new_column] = X[column].dt.month

        if 'day' in self._transforms:
            new_column = column + '_Day'
            X[new_column] = X[column].dt.day

        if 'weekday' in self._transforms:
            new_column = column + '_WeekDay'
            X[new_column] = X[column].dt.weekday

        if 'dayofyear' in self._transforms:
            new_column = column + '_DayOfYear'
            X[new_column] = X[column].dt.dayofyear

        if 'hour' in self._transforms:
            new_column = column + '_Hour'
            X[new_column] = X[column].dt.hour

        if 'days_diff' in self._transforms:
            new_column = column + '_Days_Diff_To_Min'
            X[new_column] = (X[column] - self._min_datetime).dt.days

        X.drop(column, axis=1, inplace=True)
        return X

    def to_json(self):
        data_json = {
            'new_columns': list(self._new_columns),
            'old_column': self._old_column,
            'min_datetime': str(self._min_datetime),
            'transforms': list(self._transforms),
        }
        return data_json

    def from_json(self, data_json):
        self._new_columns = data_json.get('new_columns', None)
        self._old_column = data_json.get('old_column', None)
        d = data_json.get('min_datetime', None)
        self._min_datetime = None if d is None else pd.to_datetime(d)
        self._transforms = data_json.get('transforms', [])
