from typing import Dict, Optional, Tuple, Union

import numpy as np
from autotabular.pipeline.base import DATASET_PROPERTIES_TYPE
from autotabular.pipeline.components.base import AutotabularPreprocessingAlgorithm
from autotabular.pipeline.constants import COLUMNNAME_POSTFIX_DISCRETE, DATATYPE_LABEL, DENSE, INPUT, SPARSE, UNSIGNED_DATA
from ConfigSpace.configuration_space import ConfigurationSpace
from sklearn.preprocessing import KBinsDiscretizer


class MultiKBinsDiscretizer(AutotabularPreprocessingAlgorithm):

    def __init__(self,
                 columns=None,
                 bins=None,
                 strategy='quantile',
                 random_state: Optional[np.random.RandomState] = None):
        super(MultiKBinsDiscretizer, self).__init__()

        print(f'{len(columns)} variables to discrete.')
        self.columns = columns
        self.bins = bins
        self.strategy = strategy
        self.new_columns = []
        self.encoders = {}

    def fit(self, X, y=None):
        self.new_columns = []
        if self.columns is None:
            self.columns = X.columns.tolist()
        for col in self.columns:
            new_name = col + COLUMNNAME_POSTFIX_DISCRETE
            n_unique = X.loc[:, col].nunique()
            # n_null = X.loc[:, col].isnull().sum()
            c_bins = self.bins
            if c_bins is None or c_bins <= 0:
                c_bins = round(n_unique**0.25) + 1
            encoder = KBinsDiscretizer(
                n_bins=c_bins, encode='ordinal', strategy=self.strategy)
            self.new_columns.append((col, new_name, encoder.n_bins))
            encoder.fit(X[[col]])
            self.encoders[col] = encoder
        return self

    def transform(self, X):
        for col in self.columns:
            new_name = col + COLUMNNAME_POSTFIX_DISCRETE
            encoder = self.encoders[col]
            nc = encoder.transform(X[[col]]).astype(DATATYPE_LABEL).reshape(-1)
            X[new_name] = nc
        return X

    @staticmethod
    def get_properties(
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None
    ) -> Dict[str, Optional[Union[str, int, bool, Tuple]]]:
        return {
            'shortname': 'MultiKBinsDiscretizer',
            'name': 'MultiKBins Discretizer',
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
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
    ) -> ConfigurationSpace:
        return ConfigurationSpace()
