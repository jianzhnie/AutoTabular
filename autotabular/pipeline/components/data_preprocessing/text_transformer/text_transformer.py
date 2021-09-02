from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from autotabular.pipeline.base import DATASET_PROPERTIES_TYPE, PIPELINE_DATA_DTYPE
from autotabular.pipeline.components.base import AutotabularPreprocessingAlgorithm
from autotabular.pipeline.constants import DENSE, INPUT, SPARSE, UNSIGNED_DATA
from ConfigSpace.configuration_space import ConfigurationSpace
from sklearn.feature_extraction.text import TfidfVectorizer


class TextTFIDFTransformer(AutotabularPreprocessingAlgorithm):

    def __init__(self,
                 column,
                 random_state: Optional[np.random.RandomState] = None):
        self.column = column
        self.random_state = random_state

    def fit(self,
            X: PIPELINE_DATA_DTYPE,
            y: Optional[PIPELINE_DATA_DTYPE] = None) -> 'TextTFIDFTransformer':

        self.preprocessor = TfidfVectorizerTransformer()
        self.preprocessor.fit(X, self.column)
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
            'shortname': 'TextTFIDFTransformer',
            'name': 'Text TFIDF Transformer',
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


class TfidfVectorizerTransformer(object):

    def __init__(self):
        self._new_columns = []
        self._old_column = None
        self._max_features = 100
        self._vectorizer = None

    def fit(self, X, column):
        self._old_column = column
        self._vectorizer = TfidfVectorizer(
            analyzer='word',
            stop_words='english',
            lowercase=True,
            max_features=self._max_features,
        )

        x = X[column][~pd.isnull(X[column])]
        self._vectorizer.fit(x)
        for f in self._vectorizer.get_feature_names():
            new_col = self._old_column + '_' + f
            self._new_columns += [new_col]

    def transform(self, X):

        ii = ~pd.isnull(X[self._old_column])
        x = X[self._old_column][ii]
        vect = self._vectorizer.transform(x)

        for f in self._new_columns:
            X[f] = 0.0

        X.loc[ii, self._new_columns] = vect.toarray()
        X.drop(self._old_column, axis=1, inplace=True)
        return X
