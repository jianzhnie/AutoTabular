import unittest

import numpy as np
from autotabular.pipeline.components.data_preprocessing.categorical_encoding.no_encoding import NoEncoding
from autotabular.pipeline.components.data_preprocessing.category_encoders.glmm_encoder import GLMMEncoderTransformer
from autotabular.pipeline.util import _test_preprocessing
from scipy import sparse


def create_X(instances=1000, n_feats=10, categs_per_feat=5, seed=0):
    rs = np.random.RandomState(seed)
    size = (instances, n_feats)
    X = rs.randint(0, categs_per_feat, size=size)
    return X


def create_Y(instances=1000, categs=5, seed=0):
    rs = np.random.RandomState(seed)
    size = (instances, 1)
    Y = rs.randint(0, categs, size=size)
    return Y


class GLMMEncoderTransformerTest(unittest.TestCase):

    def setUp(self):
        self.X_train = create_X()
        self.Y_train = create_Y()

    def test_data_type_consistency(self):
        X = np.random.randint(3, 6, (3, 4))
        y = np.random.randint(3, 6, (3, 1))
        Y = GLMMEncoderTransformer().fit_transform(X, y)
        self.assertFalse(sparse.issparse(Y))

        # X = sparse.csc_matrix(([3, 6, 4, 5], ([0, 1, 2, 1], [3, 2, 1, 0])),
        #                       shape=(3, 4))
        # y = sparse.csc_matrix(([0,1,1], ([0,1,2], [0,0,0])), shape=(3,1))
        # Y = GLMMEncoderTransformer().fit_transform(X, y)
        # self.assertTrue(sparse.issparse(Y))

    def test_default_configuration(self):
        transformations = []
        for i in range(2):
            configuration_space = GLMMEncoderTransformer.get_hyperparameter_search_space(
            )
            default = configuration_space.get_default_configuration()

            preprocessor = GLMMEncoderTransformer(
                random_state=1,
                **{
                    hp_name: default[hp_name]
                    for hp_name in default if default[hp_name] is not None
                })

            transformer = preprocessor.fit(self.X_train.copy(),
                                           self.Y_train.copy())
            Xt = transformer.transform(self.X_train.copy())
            transformations.append(Xt)
            if len(transformations) > 1:
                np.testing.assert_array_equal(transformations[-1],
                                              transformations[-2])

    def test_default_configuration_no_encoding(self):
        transformations = []
        for i in range(2):
            transformation, original = _test_preprocessing(NoEncoding)
            self.assertEqual(transformation.shape, original.shape)
            self.assertTrue((transformation == original).all())
            transformations.append(transformation)
            if len(transformations) > 1:
                self.assertTrue(
                    (transformations[-1] == transformations[-2]).all())

    # def test_default_configuration_sparse_data(self):
    #     transformations = []

    #     self.X_train[~np.isfinite(self.X_train)] = 0
    #     self.X_train = sparse.csc_matrix(self.X_train)

    #     for i in range(2):
    #         configuration_space = GLMMEncoderTransformer.get_hyperparameter_search_space(
    #         )
    #         default = configuration_space.get_default_configuration()

    #         preprocessor = GLMMEncoderTransformer(
    #             random_state=1,
    #             **{
    #                 hp_name: default[hp_name]
    #                 for hp_name in default if default[hp_name] is not None
    #             })

    #         transformer = preprocessor.fit(self.X_train.copy())
    #         Xt = transformer.transform(self.X_train.copy())
    #         transformations.append(Xt)
    #         if len(transformations) > 1:
    #             self.assertEqual((transformations[-1] !=
    #                               transformations[-2]).count_nonzero(), 0)

    def test_default_configuration_sparse_no_encoding(self):
        transformations = []

        for i in range(2):
            transformation, original = _test_preprocessing(
                NoEncoding, make_sparse=True)
            self.assertEqual(transformation.shape, original.shape)
            self.assertTrue(
                (transformation.todense() == original.todense()).all())
            transformations.append(transformation)
            if len(transformations) > 1:
                self.assertEqual((transformations[-1] !=
                                  transformations[-2]).count_nonzero(), 0)
