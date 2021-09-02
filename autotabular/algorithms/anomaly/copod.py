"""Copula Based Outlier Detector (COPOD)"""
# Author: Zheng Li <jk_zhengli@hotmail.com>
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

from __future__ import division, print_function
import warnings

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from scipy.stats import skew
from sklearn.utils import check_array
from statsmodels.distributions.empirical_distribution import ECDF

from .base import BaseDetector
from .sklearn_base import _partition_estimators


def ecdf(X):
    """Calculated the empirical CDF of a given dataset.
    Parameters
    ----------
    X : numpy array of shape (n_samples, n_features)
        The training dataset.
    Returns
    -------
    ecdf(X) : float
        Empirical CDF of X
    """
    ecdf = ECDF(X)
    return ecdf(X)


def _parallel_ecdf(n_dims, X):
    """Private method to calculate ecdf in parallel.
    Parameters
    ----------
    n_dims : int
        The number of dimensions of the current input matrix

    X : numpy array
        The subarray for building the ECDF

    Returns
    -------
    U_l_mat : numpy array
        ECDF subarray.

    U_r_mat : numpy array
        ECDF subarray.
    """
    U_l_mat = np.zeros([X.shape[0], n_dims])
    U_r_mat = np.zeros([X.shape[0], n_dims])

    for i in range(n_dims):
        U_l_mat[:, i] = ecdf(X[:, i])
        U_r_mat[:, i] = ecdf(X[:, i] * -1)
    return U_l_mat, U_r_mat


class COPOD(BaseDetector):
    """COPOD class for Copula Based Outlier Detector. COPOD is a parameter-
    free, highly interpretable outlier detection algorithm based on empirical
    copula models. See :cite:`li2020copod` for details.

    Parameters
    ----------
    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e.
        the proportion of outliers in the data set. Used when fitting to
        define the threshold on the decision function.

    n_jobs : optional (default=1)
        The number of jobs to run in parallel for both `fit` and
        `predict`. If -1, then the number of jobs is set to the
        number of cores.

    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is
        fitted.
    threshold_ : float
        The threshold is based on ``contamination``. It is the
        ``n_samples * contamination`` most abnormal samples in
        ``decision_scores_``. The threshold is calculated for generating
        binary outlier labels.
    labels_ : int, either 0 or 1
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers/anomalies. It is generated by applying
        ``threshold_`` on ``decision_scores_``.
    """

    def __init__(self, contamination=0.1, n_jobs=1):
        super(COPOD, self).__init__(contamination=contamination)
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Fit detector. y is ignored in unsupervised methods.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.
        y : Ignored
            Not used, present for API consistency by convention.
        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = check_array(X)
        self._set_n_classes(y)
        self.X_train = X
        self.decision_scores_ = self.decision_function(X)
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.
         For consistency, outliers are assigned with larger anomaly scores.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.
        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        # use multi-thread execution
        if self.n_jobs != 1:
            return self._decision_function_parallel(X)

        if hasattr(self, 'X_train'):
            original_size = X.shape[0]
            X = np.concatenate((self.X_train, X), axis=0)

        self.U_l = -1 * np.log(np.apply_along_axis(ecdf, 0, X))
        self.U_r = -1 * np.log(np.apply_along_axis(ecdf, 0, -X))

        skewness = np.sign(skew(X, axis=0))
        self.U_skew = self.U_l * -1 * np.sign(
            skewness - 1) + self.U_r * np.sign(skewness + 1)
        self.O = np.maximum(self.U_skew, np.add(self.U_l, self.U_r) / 2)
        if hasattr(self, 'X_train'):
            decision_scores_ = self.O.sum(axis=1)[-original_size:]
        else:
            decision_scores_ = self.O.sum(axis=1)
        return decision_scores_.ravel()

    def _decision_function_parallel(self, X):
        """Predict raw anomaly score of X using the fitted detector.
         For consistency, outliers are assigned with larger anomaly scores.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.
        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        if hasattr(self, 'X_train'):
            original_size = X.shape[0]
            X = np.concatenate((self.X_train, X), axis=0)

        n_samples, n_features = X.shape[0], X.shape[1]

        if n_features < 2:
            raise ValueError(
                'n_jobs should not be used on one dimensional dataset')

        if n_features <= self.n_jobs:
            self.n_jobs = n_features
            warnings.warn('n_features <= n_jobs; setting them equal instead.')

        n_jobs, n_dims_list, starts = _partition_estimators(
            n_features, self.n_jobs)

        all_results = Parallel(
            n_jobs=n_jobs, max_nbytes=None, verbose=True)(
                delayed(_parallel_ecdf)(
                    n_dims_list[i],
                    X[:, starts[i]:starts[i + 1]],
                ) for i in range(n_jobs))

        # recover the results
        self.U_l = np.zeros([n_samples, n_features])
        self.U_r = np.zeros([n_samples, n_features])

        for i in range(n_jobs):
            self.U_l[:, starts[i]:starts[i + 1]] = all_results[i][0]
            self.U_r[:, starts[i]:starts[i + 1]] = all_results[i][1]

        # self.U_l = pd.DataFrame(-1 * np.log(self.U_l))
        # self.U_r = pd.DataFrame(-1 * np.log(self.U_r))

        self.U_l = -1 * np.log(self.U_l)
        self.U_r = -1 * np.log(self.U_r)

        skewness = np.sign(skew(X, axis=0))
        self.U_skew = self.U_l * -1 * np.sign(
            skewness - 1) + self.U_r * np.sign(skewness + 1)
        self.O = np.maximum(self.U_skew, np.add(self.U_l, self.U_r) / 2)
        if hasattr(self, 'X_train'):
            decision_scores_ = self.O.sum(axis=1)[-original_size:]
        else:
            decision_scores_ = self.O.sum(axis=1)
        return decision_scores_.ravel()

    def explain_outlier(self,
                        ind,
                        columns=None,
                        cutoffs=None,
                        feature_names=None):  # pragma: no cover
        """Plot dimensional outlier graph for a given data
            point within the dataset.
        Parameters
        ----------
        ind : int
            The index of the data point one wishes to obtain
            a dimensional outlier graph for.

        columns : list
            Specify a list of features/dimensions for plotting.

        cutoffs : list of floats in (0., 1), optional (default=[0.95, 0.99])
            The significance cutoff bands of the dimensional outlier graph.

        feature_names: list of strings
            The display names of all columns of the dataset,
            to show on the x-axis of the plot.

        Returns
        -------
        Plot : matplotlib plot
            The dimensional outlier graph for data point with index ind.
        """
        if columns is None:
            columns = self.O.columns
            column_range = range(1, self.O.shape[1] + 1)
        else:
            column_range = range(1, len(columns) + 1)

        cutoffs = [1 -
                   self.contamination, 0.99] if cutoffs is None else cutoffs
        plt.plot(column_range, self.O.loc[ind, columns], label='Outlier Score')
        for i in cutoffs:
            plt.plot(
                column_range,
                self.O.loc[:, columns].quantile(q=i, axis=0),
                '-',
                label='{percentile} Cutoff Band'.format(percentile=i))
        plt.xlim([1, max(column_range)])
        plt.ylim([0, int(self.O.loc[:, columns].max().max()) + 1])
        plt.ylabel('Dimensional Outlier Score')
        plt.xlabel('Dimension')

        ticks = column_range
        if feature_names is not None:
            assert len(feature_names) == len(ticks), \
                'Length of feature_names does not match dataset dimensions.'
            plt.xticks(ticks, labels=feature_names)
        else:
            plt.xticks(ticks)

        plt.yticks(range(0, int(self.O.loc[:, columns].max().max()) + 1))
        label = 'Outlier' if self.labels_[ind] == 1 else 'Inlier'
        plt.title('Outlier Score Breakdown for Data #{index} ({label})'.format(
            index=ind + 1, label=label))
        plt.legend()
        plt.show()
        return self.O.loc[ind, columns], self.O.loc[:, columns].quantile(
            q=cutoffs[0], axis=0), self.O.loc[:, columns].quantile(
                q=cutoffs[1], axis=0)
