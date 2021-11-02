import lightgbm as lgb
import numpy as np
import pandas as pd
from optuna.integration.lightgbm import LightGBMTunerCV
from supervised.algorithms.lightgbm import lightgbm_eval_metric, lightgbm_objective
from supervised.algorithms.registry import MULTICLASS_CLASSIFICATION
from supervised.utils.metric import (
    lightgbm_eval_metric_accuracy, lightgbm_eval_metric_average_precision,
    lightgbm_eval_metric_f1, lightgbm_eval_metric_pearson,
    lightgbm_eval_metric_r2, lightgbm_eval_metric_spearman,
    lightgbm_eval_metric_user_defined)

EPS = 1e-8


class LightgbmObjective:

    def __init__(
        self,
        ml_task,
        X_train,
        y_train,
        sample_weight,
        X_validation,
        y_validation,
        sample_weight_validation,
        eval_metric,
        cat_features_indices,
        n_jobs,
        random_state,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.sample_weight = sample_weight
        self.X_validation = X_validation
        self.y_validation = y_validation
        self.sample_weight_validation = sample_weight_validation
        self.dtrain = lgb.Dataset(
            self.X_train.to_numpy()
            if isinstance(self.X_train, pd.DataFrame) else self.X_train,
            label=self.y_train,
            weight=self.sample_weight,
            free_raw_data=False,
        )
        self.dvalid = lgb.Dataset(
            self.X_validation.to_numpy() if isinstance(
                self.X_validation, pd.DataFrame) else self.X_validation,
            label=self.y_validation,
            weight=self.sample_weight_validation,
            free_raw_data=False,
        )

        self.cat_features_indices = cat_features_indices
        self.eval_metric = eval_metric
        self.learning_rate = 0.025
        self.rounds = 1000
        self.early_stopping_rounds = 50
        self.seed = random_state

        self.n_jobs = n_jobs
        if n_jobs == -1:
            self.n_jobs = 0

        self.objective = ''
        self.eval_metric_name = ''

        self.eval_metric_name, self.custom_eval_metric_name = lightgbm_eval_metric(
            ml_task, eval_metric.name)

        self.custom_eval_metric = None
        if self.eval_metric.name == 'r2':
            self.custom_eval_metric = lightgbm_eval_metric_r2
        elif self.eval_metric.name == 'spearman':
            self.custom_eval_metric = lightgbm_eval_metric_spearman
        elif self.eval_metric.name == 'pearson':
            self.custom_eval_metric = lightgbm_eval_metric_pearson
        elif self.eval_metric.name == 'f1':
            self.custom_eval_metric = lightgbm_eval_metric_f1
        elif self.eval_metric.name == 'average_precision':
            self.custom_eval_metric = lightgbm_eval_metric_average_precision
        elif self.eval_metric.name == 'accuracy':
            self.custom_eval_metric = lightgbm_eval_metric_accuracy
        elif self.eval_metric.name == 'user_defined_metric':
            self.custom_eval_metric = lightgbm_eval_metric_user_defined

        self.num_class = (
            len(np.unique(y_train))
            if ml_task == MULTICLASS_CLASSIFICATION else None)
        self.objective = lightgbm_objective(ml_task, eval_metric.name)

    def optimize(self):
        param = {
            'objective': self.objective,
            'metric': self.eval_metric_name,
            'verbosity': -1,
            'seed': self.seed,
            'num_threads': self.n_jobs,
        }

        if self.num_class is not None:
            param['num_class'] = self.num_class

        # Reformat the data for LightGBM cross validation method
        train_set = lgb.Dataset(
            data=pd.concat([self.dtrain.data,
                            self.dvalid.data]).reset_index(drop=True),
            label=pd.concat([self.dtrain.label,
                             self.dvalid.label]).reset_index(drop=True),
            categorical_feature=self.dtrain.categorical_feature,
            free_raw_data=False,
        )

        train_index = range(len(self.dtrain.data))
        valid_index = range(len(self.dtrain.data), len(train_set.data))

        # Run the hyper-parameter tuning
        self.tuner = LightGBMTunerCV(
            params=param,
            train_set=self.dtrain,
            folds=[(train_index, valid_index)],
            verbose_eval=False,
            num_boost_round=100,
            early_stopping_rounds=50,
        )

        self.tuner.run()
        self.best = self.tuner.best_params
        return self.best
