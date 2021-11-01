import warnings
from typing import Any, Dict

import lightgbm as lgb
import pandas as pd
from lightgbm import Dataset as lgbDataset
from optuna.integration.lightgbm import LightGBMTunerCV
from sklearn.metrics import log_loss

warnings.filterwarnings("ignore")

class LGBOptimizerOptuna(object):

    def __init__(self, ml_task):

        self.ml_task = ml_task

    def optimize(self,
                 X_train,
                 y_train,
                 sample_weight,
                 X_validation,
                 y_validation,
                 sample_weight_validation,
                 maxevals: int = 200):

        dtrain = lgb.Dataset(
            X_train.to_numpy()
            if isinstance(X_train, pd.DataFrame) else X_train,
            label=y_train,
            weight=sample_weight,
            free_raw_data=False,
        )
        dvalid = lgb.Dataset(
            X_validation.to_numpy()
            if isinstance(X_validation, pd.DataFrame) else X_validation,
            label=y_validation,
            weight=sample_weight_validation,
            free_raw_data=False,
        )


        self.best = self.tuner.best_params
        # since n_estimators is not among the params that Optuna optimizes we
        # need to add it manually. We add a high value since it will be used
        # with early_stopping_rounds
        self.best["n_estimators"] = 1000  # type: ignore

    def get_objective(self, dtrain: lgbDataset, deval: lgbDataset):

        def objective():
            params["n_estimators"] = int(params["n_estimators"])
            params["num_leaves"] = int(params["num_leaves"])
            params["min_child_samples"] = int(params["min_child_samples"])
            params["verbose"] = -1
            params["seed"] = 1

            params["feature_pre_filter"] = False

            params["objective"] = self.objective

            model = lgb.train(
                params,
                dtrain,
                valid_sets=[deval],
                early_stopping_rounds=50,
                verbose_eval=False,
            )
            preds = model.predict(deval.data)
            score = log_loss(deval.label, preds)
            return score

        return objective

