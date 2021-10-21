# """Optuna example that optimizes a classifier configuration for cancer dataset
# using XGBoost.

# In this example, we optimize the validation accuracy of cancer detection using XGBoost. We optimize both the choice of booster model and its hyperparameters.
# """

# import numpy as np
# import optuna
# import pandas as pd
# import sklearn.datasets
# import sklearn.metrics
# import xgboost as xgb
# from autofe.get_feature import get_baseline_total_data, train_and_evaluate
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, roc_auc_score
# from sklearn.model_selection import train_test_split

# class OptunaHpo(object):

#     def __init__(
#         self,
#         estimator: str = 'randomforest',
#         objective: str = 'binary',
#         verbose: bool = False,
#     ):

#         self.objective = objective
#         self.verbose = verbose
#         self.early_stop_dict: Dict = {}

#     def optimize(self, X, y, max_evals=100):

#     def optimize(
#         self,
#         dtrain: lgbDataset,
#         deval: lgbDataset,
#         maxevals: int = 200,
#     ):

#         if self.objective == 'regression':
#             self.best = lgb.LGBMRegressor().get_params()
#         else:
#             self.best = lgb.LGBMClassifier().get_params()
#         del (self.best['silent'], self.best['importance_type'])

#         param_space = self.hyperparameter_space()
#         objective = self.get_objective(dtrain, deval)
#         return None
