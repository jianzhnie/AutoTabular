#!/bin/bash
# python run_experiments/adult/adult_lr.py
# python run_experiments/adult/adult_random_forest.py
# python run_experiments/adult/adult_xgboost.py
# python run_experiments/adult/adult_tabnet.py

# python run_experiments/house/house_optuna/xgboost_sklearn.py
# sleep 2h
python run_experiments/house/house_optuna/randomforest_sklearn.py
sleep 4h
python run_experiments/shelter/shelter_optuna/randomforest_sklearn.py
sleep 6h
python run_experiments/shelter/shelter_optuna/xgboost_sklearn.py