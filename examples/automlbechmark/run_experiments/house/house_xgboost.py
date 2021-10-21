from pathlib import Path

import pandas as pd
from autofe.get_feature import get_baseline_total_data, train_and_evaluate
from xgboost import XGBRegressor

if __name__ == '__main__':
    ROOTDIR = Path('./')
    PROCESSED_DATA_DIR = ROOTDIR / 'data/processed_data/house/'

    train_datafile = PROCESSED_DATA_DIR / 'train_data.csv'
    test_datafile = PROCESSED_DATA_DIR / 'test_data.csv'

    train_data = pd.read_csv(PROCESSED_DATA_DIR / 'train_data.csv')
    test_data = pd.read_csv(PROCESSED_DATA_DIR / 'test_data.csv')
    total_data = pd.concat([train_data, test_data]).reset_index(drop=True)
    len_train = len(train_data)
    target_name = 'SalePrice'
    classfier = XGBRegressor()
    """xgboost baseline"""
    # r2_score: 0.9016007093834981
    total_data_base = get_baseline_total_data(total_data)
    print('xgboost baseline:')
    score = train_and_evaluate(
        total_data_base,
        target_name,
        len_train,
        classfier,
        task_type='regression')

    param = {
        'subsample': 0.3653289846869241,
        'colsample_bytree': 0.9545839225514163,
        'learning_rate': 0.02858607644743819,
        'max_depth': 24,
        'n_estimators': 600
    }
    classfier = XGBRegressor(**param)
    total_data_base = get_baseline_total_data(total_data)
    score = train_and_evaluate(
        total_data_base,
        target_name,
        len_train,
        classfier,
        task_type='regression')
