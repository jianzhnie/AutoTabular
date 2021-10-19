import numpy as np
import pandas as pd
from autofe.get_feature import *
from xgboost import XGBRegressor

if __name__ == '__main__':
    root_path = './data/house/'
    train_data = pd.read_csv(root_path + 'data_train.csv')
    len_train = len(train_data)
    test_data = pd.read_csv(root_path + 'data_test.csv')
    total_data = pd.concat([train_data, test_data]).reset_index(drop=True)

    target_name = 'SalePrice'

    classfier = XGBRegressor()


    """xgboost baseline"""
    # r2_score: 0.9016007093834981
    total_data_base = get_baseline_total_data(total_data)
    print("xgboost baseline:")
    score = train_and_evaluate(total_data_base, target_name, len_train,
                                  classfier, task_type='regression')
