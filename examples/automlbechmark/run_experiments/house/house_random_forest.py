import numpy as np
import pandas as pd
from autofe.get_feature import *
from sklearn.ensemble import RandomForestRegressor


if __name__ == '__main__':
    root_path = './data/house/'
    train_data = pd.read_csv(root_path + 'data_train.csv')
    len_train = len(train_data)
    test_data = pd.read_csv(root_path + 'data_test.csv')
    total_data = pd.concat([train_data, test_data]).reset_index(drop=True)

    target_name = 'SalePrice'

    classifier = RandomForestRegressor(random_state=0)

    """RF baseline"""
    # r2_score: 0.8899566255985333
    total_data_base = get_baseline_total_data(total_data)
    print("RF baseline: ")
    score = train_and_evaluate(total_data_base, target_name, len_train,
                                  classifier, task_type='regression')
    