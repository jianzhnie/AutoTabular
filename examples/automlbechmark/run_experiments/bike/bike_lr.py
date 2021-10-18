import numpy as np
import pandas as pd
from autofe.get_feature import *
from sklearn.linear_model import Ridge


if __name__ == '__main__':
    root_path = './data/bike/'
    train_data = pd.read_csv(root_path + 'data_train.csv')
    len_train = len(train_data)
    test_data = pd.read_csv(root_path + 'data_test.csv')
    total_data = pd.concat([train_data, test_data]).reset_index(drop=True)

    target_name = 'count'

    classfier = Ridge(random_state=0)


    """lr baseline"""
    r2_score: 0.9999999999999997
    total_data_base = get_baseline_total_data(total_data)
    score = train_and_evaluate(total_data_base, target_name, len_train,
                                  classfier, task_type='regression')
