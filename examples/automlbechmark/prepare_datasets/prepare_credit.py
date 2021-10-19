import os
import pandas as pd
import numpy as np

from autofe.feature_engineering.data_preprocess import preprocess, split_train_test

if __name__ == "__main__":
    root_dir = "./data/credit/"
    data = pd.read_excel(root_dir + 'credit.xls', sheet_name='Data', skiprows=1)
    id_cols = ['ID']
    cat_cols = ['EDUCATION', 'SEX', 'MARRIAGE']
    num_cols = [
        'LIMIT_BAL', 'AGE',
        'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
        'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
        'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
    ]
    label_col = 'default payment next month'
    for col in cat_cols:
        data[col] = data[col].astype('object')
    for col in num_cols:
        data[col] = data[col].astype(np.float32)
    data[label_col] = data[label_col].astype(np.float32)
    target_name = 'payment'
    data = data.rename(columns={label_col: target_name})

    data = preprocess(data, target_name)

    data_train, data_test = split_train_test(data, target_name, 0.2)
    data_train.to_csv(root_dir + 'data_train.csv', index = False)
    data_test.to_csv(root_dir + 'data_test.csv', index = False)
