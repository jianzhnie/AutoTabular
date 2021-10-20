import os
from pathlib import Path

import pandas as pd
from autofe.feature_engineering.data_preprocess import preprocess
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    ROOT_DIR = Path('./')
    RAW_DATA_DIR = ROOT_DIR / 'data/raw_data/credit'
    PROCESSED_DATA_DIR = ROOT_DIR / 'data/processed_data/credit'
    if not os.path.isdir(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)

    data = pd.read_excel(
        RAW_DATA_DIR / 'credit.xls', sheet_name='Data', skiprows=1)

    id_cols = ['ID']
    cat_cols = ['EDUCATION', 'SEX', 'MARRIAGE']
    num_cols = [
        'LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5',
        'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4',
        'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
        'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
    ]
    label_col = 'default payment next month'
    print(data.info())
    for c in cat_cols:
        try:
            data[c] = data[c].apply(str)
        except AttributeError:
            pass
    print(data.info())
    target_name = 'payment'
    data = data.rename(columns={label_col: target_name})
    data = preprocess(data, target_name)
    train_data, test_data = train_test_split(
        data, test_size=0.2, random_state=2021)
    train_data.to_csv(PROCESSED_DATA_DIR / 'train_data.csv', index=False)
    test_data.to_csv(PROCESSED_DATA_DIR / 'test_data.csv', index=False)
