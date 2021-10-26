import os
from pathlib import Path

import numpy as np
import pandas as pd
from autofe.feature_engineering.data_preprocess import preprocess
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    ROOT_DIR = Path('./')
    RAW_DATA_DIR = ROOT_DIR / 'data/raw_data/house'
    PROCESSED_DATA_DIR = ROOT_DIR / 'data/processed_data/house'
    if not os.path.isdir(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
    data = pd.read_csv(RAW_DATA_DIR / 'train.csv')
    target_name = 'SalePrice'
    data = preprocess(data, target_name)

    data['SalePrice'] = np.log1p(data['SalePrice'])
    train_data, test_data = train_test_split(
        data, test_size=0.2, random_state=2021)

    train_data.to_csv(PROCESSED_DATA_DIR / 'train_data.csv', index=False)
    test_data.to_csv(PROCESSED_DATA_DIR / 'test_data.csv', index=False)
