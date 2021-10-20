import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

pd.options.display.max_columns = 100

SEED = 2

ROOT_DIR = Path('./')
RAW_DATA_DIR = ROOT_DIR / 'data/raw_data/bank_marketing'
PROCESSED_DATA_DIR = ROOT_DIR / 'data/processed_data/bank_marketing'
if not os.path.isdir(PROCESSED_DATA_DIR):
    os.makedirs(PROCESSED_DATA_DIR)

bankm = pd.read_csv(RAW_DATA_DIR / 'bank-additional-full.csv', sep=';')
bankm.drop('duration', axis=1, inplace=True)

bankm['target'] = (bankm['y'].apply(lambda x: x == 'yes')).astype(int)
bankm.drop('y', axis=1, inplace=True)

bankm.to_csv(PROCESSED_DATA_DIR / 'bankm.csv', index=None)

train_data, test_data = train_test_split(bankm, test_size=0.2)

train_data.to_csv(PROCESSED_DATA_DIR / 'train_data.csv', index=None)
test_data.to_csv(PROCESSED_DATA_DIR / 'test_data.csv', index=None)
