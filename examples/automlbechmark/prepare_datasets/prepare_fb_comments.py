import os
from pathlib import Path

import pandas as pd

pd.options.display.max_columns = 100

SEED = 2

ROOT_DIR = Path('/home/robin/jianzh/autotabular/examples/automlbechmark')
RAW_DATA_DIR = ROOT_DIR / 'data/raw_data/fb_comments'
PROCESSED_DATA_DIR = ROOT_DIR / 'data/processed_data/fb_comments'

if not os.path.isdir(PROCESSED_DATA_DIR):
    os.makedirs(PROCESSED_DATA_DIR)

cols = ['_'.join(['col', str(i)]) for i in range(54)]

fb_comments = pd.read_csv(RAW_DATA_DIR / 'Features_Variant_5.csv', names=cols)
fb_comments['target'] = fb_comments.col_53
fb_comments.drop('col_53', axis=1, inplace=True)

print(fb_comments)
print(fb_comments.head())
print(fb_comments.describe())
print(fb_comments.info())

fb_comments.to_csv(PROCESSED_DATA_DIR / 'fb_comments.csv', index=None)
