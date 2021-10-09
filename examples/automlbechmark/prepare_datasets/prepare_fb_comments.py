import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

pd.options.display.max_columns = 100

SEED = 2

ROOT_DIR = Path(os.getcwd())

RAW_DATA_DIR = ROOT_DIR / 'data/raw_data/fb_comments'
PROCESSED_DATA_DIR = ROOT_DIR / 'data/processed_data/fb_comments'

if not os.path.isdir(PROCESSED_DATA_DIR):
    os.makedirs(PROCESSED_DATA_DIR)

cols = ['_'.join(['col', str(i)]) for i in range(54)]

fb_comments = pd.read_csv(RAW_DATA_DIR / 'Features_Variant_5.csv', names=cols)
fb_comments['target'] = fb_comments.col_53
fb_comments.drop('col_53', axis=1, inplace=True)

fb_comments_train, fb_comments_test = train_test_split(
    fb_comments, random_state=SEED, test_size=0.2)
fb_comments_val, fb_comments_test = train_test_split(
    fb_comments_test,
    random_state=SEED,
    test_size=0.5,
)

fb_comments_train.to_pickle(PROCESSED_DATA_DIR / 'fb_comments_train.pkl')
fb_comments_val.to_pickle(PROCESSED_DATA_DIR / 'fb_comments_val.pkl')
fb_comments_test.to_pickle(PROCESSED_DATA_DIR / 'fb_comments_test.pkl')
