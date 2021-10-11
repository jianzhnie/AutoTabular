import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

SEED = 1
ROOT_DIR = Path('/home/robin/jianzh/autotabular/examples/automlbechmark')
RAW_DATA_DIR = ROOT_DIR / 'data/raw_data/adult'
PROCESSED_DATA_DIR = ROOT_DIR / 'data/processed_data/adult'
if not os.path.isdir(PROCESSED_DATA_DIR):
    os.makedirs(PROCESSED_DATA_DIR)

colnames = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
    'income',
]

colnames = [c.replace('-', '_') for c in colnames]
adult_train = pd.read_csv(RAW_DATA_DIR / 'adult.data', names=colnames)
adult_test = pd.read_csv(
    RAW_DATA_DIR / 'adult.test', skiprows=1, names=colnames)
adult = pd.concat([adult_train, adult_test])

# fill na
adult = adult.replace(to_replace=' ?', value=np.nan)
fill_transformer = SimpleImputer(
    missing_values=np.nan, strategy='most_frequent')
adult = fill_transformer.fit_transform(adult)
adult = pd.DataFrame(adult, columns=colnames)

# types convert
adult.age = adult.age.astype(float)
adult['hours_per_week'] = adult['hours_per_week'].astype(float)

for c in adult.columns:
    try:
        adult[c] = adult[c].str.lower()
    except AttributeError:
        pass

adult['target'] = (adult['income'].apply(lambda x: '>50' in x)).astype(int)
adult.drop('income', axis=1, inplace=True)
adult.drop('education_num', axis=1, inplace=True)
adult = adult.sample(frac=1).reset_index(drop=True)

print(adult)
print(adult.head())
print(adult.describe(include=['O']))
print(adult.info())

adult_train, adult_test = train_test_split(
    adult, test_size=0.2, random_state=SEED, stratify=adult.target)
adult_val, adult_test = train_test_split(
    adult_test, test_size=0.5, random_state=SEED, stratify=adult_test.target)

adult.to_csv(PROCESSED_DATA_DIR / 'adult.csv', index=None)
adult_train.to_csv(PROCESSED_DATA_DIR / 'adult_train.csv', index=None)
adult_val.to_csv(PROCESSED_DATA_DIR / 'adult_val.csv', index=None)
adult_test.to_csv(PROCESSED_DATA_DIR / 'adult_test.csv', index=None)
