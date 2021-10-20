import calendar
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from autofe.feature_engineering.data_preprocess import preprocess
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def datatime_preprocess(df, col):
    df['tempDate'] = df[col].apply(lambda x: x.split())
    df['year'] = df.tempDate.apply(lambda x: x[0].split('-')[0])
    df['month'] = df.tempDate.apply(lambda x: x[0].split('-')[1])
    df['day'] = df.tempDate.apply(lambda x: x[0].split('-')[2])
    df['hour'] = df.tempDate.apply(lambda x: x[1].split(':')[0])
    df['weekday'] = df.tempDate.apply(lambda x: calendar.day_name[
        datetime.strptime(x[0], '%Y-%m-%d').weekday()])
    df['year'] = pd.to_numeric(df.year, errors='coerce')
    df['month'] = pd.to_numeric(df.month, errors='coerce')
    df['day'] = pd.to_numeric(df.day, errors='coerce')
    df['hour'] = pd.to_numeric(df.hour, errors='coerce')
    df = df.drop(['tempDate', col], axis=1)
    return df


if __name__ == '__main__':
    ROOT_DIR = Path('./')
    RAW_DATA_DIR = ROOT_DIR / 'data/raw_data/shelter'
    PROCESSED_DATA_DIR = ROOT_DIR / 'data/processed_data/shelter'
    if not os.path.isdir(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
    data = pd.read_csv(RAW_DATA_DIR / 'train.csv')
    data = datatime_preprocess(data, 'DateTime')

    target_name = 'OutcomeType'
    le = LabelEncoder()
    data[target_name] = le.fit_transform(data[target_name])
    data = preprocess(data, target_name)

    train_data, test_data = train_test_split(
        data, test_size=0.2, random_state=2021)
    train_data.to_csv(PROCESSED_DATA_DIR / 'train_data.csv', index=False)
    test_data.to_csv(PROCESSED_DATA_DIR / 'test_data.csv', index=False)
