import os
import pandas as pd
import calendar
from datetime import datetime

from autofe.feature_engineering.data_preprocess import preprocess

def datatime_preprocess(df, col):
    df["tempDate"] = df[col].apply(lambda x:x.split())
    df['year'] = df.tempDate.apply(lambda x:x[0].split('-')[0])
    df['month'] = df.tempDate.apply(lambda x:x[0].split('-')[1])
    df['day'] = df.tempDate.apply(lambda x:x[0].split('-')[2])
    df['hour'] = df.tempDate.apply(lambda x:x[1].split(':')[0])
    df['weekday'] = df.tempDate.apply(lambda x:calendar.day_name[datetime.strptime(x[0],"%Y-%m-%d").weekday()])
    df['year'] = pd.to_numeric(df.year,errors='coerce')
    df['month'] = pd.to_numeric(df.month,errors='coerce')
    df['day'] = pd.to_numeric(df.day,errors='coerce')
    df['hour'] = pd.to_numeric(df.hour,errors='coerce')
    df = df.drop(["tempDate", col], axis=1)
    return df


if __name__ == "__main__":
    root_dir = "./data/bike/"
    train_data = pd.read_csv(root_dir + 'train.csv')
    test_data = pd.read_csv(root_dir + 'test.csv')

    train_data = datatime_preprocess(train_data, "datetime")
    test_data = datatime_preprocess(test_data, "datetime")

    target_name = "count"
    train_data = preprocess(train_data, target_name)
    test_data = preprocess(test_data, target_name)

    train_data.to_csv(root_dir + 'train.csv', index = False)
    test_data.to_csv(root_dir + 'test.csv', index = False)