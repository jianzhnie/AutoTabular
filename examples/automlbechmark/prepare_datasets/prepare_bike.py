import os
import pandas as pd
import calendar
from datetime import datetime

from autofe.feature_engineering.data_preprocess import preprocess, split_train_test

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
    data = pd.read_csv(root_dir + 'train.csv')
    data = datatime_preprocess(data, "datetime")
    target_name = "count"
    data = preprocess(data, target_name)

    data_train, data_test = split_train_test(data, target_name, 0.2)
    data_train.to_csv(root_dir + 'data_train.csv', index = False)
    data_test.to_csv(root_dir + 'data_test.csv', index = False)