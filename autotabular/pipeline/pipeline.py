import os
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch.utils.data import Dataset, DataLoader


data_path = "/home/wenqi.ao/workdir/autotabular/autotabular/datasets/data/heart-disease-uci.csv"
data = pd.read_csv(data_path)
label_column = "target"
x, y = data.drop(label_column, axis = 1), data[label_column]

def split_train_test(X, Y, test_size = 0.2):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, train_size=1-test_size, random_state=2021)
    for train_index, test_index in sss.split(X, Y):
        x_train, x_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
    data_train = pd.concat([x_train, y_train], axis = 1)
    data_test = pd.concat([x_test, y_test], axis = 1)
    return data_train, data_test

class StuctureData(Dataset):
    def __init__(self, X, Y):
        self.x = X.values
        self.y = Y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])
