import argparse
import math
import os
import random

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR


class DevNet(nn.Module):
    """docstring for ClassName."""

    def __init__(self, feature_dim):
        super(DevNet, self).__init__()
        self.dnn = nn.Sequential(
            nn.Linear(feature_dim, 1024), nn.ReLU(), nn.Linear(1000, 256),
            nn.ReLU(), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 10),
            nn.ReLU(), nn.Linear(10, 2))

    def forward(self, x):
        out = self.dnn(x)
        return out


class task_generator(object):

    def __init__(self, data_name, data_path, label_name, saved=False):
        super(task_generator, self).__init__()
        self.data_name = data_name
        self.data_path = data_path
        self.label_name = label_name
        self.saved = saved

    def data_split(self):
        """split dataset to train set and test set."""
        data_all = pd.read_csv(self.data_path)
        data_all_x = data_all.drop(self.label_name, axis=1)
        data_all_y = data_all[self.label_name]
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, train_size=0.8, random_state=2021)
        for train_index, test_index in sss.split(data_all_x, data_all_y):
            x_train, x_test = data_all_x.iloc[train_index], data_all_x.iloc[
                test_index]
            y_train, y_test = data_all_y.iloc[train_index], data_all_y.iloc[
                test_index]
            print(
                f'x_train shape: {x_train.shape}, x_test shape: {x_test.shape}, y_train shape: {y_train.shape}, y_test shape: {y_test.shape}'
            )
        data_train = pd.concat([x_train, y_train], axis=1)
        data_test = pd.concat([x_test, y_test], axis=1)
        if self.saved:
            to_path = os.path.join(
                os.path.dirname(self.data_path), self.data_name)
            if not os.path.exists(to_path):
                os.makedirs(to_path)
            data_train.to_csv(os.path.join(to_path, 'train.csv'))
            data_test.to_csv(os.path.join(to_path, 'test.csv'))
        return data_train, data_test

    def run(self):
        """generate meta_train task contain support set and query set."""


if __name__ == '__main__':
    data_name = 'KDD2014'
    data_path = '/home/wenqi.ao/workdir/anomaly_detection/deviation_network/dataset/KDD2014_donors_10feat_nomissing_normalised.csv'
    label_name = 'class'
    task_generator = task_generator(data_name, data_path, label_name)
    train_set, test_set = task_generator.data_split()
