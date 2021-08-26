import os
import math
import argparse
import random
import pandas as pd
import random
import sklearn
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np



class DevNet(nn.Module):
    """docstring for ClassName"""
    def __init__(self, 
                feature_dim,
                num_class,
                ae_hidden_neurons = [512, 256, 128],
                cl_hidden_neurons = [64, 32, 10],
                drop_rate = 0.2,
                batch_norm = True,
                hidden_activation = "relu",
                output_activation = "sigmoid"):
        super(DevNet, self).__init__()
        self.feature_dim = feature_dim
        self.num_class = num_class
        self.layers_neurons_encoder = [self.feature_dim, *ae_hidden_neurons]
        self.layers_neurons_decoder = self.layers_neurons_encoder[::-1]
        self.cl_hidden_neurons = [*cl_hidden_neurons, self.num_class]
        self.drop_rate = drop_rate
        self.batch_norm = batch_norm
        self.hidden_activation = nn.ReLU()
        self.output_activation = nn.Sigmoid()
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
        self.classifier = nn.Sequential()

        #create encoder model
        for idx, layer in enumerate(self.layers_neurons_encoder[:-1]):
            self.encoder.add_module("linear" + str(idx),
                                    nn.Linear(self.layers_neurons_encoder[idx],
                                            self.layers_neurons_encoder[idx + 1]))
            self.encoder.add_module("batch_norm" + str(idx),
                                    nn.BatchNorm1d(self.layers_neurons_encoder[idx + 1]))
            self.encoder.add_module("dropout" + str(idx),
                                    nn.Dropout(self.drop_rate))
            self.encoder.add_module(hidden_activation + str(idx),
                                    self.hidden_activation)
        #create decoder model
        for idx, layer in enumerate(self.layers_neurons_decoder[:-1]):
            self.encoder.add_module("linear" + str(idx),
                                    nn.Linear(self.layers_neurons_encoder[idx],
                                            self.layers_neurons_encoder[idx + 1]))
            self.encoder.add_module("batch_norm" + str(idx),
                                    nn.BatchNorm1d(self.layers_neurons_encoder[idx + 1]))
            self.encoder.add_module("dropout" + str(idx),
                                    nn.Dropout(self.drop_rate))
            if idx == len(self.layers_neurons_decoder) - 2:
                self.encoder.add_module(output_activation + str(idx),
                                        self.output_activation)
            else:
                self.encoder.add_module(hidden_activation + str(idx),
                                        self.hidden_activation)
        
        #create classifier
        for idx, layer in enumerate(self.cl_hidden_neurons[:-1]):
            self.classifier.add_module("linear" + str(idx),
                                    nn.Linear(self.cl_hidden_neurons[idx],
                                            self.cl_hidden_neurons[idx + 1]))
            self.classifier.add_module("batch_norm" + str(idx),
                                    nn.BatchNorm1d(self.cl_hidden_neurons[idx + 1]))
            self.classifier.add_module("dropout" + str(idx),
                                    nn.Dropout(self.drop_rate))
            if idx == len(self.cl_hidden_neurons) - 2:
                self.classifier.add_module(output_activation + str(idx),
                                        self.output_activation)
            else:
                self.classifier.add_module(hidden_activation + str(idx),
                                        self.hidden_activation)

    def forward(self,x):
        feature_vector = self.encoder(x)
        ae_output = self.decoder(feature_vector)
        cl_output = self.classifier(feature_vector) 
        return ae_output, cl_output


class task_generator(object):
    def __init__(self, data_name, data_path, label_name, saved = False):
        super(task_generator, self).__init__()
        self.data_name = data_name
        self.data_path = data_path
        self.label_name = label_name
        self.saved = saved

    def data_split(self):
        """split dataset to train set and test set"""
        data_all = pd.read_csv(self.data_path)
        data_all_x = data_all.drop(self.label_name, axis=1)
        data_all_y = data_all[self.label_name]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, train_size=0.8, random_state=2021)
        for train_index, test_index in sss.split(data_all_x, data_all_y):
            x_train, x_test = data_all_x.iloc[train_index], data_all_x.iloc[test_index]
            y_train, y_test = data_all_y.iloc[train_index], data_all_y.iloc[test_index]
            print(f"x_train shape: {x_train.shape}, x_test shape: {x_test.shape}, y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
        data_train = pd.concat([x_train, y_train], axis = 1)
        data_test = pd.concat([x_test, y_test], axis = 1)
        if self.saved:
            to_path = os.path.join(os.path.dirname(self.data_path), self.data_name)
            if not os.path.exists(to_path):
                os.makedirs(to_path)
            data_train.to_csv(os.path.join(to_path, "train.csv"))
            data_test.to_csv(os.path.join(to_path, "test.csv"))
        return data_train, data_test

    def run(self, data_train, num_normal, num_anomaly):
        """generate meta_train task contain support set and query set"""
        normal_set = data_train[data_train[self.label_name] == 0].reset_index(drop = True)
        anomaly_set = data_train[data_train[self.label_name] == 1].reset_index(drop = True)
        sample_normal_index = random.sample(range(normal_set.shape[0]), num_normal)
        sample_anomaly_index = random.sample(range(anomaly_set.shape[0]), num_anomaly)
        sampled_normal_set = normal_set.iloc[sample_normal_index]
        sampled_anomaly_set = anomaly_set.iloc[sample_anomaly_index]
        sampled_set = pd.concat([sampled_anomaly_set, sampled_normal_set], axis = 0)
        sampled_set_x = sampled_set.drop(self.label_name, axis = 1)
        sampled_set_y = sampled_set[self.label_name]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.75, train_size=0.25, random_state=2021)
        for train_index, test_index in sss.split(sampled_set_x, sampled_set_y):
            x_train, x_test = sampled_set_x.iloc[train_index], sampled_set_x.iloc[test_index]
            y_train, y_test = sampled_set_y.iloc[train_index], sampled_set_y.iloc[test_index]
            print(f"x_train shape: {x_train.shape}, x_test shape: {x_test.shape}, y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
        meta_train = pd.concat([x_train, y_train], axis = 1)
        meta_test = pd.concat([x_test, y_test], axis = 1)
        print(meta_train)
        print(meta_test)
        return  meta_train, meta_test



if __name__ == "__main__":
    data_name = "KDD2014"
    data_path = "/home/wenqi.ao/workdir/anomaly_detection/deviation_network/dataset/KDD2014_donors_10feat_nomissing_normalised.csv"
    label_name = "class"
    task_generator = task_generator(data_name, data_path, label_name)
    train_set, test_set = task_generator.data_split()
    meta_train, meta_test = task_generator.run(train_set, 100, 20)