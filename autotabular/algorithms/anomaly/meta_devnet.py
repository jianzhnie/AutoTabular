import os
import math
import argparse
import random
import pandas as pd
import random
import sklearn
import  argparse
import copy
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
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
        self.cl_hidden_neurons = [ae_hidden_neurons[-1], *cl_hidden_neurons, 1]
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
            self.decoder.add_module("linear" + str(idx),
                                    nn.Linear(self.layers_neurons_decoder[idx],
                                            self.layers_neurons_decoder[idx + 1]))
            self.decoder.add_module("batch_norm" + str(idx),
                                    nn.BatchNorm1d(self.layers_neurons_decoder[idx + 1]))
            self.decoder.add_module("dropout" + str(idx),
                                    nn.Dropout(self.drop_rate))
            if idx == len(self.layers_neurons_decoder) - 2:
                self.decoder.add_module(output_activation + str(idx),
                                        self.output_activation)
            else:
                self.decoder.add_module(hidden_activation + str(idx),
                                        self.hidden_activation)
        
        #create classifier
        for idx, layer in enumerate(self.cl_hidden_neurons[:-2]):
            self.classifier.add_module("linear" + str(idx),
                                    nn.Linear(self.cl_hidden_neurons[idx],
                                            self.cl_hidden_neurons[idx + 1]))
            self.classifier.add_module("batch_norm" + str(idx),
                                    nn.BatchNorm1d(self.cl_hidden_neurons[idx + 1]))
            self.classifier.add_module("dropout" + str(idx),
                                    nn.Dropout(self.drop_rate))
            self.classifier.add_module(hidden_activation + str(idx),
                                    self.hidden_activation)
        idx += 1
        self.classifier.add_module("linear" + str(idx),
                                    nn.Linear(self.cl_hidden_neurons[idx],
                                            self.cl_hidden_neurons[idx + 1]))


    def forward(self,x):
        feature_vector = self.encoder(x)
        ae_output = self.decoder(feature_vector)
        cls_output = self.classifier(feature_vector) 
        return ae_output, cls_output


class Task(object):
    def __init__(self, data_name, data_path, label_name, saved = False):
        super(Task, self).__init__()
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
            # print(f"x_train shape: {x_train.shape}, x_test shape: {x_test.shape}, y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
        data_train = pd.concat([x_train, y_train], axis = 1)
        data_test = pd.concat([x_test, y_test], axis = 1)
        if self.saved:
            to_path = os.path.join(os.path.dirname(self.data_path), self.data_name)
            if not os.path.exists(to_path):
                os.makedirs(to_path)
            data_train.to_csv(os.path.join(to_path, "train.csv"))
            data_test.to_csv(os.path.join(to_path, "test.csv"))
        return data_train, data_test

    def generator(self, data_train, n_task, n_support, n_query, unbalanced_rate = 0.5):
        """generate meta_train task contain support set and query set"""
        n_samples = n_support + n_query
        normal_set = data_train[data_train[self.label_name] == 0].reset_index(drop = True)
        anomaly_set = data_train[data_train[self.label_name] == 1].reset_index(drop = True)
        num_normal = int(n_samples * (1 - unbalanced_rate))
        num_anomaly = n_samples - num_normal
        b_x_spt = []
        b_y_spt = []
        b_x_qry = []
        b_y_qry = []
        for i in range(n_task):
            sample_normal_index = random.sample(range(normal_set.shape[0]), num_normal)
            sample_anomaly_index = random.sample(range(anomaly_set.shape[0]), num_anomaly)
            sampled_normal_set = normal_set.iloc[sample_normal_index]
            sampled_anomaly_set = anomaly_set.iloc[sample_anomaly_index]
            sampled_set = pd.concat([sampled_anomaly_set, sampled_normal_set], axis = 0)
            sampled_set_x = sampled_set.drop(self.label_name, axis = 1)
            sampled_set_y = sampled_set[self.label_name]
            test_rate = n_query / n_samples
            train_rate = 1 - test_rate
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_rate, train_size=train_rate, random_state=2021)
            for train_index, test_index in sss.split(sampled_set_x, sampled_set_y):
                x_train, x_test = sampled_set_x.iloc[train_index], sampled_set_x.iloc[test_index]
                y_train, y_test = sampled_set_y.iloc[train_index], sampled_set_y.iloc[test_index]
                # print(f"x_train shape: {x_train.shape}, x_test shape: {x_test.shape}, y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
            x_train = np.array(x_train)
            x_test = np.array(x_test)
            y_train = np.array(y_train).reshape(-1,1)
            y_test = np.array(y_test).reshape(-1,1)
            b_x_spt.append(x_train)
            b_y_spt.append(y_train)
            b_x_qry.append(x_test)
            b_y_qry.append(y_test)

        return  np.array(b_x_spt), np.array(b_y_spt), np.array(b_x_qry), np.array(b_y_qry)


class MetaLoss(nn.Module):
    '''
    z-score-based deviation loss
    ''' 
    def __init__(self, 
        confidence_margin = 5,
        alpha = 0.5):
        super(MetaLoss, self).__init__()
        self.confidence_margin = confidence_margin
        self.alpha = alpha
    
    def forward(self, x, x_rec, y_pred, y_true):
        ref = torch.randn(1, 5000).cuda()
        dev = (y_pred - torch.mean(ref, dim=1)) / torch.std(ref, dim=1)
        inlier_loss = torch.abs(dev)
        outlier_loss = torch.abs(torch.max(self.confidence_margin-dev, torch.zeros(1).cuda()))
        cls_loss = (1 - y_true) * inlier_loss + y_true * outlier_loss
        cls_loss = torch.sum(cls_loss, dim=0)
        mseloss = nn.MSELoss()
        ae_loss = mseloss(x, x_rec)
        loss = self.alpha*ae_loss + (1-self.alpha)*cls_loss
        return loss


class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args):
        super(Meta, self).__init__()
        """
        :param args:
        """
        self.featrue_dim = args.feature_dim
        self.num_class = args.num_class
        self.alpha = args.alpha
        self.alpha = torch.Tensor(np.array(self.alpha)).cuda()
        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.net = DevNet(self.featrue_dim, self.num_class)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)
    
    def assign_network_value(self, vars):
        idx = 0
        for name, param in self.net.named_parameters():
            print(name, param)
            print(vars[idx])
            param.copy_(vars[idx])
            idx += 1
        assert idx == len(vars)

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """
        :param x_spt:   [b, setsz, feature_dim]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, feature_dim]
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num = x_spt.size(0)
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]
        device = torch.device('cuda')
        # self.update_lr = Variable(torch.Tensor(np.array(self.update_lr)), requires_grad=False)


        for i in range(self.task_num):

            # 1. run the i-th task and compute loss for k=0
            x_rec, y_pred = self.net(x_spt[i])
            model_loss = MetaLoss().to(device)
            loss = model_loss(x_spt[i], x_rec, y_pred, y_spt[i])

            self.net.parameters = nn.ParameterList(self.net.parameters())
            copy_net_parameters = copy.deepcopy(self.net.parameters)
            # for name, param in self.net.named_parameters():
            #     print(name, param)
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))
            print(len(fast_weights))
            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                x_rec_q, y_pred_q = self.net(x_qry[i])
                # model_loss = MetaLoss().to(device)
                loss_q = model_loss(x_qry[i], x_rec_q, y_pred_q, y_qry[i])
                print(f"loss_q: {loss_q}")
                losses_q[0] += loss_q

                pred_q = F.softmax(y_pred_q, dim=1).argmax(dim=1).unsqueeze(1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                self.assign_network_value(fast_weights)
                x_rec_q, y_pred_q = self.net(x_qry[i])
                # model_loss = MetaLoss().to(device)
                loss_q = model_loss(x_qry[i], x_rec_q, y_pred_q, y_qry[i])
                print(f"loss_q: {loss_q}")
                losses_q[1] += loss_q
                # [setsz]
                pred_q = F.softmax(y_pred_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                x_rec, y_pred = self.net(x_spt[i])
                # model_loss = MetaLoss().to(device)
                loss = model_loss(x_spt[i], x_rec, y_pred, y_spt[i])
                print(f"loss: {loss}")
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, self.net.parameters())
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))
                print(len(fast_weights))

                self.assign_network_value(fast_weights)
                x_rec_q, y_pred_q = self.net(x_qry[i])
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = model_loss(x_qry[i], x_rec_q, y_pred_q, y_qry[i])
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    pred_q = F.softmax(y_pred_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct



        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        # print('meta update')
        # for p in self.net.parameters()[:5]:
        # 	print(torch.norm(p).item())
        self.meta_optim.step()


        accs = np.array(corrects) / (querysz * task_num)

        return accs

if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=2)
    argparser.add_argument('--num_class', type=int, help='n class', default=2)
    argparser.add_argument('--feature_dim', type=int, help='number of features', default=10)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=32)
    argparser.add_argument('--alpha', type=float, help='factor of two different loss', default=0.5)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.4)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)

    args = argparser.parse_args()
    data_name = "KDD2014"
    data_path = "/home/wenqi.ao/workdir/anomaly_detection/deviation_network/dataset/KDD2014_donors_10feat_nomissing_normalised.csv"
    label_name = "class"
    task_generator = Task(data_name, data_path, label_name)
    train_set, test_set = task_generator.data_split()
    device = torch.device('cuda')
    maml = Meta(args).to(device)
    for epoch in range(args.epoch):
        x_spt, y_spt, x_qry, y_qry = task_generator.generator(train_set, 10, 20, 10)
        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).float().to(device), torch.from_numpy(y_spt).float().to(device), \
                                     torch.from_numpy(x_qry).float().to(device), torch.from_numpy(y_qry).float().to(device)
        accs = maml(x_spt, y_spt, x_qry, y_qry)
        print(f"Acc: {accs}")
    # net = DevNet(feature_dim = 10, num_class = 2)
    # print(len(nn.ParameterList(net.parameters())))
    # for name, param in net.named_parameters():
    #     print(name, param) 