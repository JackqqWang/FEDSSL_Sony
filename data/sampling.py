#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import numpy as np
from torch import tensor
from multiprocessing import cpu_count
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.optim.lr_scheduler import _LRScheduler
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from sklearn.utils import shuffle

def getDataset(frac_train_test, num_users, label_rate, iid):
    if iid == 'iid':
        benign = pd.read_csv('./data/5.benign.csv')
        gfyt1 = pd.read_csv('./data/5.gafgyt.combo.csv')
        X_raw_pos = benign[:61380]
        X_raw_neg = gfyt1
    else:
        benign1 = pd.read_csv('./data/1.benign.csv')
        benign2 = pd.read_csv('./data/2.benign.csv')
        benign3 = pd.read_csv('./data/3.benign.csv')
        benign4 = pd.read_csv('./data/4.benign.csv')
        benign5 = pd.read_csv('./data/5.benign.csv')
        X_raw_pos = pd.concat([benign1, benign2, benign3, benign4, benign5])
        X_raw_pos = X_raw_pos.sample(frac = 61380 / len(X_raw_pos))
        X_raw_pos.reset_index(drop=True, inplace = True)


        gfyt1 = pd.read_csv('./data/1.gafgyt.junk.csv')
        gfyt2 = pd.read_csv('./data/2.gafgyt.scan.csv')
        gfyt3 = pd.read_csv('./data/3.gafgyt.tcp.csv')
        gfyt4 = pd.read_csv('./data/4.gafgyt.udp.csv')
        gfyt5 = pd.read_csv('./data/5.gafgyt.combo.csv')
        X_raw_neg = pd.concat([gfyt1, gfyt2, gfyt3, gfyt4, gfyt5])
        X_raw_neg = X_raw_neg.sample(frac = 61380 / len(X_raw_neg))
        X_raw_neg.reset_index(drop=True, inplace = True)


    X_clients_pos = {}
    y_clients_pos = {}

    y_raw_pos = pd.DataFrame([0] * len(X_raw_pos))
    X_raw_pos = (X_raw_pos - X_raw_pos.mean()) / X_raw_pos.std()
    X_train_pos = X_raw_pos.sample(frac = frac_train_test)
    X_test_pos = X_raw_pos.drop(X_train_pos.index)
    y_test_pos = y_raw_pos.drop(X_train_pos.index)
    y_train_pos = y_raw_pos.drop(y_test_pos.index)

    X_train_pos.reset_index(drop=True, inplace = True)
    y_train_pos.reset_index(drop=True, inplace = True)
    X_test_pos.reset_index(drop=True, inplace = True)
    y_test_pos.reset_index(drop=True, inplace = True)

    n = len(X_train_pos)
    X_train_pos_server = X_train_pos[int(n * (1 - label_rate)):]
    y_train_pos_server = pd.DataFrame([0] * len(X_train_pos_server)) 
    X_train_pos = X_train_pos[int(n * label_rate):].reset_index(drop=True)

    num_per_clients = len(X_train_pos) // num_users
    for i in range(num_users):
        X_clients_pos[i] = X_train_pos[i * num_per_clients: (i + 1) * num_per_clients].reset_index(drop=True)
        y_clients_pos[i] = pd.DataFrame([-1] * len(X_clients_pos[i])) 


    X_clients_neg = {}
    y_clients_neg = {}

    y_raw_neg = pd.DataFrame([1] * len(X_raw_neg))
    X_raw_neg = (X_raw_neg - X_raw_neg.mean()) / X_raw_neg.std()
    X_train_neg = X_raw_neg.sample(frac = frac_train_test)
    X_test_neg = X_raw_neg.drop(X_train_neg.index)
    y_test_neg = y_raw_neg.drop(X_train_neg.index)
    y_train_neg = y_raw_neg.drop(y_test_neg.index)

    X_train_neg.reset_index(drop=True, inplace = True)
    y_train_neg.reset_index(drop=True, inplace = True)
    X_test_neg.reset_index(drop=True, inplace = True)
    y_test_neg.reset_index(drop=True, inplace = True)

    n = len(X_train_neg)
    X_train_neg_server = X_train_neg[int(n * (1 - label_rate)):]
    y_train_neg_server = pd.DataFrame([1] * len(X_train_neg_server)) 
    X_train_neg = X_train_neg[int(n * label_rate):].reset_index(drop=True)

    for i in range(num_users):
        X_clients_neg[i] = X_train_neg[i * num_per_clients: (i + 1) * num_per_clients].reset_index(drop=True)
        y_clients_neg[i] = pd.DataFrame([-1] * len(X_clients_neg[i])) 

    return X_train_pos_server, X_train_neg_server, X_clients_pos, X_clients_neg, X_test_pos, X_test_neg, \
        y_train_pos_server, y_train_neg_server, y_clients_pos, y_clients_neg, y_test_pos, y_test_neg


def iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def iid_pseudo(dataset, num_users, label_rate):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    dict_users_labeled = set()
    pseudo_label = [i for i in range(len(dataset))]
    for idxs in range(len(dataset)):
        pseudo_label[idxs] = dataset[idxs][1]
    
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        dict_users_labeled = dict_users_labeled | set(np.random.choice(list(dict_users[i]), int(num_items * label_rate), replace=False))                     
        all_idxs = list(set(all_idxs) - dict_users[i])
        for idxs in list(dict_users[i] - dict_users_labeled):
            pseudo_label[idxs] = tensor(-1)
    return dict_users, dict_users_labeled, pseudo_label


# def sample(dataset, num_users, label_rate, iid_if):
#     num_items = int(len(dataset)/num_users)
#     dict_users, dict_users_labeled = {}, {}
#     pseduo_label, all_idxs = [i for i in range(len(dataset))] ,[i for i in range(len(dataset))]
#     dict_idxs_class = {i : set() for i in range(10)}
#     for idxs in range(len(dataset)):
#         label_class = dataset[idxs][1]
#         dict_idxs_class[label_class] = dict_idxs_class[label_class] | set([idxs])
#         pseduo_label[idxs] = dataset[idxs][1]
#     for i in range(10):
#         dict_idxs_class[i] = list(dict_idxs_class[i])
#     label_rate_dict = {}
#     for i in range(num_users):
#         label_rate_dict[i] = label_rate

#     for i in range(num_users):
#         dict_users[i], dict_users_labeled[i], dict_idxs_class = get_dict(dict_idxs_class, num_users, iid_if, label_rate_dict[i], i)


#     return dict_users, dict_users_labeled, pseduo_label

# def noniid(dataset, num_users):
#     """
#     Sample non-I.I.D client data from MNIST dataset
#     :param dataset:
#     :param num_users:
#     :return:
#     """
#     # 60,000 training imgs -->  200 imgs/shard X 300 shards
#     num_shards, num_imgs = 200, 300
#     idx_shard = [i for i in range(num_shards)]
#     dict_users = {i: np.array([]) for i in range(num_users)}
#     idxs = np.arange(num_shards*num_imgs)
#     labels = dataset.targets.numpy()

#     # sort labels
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
#     idxs = idxs_labels[0, :]

#     # divide and assign 2 shards/client
#     for i in range(num_users):
#         rand_set = set(np.random.choice(idx_shard, 2, replace=False))
#         idx_shard = list(set(idx_shard) - rand_set)
#         for rand in rand_set:
#             dict_users[i] = np.concatenate(
#                 (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
#     return dict_users