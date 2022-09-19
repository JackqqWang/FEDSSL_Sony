#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torch.autograd import Variable
        

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, pseudo_label = None):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.pseudo_label = pseudo_label

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        if self.pseudo_label != None:
            label = int(self.pseudo_label[self.idxs[item]]) 
        return image, label



# class LocalUpdate_fedavg(object):
#     def __init__(self, args, dataset=None, idxs=set()):
#         self.args = args
#         self.loss_func = nn.CrossEntropyLoss()
#         self.selected_clients = []

#         idxs_train = np.random.permutation(list(idxs))
        
#         self.ldr_train = DataLoader(
#             DatasetSplit(dataset = dataset, idxs = idxs_train),
#             batch_size=self.args.local_bs, 
#             shuffle=True
#             )

#     def train(self, net):

#         net.train()
#         optimizer = torch.optim.SGD(net.parameters(), lr=0.003, momentum=0.9)
#         epoch_loss = []
#         for iter in range(self.args.local_ep):
#             batch_loss = []
#             for batch_idx, (images, labels) in enumerate(self.ldr_train):   
#                 images, labels = images.to(self.args.device), labels.to(self.args.device)
#                 net.zero_grad()
#                 log_probs = net(images)
#                 loss = self.loss_func(log_probs, labels)
#                 loss.backward()
#                 optimizer.step()
#                 batch_loss.append(loss.item())
#             epoch_loss.append(sum(batch_loss)/len(batch_loss))
#         return net.state_dict(), sum(epoch_loss) / len(epoch_loss)





# class LocalUpdate_fedavg_pseudo(object):
#     def __init__(self, args, dataset = None, pseudo_label = None, idxs = set()):
#         self.args = args
#         self.loss_func = nn.CrossEntropyLoss(ignore_index= -1)
#         self.selected_clients = []
#         self.pseudo_label = pseudo_label

#         idxs_train = np.random.permutation(list(idxs))
        
#         self.ldr_train = DataLoader(
#             DatasetSplit(dataset = dataset, idxs = idxs_train, pseudo_label = self.pseudo_label),
#             batch_size=self.args.local_bs, 
#             shuffle=True
#             )

#     def train(self, net):

#         net.train()
#         optimizer = torch.optim.SGD(net.parameters(), lr=0.003, momentum=0.9)
#         epoch_loss = []
#         for iter in range(self.args.local_ep):
#             batch_loss = []
#             for batch_idx, (images, labels) in enumerate(self.ldr_train):   
#                 images, labels = images.to(self.args.device), labels.to(self.args.device)
#                 net.zero_grad()
#                 log_probs = net(images)
#                 loss = self.loss_func(log_probs, labels)
#                 loss.backward()
#                 optimizer.step()
#                 batch_loss.append(loss.item())
#             epoch_loss.append(sum(batch_loss)/len(batch_loss))
#         return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

#     def train_pseudo(self, net):

#         net.train()
#         optimizer = torch.optim.SGD(net.parameters(), lr=0.003, momentum=0.9)
#         epoch_loss = []
#         for iter in range(self.args.local_ep):
#             batch_loss = []
#             for batch_idx, (images, labels) in enumerate(self.ldr_train): 
#                 images, labels = images.to(self.args.device), labels.to(self.args.device)
#                 net.zero_grad()
#                 log_probs = net(images)
#                 probs = torch.softmax(log_probs, dim=-1)
#                 max_probs, labels_pseudo = torch.max(probs, dim=-1)
#                 # pseudo label threshold
#                 mask_pseudo = max_probs.ge(0.95)
#                 # unlabeled data mask
#                 mask_unlabeled = labels.le(-1)
#                 # mask the pseudo label in low-quality and have ground truth
#                 labels_pseudo = labels_pseudo * mask_unlabeled * mask_pseudo
                
#                 labels_pseudo = labels_pseudo + (labels_pseudo != 0) + torch.tensor(-1)
#                 # get the final labels
#                 labels_pseudo = torch.max(labels_pseudo, labels)

#                 loss = self.loss_func(log_probs, labels_pseudo)
#                 # loss = self.loss_func(log_probs, labels)
#                 loss.backward()
#                 optimizer.step()
#                 batch_loss.append(loss.item())
#             epoch_loss.append(sum(batch_loss)/len(batch_loss))
#         return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


def batch_generator(x_mat, y, batch_size, seq_len):
    mean_data_x = np.mean(x_mat, axis=0)
    mean_data_y = np.mean(y, axis=0)
    # pad the beginning of the data with mean rows in order to minimize the error
    # on first rows while using sequence
    prefix_padding_x = np.asarray([mean_data_x for _ in range(seq_len - 1)])
    prefix_padding_y = np.asarray([mean_data_y for _ in range(seq_len - 1)])
    padded_data_x = np.vstack((prefix_padding_x, x_mat))
    padded_data_y = np.vstack((prefix_padding_y, y))
    seq_data = []
    seq_y = []
    for i in range(len(padded_data_x) - seq_len + 1):
        seq_data.append(padded_data_x[i:i + seq_len, :])
        # seq_y.append(padded_data_y[i + seq_len - 1:i + seq_len, :])
        seq_y.append(padded_data_y[i + seq_len - 2:i + seq_len - 1, :])
        if len(seq_data) == batch_size:
            if torch.cuda.is_available():
                yield Variable(torch.cuda.FloatTensor(seq_data)), Variable(torch.cuda.LongTensor(seq_y))
            else:
                yield Variable(torch.FloatTensor(seq_data)), Variable(torch.LongTensor(seq_y))
            seq_data = []
            seq_y = []
    if len(seq_data) > 0:  # handle data which is not multiply of batch size
        if torch.cuda.is_available():
            yield Variable(torch.cuda.FloatTensor(seq_data)), Variable(torch.cuda.LongTensor(seq_y))
        else:
            yield Variable(torch.FloatTensor(seq_data)), Variable(torch.LongTensor(seq_y))

class ServerUpdate_fedavg(object):
    def __init__(self, args, X_train_pos_server, X_train_neg_server, y_train_pos_server, y_train_neg_server):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss(ignore_index= -1)
        self.X_train_pos_server = X_train_pos_server
        self.X_train_neg_server = X_train_neg_server
        self.y_train_pos_server = y_train_pos_server
        self.y_train_neg_server = y_train_neg_server


    def train(self, model):
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=0.0001)

        # print('Start model training')

        for epoch in range(self.args.local_ep):
            loss_total = []
            for i, ((x_batch_pos, y_batch_pos), (x_batch_neg, y_batch_neg)) in enumerate(zip(batch_generator(self.X_train_pos_server, self.y_train_pos_server, self.args.local_bs, 2), batch_generator(self.X_train_neg_server, self.y_train_neg_server, self.args.local_bs, 2))):
                # feed data from two dataloader
                y_batch_pos = y_batch_pos.to(self.args.device)
                y_batch_neg = y_batch_neg.to(self.args.device)
                # print(y_batch_pos)
                opt.zero_grad()
                out_pos = model(x_batch_pos)
                out_neg = model(x_batch_neg)
                loss_pos = self.loss_func(out_pos, torch.flatten(y_batch_pos))
                loss_neg = self.loss_func(out_neg, torch.flatten(y_batch_neg))
                loss = loss_pos + loss_neg
                loss.backward()
                # print(loss.item())
                loss_total.append(loss.item())
                opt.step()
        return model.state_dict(), sum(loss_total) / len(loss_total)


        # optimizer = torch.optim.SGD(net.parameters(), lr=0.003, momentum=0.9)
        # epoch_loss = []
        # for iter in range(self.args.local_ep):
        #     batch_loss = []
        #     for batch_idx, (images, labels) in enumerate(self.ldr_train):   
        #         images, labels = images.to(self.args.device), labels.to(self.args.device)
        #         net.zero_grad()
        #         log_probs = net(images)
        #         loss = self.loss_func(log_probs, labels)
        #         loss.backward()
        #         optimizer.step()
        #         batch_loss.append(loss.item())
        #     epoch_loss.append(sum(batch_loss)/len(batch_loss))
        # return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    # def train_pseudo(self, net):

    #     net.train()
    #     optimizer = torch.optim.SGD(net.parameters(), lr=0.003, momentum=0.9)
    #     epoch_loss = []
    #     for iter in range(self.args.local_ep):
    #         batch_loss = []
    #         for batch_idx, (images, labels) in enumerate(self.ldr_train): 
    #             images, labels = images.to(self.args.device), labels.to(self.args.device)
    #             net.zero_grad()
    #             log_probs = net(images)
    #             probs = torch.softmax(log_probs, dim=-1)
    #             max_probs, labels_pseudo = torch.max(probs, dim=-1)
    #             # pseudo label threshold
    #             mask_pseudo = max_probs.ge(0.95)
    #             # unlabeled data mask
    #             mask_unlabeled = labels.le(-1)
    #             # mask the pseudo label in low-quality and have ground truth
    #             labels_pseudo = labels_pseudo * mask_unlabeled * mask_pseudo
                
    #             labels_pseudo = labels_pseudo + (labels_pseudo != 0) + torch.tensor(-1)
    #             # get the final labels
    #             labels_pseudo = torch.max(labels_pseudo, labels)

    #             loss = self.loss_func(log_probs, labels_pseudo)
    #             # loss = self.loss_func(log_probs, labels)
    #             loss.backward()
    #             optimizer.step()
    #             batch_loss.append(loss.item())
    #         epoch_loss.append(sum(batch_loss)/len(batch_loss))
    #     return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


class ClientUpdate_fedavg(object):
    def __init__(self, args, X_train_pos_server, X_train_neg_server, y_train_pos_server, y_train_neg_server):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss(ignore_index= -1)
        self.X_train_pos_server = X_train_pos_server
        self.X_train_neg_server = X_train_neg_server
        self.y_train_pos_server = y_train_pos_server
        self.y_train_neg_server = y_train_neg_server


    def train(self, model):
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=0.0001)

        # print('Start model training')

        for epoch in range(self.args.local_ep):
            loss_total = []
            for i, ((x_batch_pos, y_batch_pos), (x_batch_neg, y_batch_neg)) in enumerate(zip(batch_generator(self.X_train_pos_server, self.y_train_pos_server, self.args.local_bs, 2), batch_generator(self.X_train_neg_server, self.y_train_neg_server, self.args.local_bs, 2))):
                # feed data from two dataloader
                y_batch_pos = y_batch_pos.to(self.args.device)
                y_batch_neg = y_batch_neg.to(self.args.device)
                # print(y_batch_pos)
                opt.zero_grad()
                out_pos = model(x_batch_pos)
                out_neg = model(x_batch_neg)
                loss_pos = self.loss_func(out_pos, torch.flatten(y_batch_pos))
                loss_neg = self.loss_func(out_neg, torch.flatten(y_batch_neg))
                loss = loss_pos + loss_neg
                loss.backward()
                # print(loss.item())
                if loss.item() > 0:
                    loss_total.append(loss.item())
                opt.step()
        return model.state_dict(), sum(loss_total) / len(loss_total) if len(loss_total) != 0 else 0


        # optimizer = torch.optim.SGD(net.parameters(), lr=0.003, momentum=0.9)
        # epoch_loss = []
        # for iter in range(self.args.local_ep):
        #     batch_loss = []
        #     for batch_idx, (images, labels) in enumerate(self.ldr_train):   
        #         images, labels = images.to(self.args.device), labels.to(self.args.device)
        #         net.zero_grad()
        #         log_probs = net(images)
        #         loss = self.loss_func(log_probs, labels)
        #         loss.backward()
        #         optimizer.step()
        #         batch_loss.append(loss.item())
        #     epoch_loss.append(sum(batch_loss)/len(batch_loss))
        # return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    # def train_pseudo(self, net):

    #     net.train()
    #     optimizer = torch.optim.SGD(net.parameters(), lr=0.003, momentum=0.9)
    #     epoch_loss = []
    #     for iter in range(self.args.local_ep):
    #         batch_loss = []
    #         for batch_idx, (images, labels) in enumerate(self.ldr_train): 
    #             images, labels = images.to(self.args.device), labels.to(self.args.device)
    #             net.zero_grad()
    #             log_probs = net(images)
    #             probs = torch.softmax(log_probs, dim=-1)
    #             max_probs, labels_pseudo = torch.max(probs, dim=-1)
    #             # pseudo label threshold
    #             mask_pseudo = max_probs.ge(0.95)
    #             # unlabeled data mask
    #             mask_unlabeled = labels.le(-1)
    #             # mask the pseudo label in low-quality and have ground truth
    #             labels_pseudo = labels_pseudo * mask_unlabeled * mask_pseudo
                
    #             labels_pseudo = labels_pseudo + (labels_pseudo != 0) + torch.tensor(-1)
    #             # get the final labels
    #             labels_pseudo = torch.max(labels_pseudo, labels)

    #             loss = self.loss_func(log_probs, labels_pseudo)
    #             # loss = self.loss_func(log_probs, labels)
    #             loss.backward()
    #             optimizer.step()
    #             batch_loss.append(loss.item())
    #         epoch_loss.append(sum(batch_loss)/len(batch_loss))
    #     return net.state_dict(), sum(epoch_loss) / len(epoch_loss)