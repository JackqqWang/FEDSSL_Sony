#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.Update import batch_generator
# import numpy
from sklearn.metrics import roc_auc_score

def test(model, X_test_pos, X_test_neg, y_test_pos, y_test_neg, args):
    model.eval()
    test_loss = 0
    TP, TN, total, total_batch = 0, 0, 0, 0
    loss_func = nn.CrossEntropyLoss(ignore_index= -1)
    label_all = []
    prob_all = []

    # eval on test_pos
    for i, (x_batch, y_batch) in enumerate(batch_generator(X_test_pos, y_test_pos, args.local_bs, 2)):
        y_batch = y_batch.to(args.device)
        out = model(x_batch)
        prob_all.extend(out[:,1].cpu().detach().numpy())
        label_all.extend(torch.flatten(y_batch))
        test_loss += loss_func(out, torch.flatten(y_batch)).item()
        out = out.squeeze(dim = 0)
        preds = F.log_softmax(out, dim=1).argmax(dim=1)
        total += y_batch.size(0)
        total_batch += 1
        TP += (preds == torch.flatten(y_batch)).sum().item()

    # eval on test_neg
    for i, (x_batch, y_batch) in enumerate(batch_generator(X_test_neg, y_test_neg, args.local_bs, 2)):
        y_batch = y_batch.to(args.device)
        out = model(x_batch)
        prob_all.extend(out[:,1].cpu().detach().numpy())
        label_all.extend(torch.flatten(y_batch))
        test_loss += loss_func(out, torch.flatten(y_batch)).item()
        out = out.squeeze(dim = 0)
        preds = F.log_softmax(out, dim=1).argmax(dim=1)
        total += y_batch.size(0)
        total_batch += 1
        TN += (preds == torch.flatten(y_batch)).sum().item()

    acc_test = (TP + TN) / total
    F1 = TP / (TP + (total - TP - TN) / 2)
    auc = roc_auc_score(label_all, prob_all)
    
    return acc_test, F1, test_loss / total_batch, auc