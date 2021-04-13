# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 20:30:31 2021

@author: ASUS
"""
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from torchvision import transforms
import torch
import numpy as np
import logging

LOG_FORMAT = "%(asctime)s - %(message)s"
logging.basicConfig(filename= "./pre_trained_models/result.log",
                    level = logging.INFO,
                    format=LOG_FORMAT)

def write_log(message):
    logger = logging.getLogger()
    logger.info(message)


def save_model(args, model, name=''):
    name = name if len(name) > 0 else 'default_model'
    torch.save(model, 'pre_trained_models/{}.pt'.format(name))


def load_model(args, name=''):
    name = name if len(name) > 0 else 'default_model'
    model = torch.load('pre_trained_models/{}.pt'.format(name))
    return model

def metrics(results, truths):
    preds = results.cpu().detach().numpy()
    truth = truths.cpu().detach().numpy()

    preds = np.where(preds > 0.5, 1, 0)
    truth = np.where(truth > 0.5, 1, 0)

    f_score_micro = f1_score(truth, preds, average='micro')
    f_score_macro = f1_score(truth, preds, average='macro')
    accuarcy = accuracy_score(truth, preds)

    precision = precision_score(truth, preds)
    recall = recall_score(truth, preds)
    return accuarcy, f_score_micro, f_score_macro, precision, recall

def multiclass_acc(results, truths):
    preds = results.view(-1).cpu().detach().numpy()
    truth = truths.view(-1).cpu().detach().numpy()

    preds = np.where(preds > 0.5, 1, 0)
    truth = np.where(truth > 0.5, 1, 0)

    return np.sum(preds == truths) / float(len(truths))
