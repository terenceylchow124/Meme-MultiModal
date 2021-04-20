
from .util import *
import torch
from torch import nn
import numpy as np
import time
import sys
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"


def train(train_loader, hyp_params, model, bert, tokenizer, feature_extractor, optimizer, criterion, epoch):
    epoch_loss = 0
    model.train()
    num_batches = hyp_params.n_train // hyp_params.batch_size
    proc_loss, proc_size = 0, 0
    total_loss = 0.0
    losses = []
    results = []
    truths = []
    n_examples = hyp_params.n_train
    start_time = time.time()

    for i_batch, data_batch in enumerate(train_loader):

        input_ids = data_batch["input_ids"]
        targets = data_batch["label"]
        attention_mask = data_batch['attention_mask']

        model.zero_grad()

        if hyp_params.use_cuda:
            with torch.cuda.device(0):
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                targets = targets.cuda()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        if hyp_params.dataset == 'memotion':
            _, preds = torch.max(outputs, dim=1)
        elif hyp_params.dataset == 'reddit':
            _, preds = torch.max(outputs, dim=1)
        else:
            preds = outputs

        preds_round = (preds > 0.5).float()
        loss = criterion(outputs, targets)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
        optimizer.step()

        total_loss += loss.item() * hyp_params.batch_size
        results.append(preds)
        truths.append(targets)

        proc_loss += loss * hyp_params.batch_size
        proc_size += hyp_params.batch_size
        if i_batch % hyp_params.log_interval == 0 and i_batch > 0:
            train_acc, train_f1, train_f1_macro, train_precision, train_recall = metrics(preds_round, targets)
            avg_loss = proc_loss / proc_size
            elapsed_time = time.time() - start_time
            msg = 'Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f} | Acc {:5.4f} | micro-score {:5.4f} | macro-score {:5.4f} | precision {:5.4f} | recall {:5.4f}'.format(epoch, i_batch, num_batches, elapsed_time * 1000 / hyp_params.log_interval, avg_loss, train_acc, train_f1, train_f1_macro, train_precision, train_recall)
            print(msg)
            write_log(msg)
            proc_loss, proc_size = 0, 0
            start_time = time.time()

    avg_loss = total_loss / hyp_params.n_train
    results = torch.cat(results)
    truths = torch.cat(truths)
    return results, truths, avg_loss

def evaluate(valid_loader, hyp_params, model, bert, tokenizer, feature_extractor, criterion, train=False, train_loader=None):
    model.eval()
    loader = train_loader if train else valid_loader
    total_loss = 0.0

    results = []
    truths = []
    correct_predictions = 0

    with torch.no_grad():
        for i_batch, data_batch in enumerate(loader):
            input_ids = data_batch["input_ids"]
            targets = data_batch["label"]
            attention_mask = data_batch['attention_mask']

            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    input_ids = input_ids.cuda()
                    attention_mask = attention_mask.cuda()
                    targets = targets.cuda()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            if hyp_params.dataset == 'memotion':
                _, preds = torch.max(outputs, dim=1)
            elif hyp_params.dataset == 'reddit':
                _, preds = torch.max(outputs, dim=1)
            else:
                preds = outputs

            total_loss += criterion(outputs, targets).item() * hyp_params.batch_size
            correct_predictions += torch.sum(preds == targets)

            # Collect the results into dictionary
            results.append(preds)
            truths.append(targets)

    avg_loss = total_loss / (hyp_params.n_train if train else hyp_params.n_valid)

    results = torch.cat(results)
    truths = torch.cat(truths)
    return results, truths, avg_loss
