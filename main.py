from utils.util_args import get_args
from utils.util_loader import data_loader
from utils.util import save_model, load_model, metrics, write_log
import models
from transformers import BertModel, AlbertModel, AutoConfig
from transformers import BertTokenizer, AlbertTokenizer, AutoTokenizer

import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import time
import sys
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

output_dim_dict = {
    'memotion': 2,
    'reddit': 2
}

criterion_dict = {
    'memotion': 'CrossEntropyLoss',
    'reddit': 'CrossEntropyLoss'
}

def initiate(hyp_params, tokenizer, train_loader, valid_loader, test_loader=None):
    model = getattr(models, hyp_params.model+'Model')(hyp_params)

    if hyp_params.train_reddit == 1:
        # bert = BertModel.from_pretrained(hyp_params.bert_model)
        bert = AlbertModel.from_pretrained(hyp_params.bert_model)
    else:
        bert = torch.load('pre_trained_models/model_A1.pt')

    feature_extractor = torch.hub.load('pytorch/vision:v0.6.0', hyp_params.cnn_model, pretrained=True)
    
    for param in feature_extractor.features.parameters():
        param.requires_grad = False

    if hyp_params.use_cuda:
        model = model.cuda()
        bert = bert.cuda()
        feature_extractor = feature_extractor.cuda()

    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    criterion = getattr(nn, hyp_params.criterion)()

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1, verbose=True)

    settings = {'model': model,
                'bert': bert,
                'tokenizer': tokenizer,
                'feature_extractor': feature_extractor,
                'optimizer': optimizer,
                'criterion': criterion,
                'scheduler': scheduler}

    return settings

def train_model(settings, hyp_params, train_loader, val_loader, test_loader=None):

    if hyp_params.train_reddit == 1:
        from utils.util_train_reddit import train, evaluate
    else:
        from utils.util_train import train, evaluate

    model = settings['model']
    bert = settings['bert']
    tokenizer = settings['tokenizer']
    feature_extractor = settings['feature_extractor']
    optimizer = settings['optimizer']
    criterion = settings['criterion']

    scheduler = settings['scheduler']

    best_valid = 1e8
    for epoch in range(1, hyp_params.num_epochs+1):
        start = time.time()

        # training
        train_results, train_truths, train_loss = train(train_loader,
                                                        hyp_params,
                                                        model,
                                                        bert,
                                                        tokenizer,
                                                        feature_extractor,
                                                        optimizer,
                                                        criterion,
                                                        epoch)
        # evaluating
        results, truths, val_loss = evaluate(val_loader,
                                             hyp_params,
                                             model,
                                             bert,
                                             tokenizer,
                                             feature_extractor,
                                             criterion,
                                             train=False,
                                             train_loader=None)
        end = time.time()
        duration = end-start
        scheduler.step(val_loss)
        train_acc, train_f1, train_f1_macro, train_precision, train_recall = metrics(train_results, train_truths)
        val_acc, val_f1, val_f1_macro, val_precision, val_recall = metrics(results, truths)
        print("-"*100)
        msg = 'Epoch {:2d} | Time {:5.4f} sec | Train Loss {:5.4f} Acc {:5.4f} micro-score {:5.4f} macro-score {:5.4f} | Valid Loss {:5.4f} Acc {:5.4f} mirco-score {:5.4f}  macro-score {:5.4f}'.format(epoch, duration, train_loss, train_acc, train_f1, train_f1_macro, val_loss, val_acc, val_f1, val_f1_macro)
        print(msg)
        print("-"*100)
        msg1 = 'Epoch {:2d} | Time {:5.4f} sec | Train Loss {:5.4f} | Train Acc {:5.4f} | micro-score {:5.4f} | macro-score {:5.4f} | precision {:5.4f} | recall {:5.4f}'.format(epoch, duration, train_loss, train_acc, train_f1, train_f1_macro, train_precision, train_recall)
        msg2 = 'Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Valid Acc {:5.4f} | mirco-score {:5.4f} | macro-score {:5.4f} | precision {:5.4f} | recall {:5.4f}'.format(epoch, duration, val_loss,   val_acc,   val_f1,   val_f1_macro,   val_precision,   val_recall)
        write_log(msg1)
        write_log(msg2)

        if val_loss < best_valid:
            # print(f'Saved model at pre_trained_models/{hyp_params.name}.pt!')
            save_model(hyp_params, model, name=hyp_params.name)
            best_valid = val_loss

    model = load_model(hyp_params, name=hyp_params.name)

    results, truths, train_loss = evaluate(train_loader,
                                         hyp_params,
                                         model,
                                         bert,
                                         tokenizer,
                                         feature_extractor,
                                         criterion,
                                         train=True,
                                         train_loader=train_loader)

    train_acc, train_f1, train_f1_macro, train_precision, train_recall  = metrics(results, truths)
    msg = "\n\nTrain Loss {:5.4f} | Train Acc {:5.4f} | Train f1-score {:5.4f} | Train f1-score-macro {:5.4f} | Train precision {:5.4f} | Train recall {:5.4f}".format(train_loss, train_acc, train_f1, train_f1_macro, train_precision, train_recall)
    print(msg)
    write_log(msg)
    results, truths, test_loss = evaluate(val_loader,
                                         hyp_params,
                                         model,
                                         bert,
                                         tokenizer,
                                         feature_extractor,
                                         criterion,
                                         train=False,
                                         train_loader=None)
    test_acc, test_f1, test_f1_macro, test_precision, test_recall  = metrics(results, truths)
    msg = "\n\nTest Loss {:5.4f} | Test Acc {:5.4f} | Test f1-score {:5.4f} | Test f1-score-macro {:5.4f} | Test precision {:5.4f} | Test recall {:5.4f}".format(test_loss, test_acc, test_f1, test_f1_macro, test_precision, test_recall)
    print(msg)
    write_log(msg)
    sys.stdout.flush()

def main():
    args = get_args()

    torch.manual_seed(args.seed)
    dataset = str.lower(args.dataset.strip())

    use_cuda = False
    torch.set_default_tensor_type('torch.FloatTensor')
    if torch.cuda.is_available():
        if args.no_cuda:
            print("WARNING: You have a CUDA device, so you should probably not run with --no_cuda")
        else:
            torch.cuda.manual_seed(args.seed)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            use_cuda = True

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

    print("Start loading the data....")
    train_loader, train_size = data_loader(args, tokenizer, 'train')
    val_loader, val_size = data_loader(args, tokenizer, 'test')
    print('Finish loading the data....')

    hyp_params = args
    hyp_params.use_cuda = use_cuda
    hyp_params.dataset = dataset
    hyp_params.n_train = train_size
    hyp_params.n_valid = val_size
    hyp_params.model = args.model.strip()
    hyp_params.output_dim = output_dim_dict.get(dataset)
    hyp_params.criterion = criterion_dict.get(dataset)
    settings = initiate(hyp_params, tokenizer, train_loader, val_loader)

    train_model(settings, hyp_params, train_loader, val_loader)

if __name__ == '__main__':
    main()
