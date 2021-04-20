# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 20:20:12 2021

@author: ASUS
"""
import argparse

def get_args():

    # construct argument parser
    parser = argparse.ArgumentParser(description='Memotion Humor Detection')

    # Need to modify 
    parser.add_argument('--train_reddit', type=int, default=1,
                        help='train reddit model (default: 1)')
    parser.add_argument('--dataset', type=str, default='reddit',
                        help='dataset to use (default: memotion)')
    parser.add_argument('--model', type=str, default='RedditAlbert',
                        help='name of the model to use (Transformer, etc.)')
    parser.add_argument('--bert_model', type=str, default="albert-base-v2",
                        help='pretrained bert model to use')
    # parser.add_argument('--bert_model', type=str, default="bert-base-uncased",
    #                    help='pretrained bert model to use')

    # Dataset
    parser.add_argument('--root_path', type=str, default='./data',
                        help='path for storing the dataset')

    # Multimodal Model Architecture
    parser.add_argument('--cnn_model', type=str, default="vgg16",
                        help='pretrained CNN to use for image feature extraction')
    parser.add_argument('--image_feature_size', type=int, default=4096,
                        help='image feature size extracted from pretrained CNN (default: 4096)')
    parser.add_argument('--bert_hidden_size', type=int, default=768,
                        help='bert hidden size for each word token (default: 768)')
    parser.add_argument('--mlp_dropout', type=float, default=0.1,
                        help='fully connected layers dropout')

    # Network Hyper-parameters
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='batch size (default: 2)')
    parser.add_argument('--max_token_length', type=int, default=128,
                        help='max number of tokens per sentence (default: 50)')
    parser.add_argument('--clip', type=float, default=0.8,
                        help='gradient clip value (default: 0.8)')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='initial learning rate (default: 2e-5)')
    parser.add_argument('--optim', type=str, default='AdamW',
                        help='optimizer to use (default: AdamW)')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='number of epochs (default: 3)')
    parser.add_argument('--when', type=int, default=2,
                        help='when to decay learning rate (default: 2)')

    # Logistics
    parser.add_argument('--log_interval', type=int, default=100,
                        help='frequency of result logging (default: 100)')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--no_cuda', action='store_true',
                        help='do not use cuda')
    parser.add_argument('--name', type=str, default='model',
                        help='name of the trial (default: "model")')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers to use for DataLoaders (default: 6)')

    args = parser.parse_args()
    return args
