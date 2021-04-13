# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 23:39:39 2021

@author: ASUS
"""
import pickle
import pandas as pd
import numpy as np
import os
import cv2

humour_scores = {'not_funny': 0 ,
                 'funny': 1,
                 'very_funny' : 1,
                 'hilarious' : 1
                 }

humour_scores_test = {'0': 0 ,
                 '1': 1,
                 '2' : 1,
                 '3' : 1
                 }

def read_train_text(train_root_dir):
    
    # train_text_path = str(train_root_dir + "/labels_pd_pickle")
    # train_df = pd.read_pickle(train_text_path)
    
    train_text_path = str(train_root_dir + "/labels.csv")
    train_df = pd.read_csv(train_text_path, encoding="utf-8")
    train_humour_df = train_df.iloc[:, [1,3,4]].copy()
    train_humour_df['humour'] = train_humour_df['humour'].map(humour_scores)
    
    # remove row(s) with at-least one missing value
    train_humour_df = train_humour_df.dropna()

    return train_humour_df
    
def read_test_text(test_root_dir):
    test_text_path = os.path.join(test_root_dir, "2000_testdata.csv")
    test_label_path = os.path.join(test_root_dir, "Meme_groundTruth.csv")
    
    # accessing data csv
    test_data_df = pd.read_csv(test_text_path, header='infer')
    test_data_clean_df = test_data_df.iloc[:, [0,3]].copy()
    
    # accessing label csv
    test_label_df = pd.read_csv(test_label_path, header='infer')
       
    # spliting label column into three task labels &
    # taking only the task b label
    foo = lambda x: pd.Series([i for i in reversed(x.split('_'))])
    sub_labels = test_label_df['Labels'].apply(foo).iloc[:,0]
    
    # taking the first character representing the humour labels
    test_data_clean_df['humour'] = sub_labels.astype(str).str[0]
    
    test_data_clean_df['humour'] = test_data_clean_df['humour'].map(humour_scores_test)
    
    return test_data_clean_df
    
    
def save_image(img_list, input_img_dir, output_img_dir):
    
    if not os.path.isdir(output_img_dir):
        os.makedirs(output_img_dir)
        
    for img in img_list:
        input_img_path = os.path.join(input_img_dir, img)
        output_img_path = os.path.join(output_img_dir, img)   
        
        image = cv2.imread(input_img_path)
        if image is not None:
            # cv2.imwrite(output_img_path, image)
            a=1
        else:
            print(img)
          
if __name__=='__main__':
    
    # define input and output directory for both training and testing dataset
    in_train_dir = "../../../Feb8/raw_dataset/memotion_dataset_7k"
    in_test_dir = "../../../Feb8/raw_dataset/memotion_dataset_7k_test"
    
    out_train_dir = "../../data/memotion/train"
    out_test_dir = "../../data/memotion/test"
    
    if not os.path.isdir(out_train_dir):
        os.makedirs(out_train_dir)
        
    if not os.path.isdir(out_test_dir):
        os.makedirs(out_test_dir)
        
    # load training and test dataframe
    train_df = read_train_text(in_train_dir)
    test_df = read_test_text(in_test_dir)
    
    # save training images and texts
    train_img_list = train_df.iloc[:,0]
    in_train_img_dir = os.path.join(in_train_dir, "images")
    out_train_img_dir = os.path.join(out_train_dir, "images")
    # save_image(train_img_list, in_train_img_dir, out_train_img_dir)
    out_train_csv = str(out_train_dir+ "/train.tsv")
    # train_df.iloc[:,0:].to_csv(out_train_csv, sep='\t',encoding="utf-8", index=False, header=True)
    
    # save test images and texts
    test_img_list = test_df.iloc[:,0]
    in_test_img_dir = os.path.join(in_test_dir, "2000_data")
    out_test_img_dir = os.path.join(out_test_dir, "images")
    # save_image(test_img_list, in_test_img_dir, out_test_img_dir)
    out_test_csv = str(out_test_dir+ "/test.tsv")
    test_df.iloc[:,0:].to_csv(out_test_csv, sep='\t',encoding="utf-8", index=False, header=True)
 
    
    
    