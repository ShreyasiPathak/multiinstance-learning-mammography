# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 16:23:57 2022

@author: PathakS
"""

import os
import math
import torch

import numpy as np
import pandas as pd
import openpyxl as op
import torch.nn as nn

import itertools
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from pytorchtools import EarlyStopping
from sklearn.metrics import confusion_matrix
from torchvision.models.resnet import BasicBlock
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from openpyxl import Workbook

import utils
#import model
import tensorboard_log

def results_viewwise(sheet4, val_stats_viewwise):
    header=[]
    if val_stats_viewwise!={}:
        for key in val_stats_viewwise.keys():
            header=header+[key,'','','']
        sheet4.append(header)
        sheet4.append(['Count','Recall','Specificity','Acc']*len(list(val_stats_viewwise.keys())))
    row_sheet4=list(itertools.chain(*val_stats_viewwise.values()))
    sheet4.append(row_sheet4)
    return sheet4

def loss_fn(activation):
    if activation=='softmax':
        criterion=nn.CrossEntropyLoss()
    elif activation=='sigmoid':
        criterion = nn.BCEWithLogitsLoss()
    return criterion

def conf_mat_create(predicted,true,correct,total_images,conf_mat,classes):
    total_images+=true.size()[0]
    correct += predicted.eq(true.view_as(predicted)).sum().item()
    conf_mat_batch=confusion_matrix(true.cpu().numpy(),predicted.cpu().numpy(),labels=classes)
    conf_mat=conf_mat+conf_mat_batch
    return correct, total_images,conf_mat,conf_mat_batch

'''def load_model(model,path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    #optimizer.load_state_dict(checkpoint['optim_dict'])
    return model'''

def case_label_from_SIL(df_test, test_labels_all, test_pred_all, sheet3):
    dic_true={}
    dic_pred={}
    idx=0
    
    test_pred_all = test_pred_all.reshape(-1)
    print(test_labels_all)
    print(test_pred_all)
    for idx in df_test.index:
        dic_key = '_'.join(df_test.loc[idx,'ImageName'].split('_')[:3])
        dic_true[dic_key] = max(dic_true.get(dic_key,0),test_labels_all[idx])
        dic_pred[dic_key] = max(dic_pred.get(dic_key,0),test_pred_all[idx])
    case_labels_true = np.array(list(dic_true.values()))
    case_labels_pred = np.array(list(dic_pred.values()))

    print(case_labels_true)
    print(case_labels_pred)
    metrics_case_labels = utils.performance_metrics(None, case_labels_true, case_labels_pred, None)
    print(metrics_case_labels)
    sheet3.append(['Precision','Recall','Specificity','F1','F1macro','F1wtmacro','Acc','Bal_Acc','Cohens Kappa','AUC'])
    sheet3.append(metrics_case_labels)
    return sheet3


def test(model, data_iterator_test, batches_test, activation, sheet2, sheet3, sheet4, device, classes, df_test):
    """Testing"""
    model.eval()
    total_images=0
    test_loss = 0
    correct = 0
    s=0
    batch_test_no=0
    #count_dic_viewwise={}
    #conf_mat_viewwise={}
    #val_stats_viewwise={}
    conf_mat_test=np.zeros((2,2))
    
    lossfn1 = loss_fn(activation)
    
    with torch.no_grad():
        for test_idx, test_batch, test_labels in data_iterator_test:
            #print("test_idx",test_idx)
            #input('halt')
            test_batch, test_labels = test_batch.to(device), test_labels.to(device)
            test_labels=test_labels.view(-1)
            #print("test batch:",test_batch.shape)
            #print("test labels:",test_labels.shape)
            #with torch.cuda.amp.autocast():
            output_test = model(test_batch)
            if activation=='sigmoid':
                output_test = output_test.squeeze(1)
                output_test = output_test.view(-1)                                                 
                test_labels = test_labels.float()
                test_pred = torch.ge(torch.sigmoid(output_test), torch.tensor(0.5)).float()
                loss1 = lossfn1(output_test, test_labels).item()
            elif activation=='softmax':
                test_pred = output_test.argmax(dim=1, keepdim=True)
                loss1 = lossfn1(output_test, test_labels).item()
            
            if batch_test_no==0:
                test_pred_all=test_pred
                test_labels_all=test_labels
                print(output_test.data.shape)
                if activation=='sigmoid':
                    output_all_ten=torch.sigmoid(output_test.data)
                elif activation=='softmax':
                    output_all_ten=F.softmax(output_test.data,dim=1)
                    output_all_ten=output_all_ten[:,1]
            else:
                test_pred_all=torch.cat((test_pred_all,test_pred),dim=0)
                test_labels_all=torch.cat((test_labels_all,test_labels),dim=0)
                if activation=='sigmoid':
                    output_all_ten=torch.cat((output_all_ten,torch.sigmoid(output_test.data)),dim=0)
                elif activation=='softmax':
                    output_all_ten=torch.cat((output_all_ten,F.softmax(output_test.data,dim=1)[:,1]),dim=0)
            
            test_loss += test_labels.size()[0]*loss1 # sum up batch loss
            correct,total_images,conf_mat_test,conf_mat_batch=conf_mat_create(test_pred,test_labels,correct,total_images,conf_mat_test,classes)
            
            '''views_names_key="+".join(views_names)
            #loss_dic_viewwise[views_names_key]=loss_dic_viewwise.get(views_names_key,0)+test_labels.size()[0]*loss1
            count_dic_viewwise[views_names_key]=count_dic_viewwise.get(views_names_key,0)+test_batch.shape[0]
            conf_mat_viewwise[views_names_key]=conf_mat_viewwise.get(views_names_key,np.zeros((2,2)))+conf_mat_batch
            count_dic_viewwise[str(len(views_names))]=count_dic_viewwise.get(str(len(views_names)),0)+test_batch.shape[0]
            conf_mat_viewwise[str(len(views_names))]=conf_mat_viewwise.get(str(len(views_names)),np.zeros((2,2)))+conf_mat_batch'''
            batch_test_no+=1
            s=s+test_labels.shape[0]
            print ('Test: Step [{}/{}], Loss: {:.4f}'.format(batch_test_no, batches_test, loss1))
    
    


    '''for key in count_dic_viewwise.keys():
        if count_dic_viewwise[key]:
            count_key=count_dic_viewwise[key]
        else:
            count_key=0
        if sum(conf_mat_viewwise[key][1,:]):
            tpr=float(conf_mat_viewwise[key][1,1])/sum(conf_mat_viewwise[key][1,:])
        else:
            tpr=0.0
        if sum(conf_mat_viewwise[key][0,:]):
            tnr=float(conf_mat_viewwise[key][0,0])/sum(conf_mat_viewwise[key][0,:])
        else:
            tnr=0.0
        if np.sum(conf_mat_viewwise[key]):
            acc=float(conf_mat_viewwise[key][0,0]+conf_mat_viewwise[key][1,1])/np.sum(conf_mat_viewwise[key])
        else:
            acc=0.0
        val_stats_viewwise[key]=[count_key,tpr,tnr,acc]'''
        #print(key, 'benign', sum(conf_mat_viewwise[key][0,:]))
        #print(key, 'malignant', sum(conf_mat_viewwise[key][1,:]))
    
    running_loss = test_loss/total_images
    print("conf_mat_test:",conf_mat_test)
    print("total_images:",total_images)
    print("s:",s)
    print('\nTest set: total test loss: {:.4f}, Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) \n'.format(
        test_loss, running_loss, correct, total_images, 100. * correct / total_images))
    
    per_model_metrics = utils.performance_metrics(conf_mat_test,test_labels_all.cpu().numpy(),test_pred_all.cpu().numpy(), output_all_ten.cpu().numpy())
    per_model_metrics = [running_loss] + per_model_metrics
    print(per_model_metrics)
    
    #sheet4 = results_viewwise(sheet4, val_stats_viewwise)
    
    #sheet3.append(['case-level prediction'])
    #sheet3 = case_label_from_SIL(df_test, test_labels_all.cpu().numpy(), test_pred_all.cpu().numpy(), sheet3)
    
    return per_model_metrics, conf_mat_test

'''def run_test(df_test, modality, preprocess_test, batch_size1, num_workers, batches_test, activation, sheet3, sheet4, path_to_model, device, feature_extractor, attention_pooling, classes, milpooling, extra, datascaling, resize, attention, batch_sampler_test, flipimage, inchans):
    #model initialization
    #if feature_extractor=='common' and attention_pooling!='breast-wise':
    #    model1 = model.MILmodel(mil_pooling, activation, device, extra)
    #elif feature_extractor=='view-wise' or attention_pooling=='breast-wise':
    #model1 = model.SeparatePipelineMIL(milpooling, activation, device, extra, attention, feature_extractor)
    model1 = model.SILmodel(activation, extra)
    model1.to(device)
    
    #if feature_extractor=='common' and attention_pooling!='breast-wise':
    #    dataset_gen_test = utils.BreastCancerDataset_generator(df_test,modality, datascaling, resize, preprocess_test)
    #    dataloader_test = DataLoader(dataset_gen_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=utils.MyCollate)
    #elif feature_extractor=='view-wise' or attention_pooling=='breast-wise':
    dataset_gen_test = utils.BreastCancerDataset_generator(df_test, modality, datascaling, resize, flipimage, inchans, preprocess_test)
    dataloader_test = DataLoader(dataset_gen_test, batch_size=batch_size1, shuffle=False, num_workers=num_workers, collate_fn=utils.MyCollate, batch_sampler=batch_sampler_test)

    
    #writer=tensorboard_log.tensorboard_log(max_epochs, file_name, dataloader_train, model)
    
    #batches_test=int(math.ceil(test_instances/batch_size))
    
    #testing
    path_to_trained_model=path_to_model
    model1 = load_model(model1, path_to_trained_model)
    sheet3, sheet4 = test(model1, dataloader_test, batches_test, activation, sheet3, sheet4, device, classes)
    
    return sheet3, sheet4
'''