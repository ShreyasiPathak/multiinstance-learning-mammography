#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 22:20:38 2021

@author: spathak
"""

import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
from sklearn import utils
from collections import Counter
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.model_selection import GroupShuffleSplit

cbis_view_dic = {'LEFT_CC': 'LCC', 'RIGHT_CC': 'RCC', 'LEFT_MLO': 'LMLO', 'RIGHT_MLO': 'RMLO'}

from utilities import data_loaders_utils

def gradual_unfreezing(model,epoch):
    for name,param in model.named_parameters():
        if 'feature_extractor.fc' in name:# or 'fc2' in name or 'fc3' in name:
            if epoch>=0:
                param.requires_grad=True
            else:
                param.requires_grad=False
        elif 'layer4' in name:
            if epoch>=1:
                param.requires_grad=True
            else:
                param.requires_grad=False
        elif 'layer3' in name:
            if epoch>=2:
                param.requires_grad=True
            else:
                param.requires_grad=False
        elif 'layer2' in name:
            if epoch>=3:
                param.requires_grad=True
            else:
                param.requires_grad=False
        elif 'layer1' in name:
            if epoch>=4:
                param.requires_grad=True
            else:
                param.requires_grad=False
        elif 'feature_extractor.conv1' in name or 'feature_extractor.bn1' in name:
            if epoch>=5:
                param.requires_grad=True
            else:
                param.requires_grad=False
    return model

def layer_selection_for_training(model, epoch, trainingmethod,epoch_step):
    if trainingmethod=='multisteplr1':
        if epoch<epoch_step:
            for name,param in model.named_parameters():
                if 'feature_extractor.fc' in name:# or 'feature_extractor.fc2' in name or 'feature_extractor.fc3' in name:
                    param.requires_grad=True
                else:
                    param.requires_grad=False
        elif epoch==epoch_step:
            for name,param in model.named_parameters():
                param.requires_grad=True
            print("all layers unfrozen")
    return model

def views_distribution(config_params, df):
    views_allowed = data_loaders_utils.views_allowed_dataset(config_params)
    views_dic={}
    views_dic_allowed={}
    single_views_dic={}
    total=df.shape[0]
    for k in range(total):
        if k%5==0:
            print(str(k)+"/"+str(total))
        study_folder=str(df.iloc[k]['FullPath'])
        series_list=os.listdir(study_folder)
        views_list=[]
        views_list_allowed=[]
        for series in series_list:
            view_name=series.split('_')[0]
            if view_name not in views_list:
                views_list.append(view_name)
            if view_name in views_allowed and view_name not in views_list_allowed:
                views_list_allowed.append(view_name)
            single_views_dic[view_name]=single_views_dic.get(view_name,0)+1
                
        views_list.sort()
        views_joined='+'.join(views_list)
        #print(views_joined)
        views_dic[views_joined]=views_dic.get(views_joined,0)+1
        #print(views_dic)
        views_list_allowed.sort()
        views_joined_allowed='+'.join(views_list_allowed)
        #print(views_joined_allowed)
        views_dic_allowed[views_joined_allowed]=views_dic_allowed.get(views_joined_allowed,0)+1
        #print(views_dic_allowed)
        df.loc[k,['Views']]=views_joined_allowed
    print(df)
    pd.DataFrame.from_dict(views_dic,orient='index').to_excel('views_dic.xlsx')
    pd.DataFrame.from_dict(views_dic_allowed,orient='index').to_excel('views_dic_allowed.xlsx')
    pd.DataFrame.from_dict(single_views_dic,orient='index').to_excel('single_views_dic.xlsx')
    #df.to_csv('/home/spathak/multiview_mammogram/input_data/MG_training_files_studyUID_accessionNum_final.csv',sep=';',na_rep='NULL',index=False)   

def view_distribution_plot(filename):
    df=pd.read_excel(filename).sort_values(by=['Count'],ascending=False)
    print(df['Views'].tolist())
    print(df['Count'].tolist())
    plt.figure(figsize=(5,5))
    plt.bar(df['Views'].tolist(),df['Count'].tolist())
    plt.xticks(rotation=45,ha='right')
    plt.savefig('view_distribution.png', bbox_inches='tight') 

def stratified_class_count(df):
    class_count=df.groupby(by=['Groundtruth']).size()
    return class_count

def class_distribution_weightedloss(config_params, df):
    df_groundtruth=df['Groundtruth'].map(config_params['groundtruthdic'])
    print(np.array(config_params['classes']))
    print(df_groundtruth)
    class_weight=utils.class_weight.compute_class_weight(class_weight = 'balanced', classes = np.array(config_params['classes']), y = df_groundtruth)
    print("class count:", dict(Counter(df_groundtruth)))
    print("class weight:", class_weight)
    return torch.tensor(class_weight,dtype=torch.float32)

def class_distribution_poswt(df):
    class_count=df.groupby(by=['Groundtruth']).size()
    pos_wt=torch.tensor([float(class_count['benign'])/class_count['malignant']],dtype=torch.float32)
    print(pos_wt)
    return pos_wt

def groupby_view_train(df):
    df['Views'] = df['Views'].str.upper().str.replace(" ","")
    view_group = df.groupby(by=['Views'])
    view_group = list(view_group)
    view_group_indices={}
    view_group_names_dic={}
    for name, item in view_group:
        view_group_indices[name]=list(item.index.values)
        view_group_names_dic[name]=item.shape[0]
    return view_group_indices, view_group_names_dic

def groupby_view_test(df):
    df['Views'] = df['Views'].str.upper().str.replace(" ","")
    view_group = df.groupby(by=['Views'])
    view_group = list(view_group)
    #random.shuffle(view_group)
    j=0
    view_group_names_dic={}
    for name, item in view_group:
        if j==0:
            df_concat = item
        else:
            df_concat = pd.concat([df_concat,item])
        j+=1
        view_group_names_dic[name]=item.shape[0]
    return df_concat,view_group_names_dic

def stratifiedgroupsplit(df, rand_seed, patient_col):
    groups = df.groupby('Groundtruth')
    all_train = []
    all_test = []
    all_val = []
    train_testsplit = GroupShuffleSplit(test_size=0.15, n_splits=2, random_state=rand_seed)
    train_valsplit = GroupShuffleSplit(test_size=0.10, n_splits=2, random_state=rand_seed)
    for group_id, group in groups:
        # if a group is already taken in test or train it must stay there
        group = group[~group[patient_col].isin(all_train+all_val+all_test)]
        # if group is empty 
        if group.shape[0] == 0:
            continue
        train_inds1, test_inds = next(train_testsplit.split(group, groups=group[patient_col]))
        train_inds, val_inds = next(train_valsplit.split(group.iloc[train_inds1], groups=group.iloc[train_inds1][patient_col]))
    
        all_train += group.iloc[train_inds1].iloc[train_inds][patient_col].tolist()
        all_val += group.iloc[train_inds1].iloc[val_inds][patient_col].tolist()
        all_test += group.iloc[test_inds][patient_col].tolist()
        
    train = df[df[patient_col].isin(all_train)]
    val = df[df[patient_col].isin(all_val)]
    test = df[df[patient_col].isin(all_test)]
    
    '''
    form_train = set(train['Patient_Id'].tolist())
    form_val = set(val['Patient_Id'].tolist())
    form_test = set(test['Patient_Id'].tolist())
    inter1 = form_train.intersection(form_test)
    inter2 = form_train.intersection(form_val)
    inter3 = form_val.intersection(form_test)
    print(df.groupby('Groundtruth').size())
    print(train.groupby('Groundtruth').size())
    print(val.groupby('Groundtruth').size())
    print(test.groupby('Groundtruth').size())
    print(inter1) # this should be empty
    print(inter2) # this should be empty
    print(inter3) # this should be empty
    print(train[train['Patient_Id'].isin(test['Patient_Id'].unique().tolist())])
    print(test[test['Patient_Id'].isin(train['Patient_Id'].unique().tolist())])
    '''
    return train, val, test

def stratifiedgroupsplit_train_val(df, rand_seed, patient_col):
    groups = df.groupby('Groundtruth')
    all_train = []
    all_val = []
    train_valsplit = GroupShuffleSplit(test_size=0.10, n_splits=2, random_state=rand_seed)
    for group_id, group in groups:
        # if a group is already taken in test or train it must stay there
        group = group[~group[patient_col].isin(all_train+all_val)]
        # if group is empty 
        if group.shape[0] == 0:
            continue
        train_inds, val_inds = next(train_valsplit.split(group, groups=group[patient_col]))
    
        all_train += group.iloc[train_inds][patient_col].tolist()
        all_val += group.iloc[val_inds][patient_col].tolist()
        
    train = df[df[patient_col].isin(all_train)]
    val = df[df[patient_col].isin(all_val)]
    return train, val

def stratifiedgroupsplit_train_test(df, rand_seed, patient_col):
    groups = df.groupby('Groundtruth')
    all_train = []
    all_val = []
    train_valsplit = GroupShuffleSplit(test_size=0.15, n_splits=2, random_state=rand_seed)
    for group_id, group in groups:
        # if a group is already taken in test or train it must stay there
        group = group[~group[patient_col].isin(all_train+all_val)]
        # if group is empty 
        if group.shape[0] == 0:
            continue
        train_inds, val_inds = next(train_valsplit.split(group, groups=group[patient_col]))
    
        all_train += group.iloc[train_inds][patient_col].tolist()
        all_val += group.iloc[val_inds][patient_col].tolist()
        
    train = df[df[patient_col].isin(all_train)]
    val = df[df[patient_col].isin(all_val)]
    return train, val

def calculate_image_size(df):
    total=df.shape[0]
    w_all=[]
    h_all=[]
    count_less=0
    count_more=0
    count_wless=0
    count_wmore=0
    
    for k in range(total):
        if k%5==0:
            print(str(k)+"/"+str(total))
        data=df.iloc[k]
        #studyuid_path = str(df.iloc[k]['FullPath'])
        #series_list = os.listdir(studyuid_path)
        img, _ = data_loaders_utils.collect_images(data)
        w,h=img.size
        w_all.append(w)
        h_all.append(h)
        if w<1600 and h<1600:
            count_less+=1
        elif w>1600 and h>1600:
            count_more+=1
        elif w<1600 and h>1600:
            count_wless+=1
        elif w>1600 and h<1600:
            count_wmore+=1
    
    print("min w:", min(w_all))
    print("min h:", min(h_all))
    print("max w:", max(w_all))
    print("max h:", max(h_all))
    print("less than 1600,1600:",count_less)
    print("more than 1600,1600:",count_more)
    print("w less than 1600, h more than 1600:",count_wless)
    print("w more than 1600, h less than 1600:",count_wmore)
    w_mean_dataset = np.mean(np.array(w_all))
    w_std_dataset = np.std(np.array(w_all))
    h_mean_dataset = np.mean(np.array(h_all))
    h_std_dataset = np.std(np.array(h_all))
    return w_mean_dataset, w_std_dataset, h_mean_dataset, h_std_dataset

def calculate_dataset_mean_stddev(df, resize, transform):
    means = []
    stds = []
    total=df.shape[0]
    if transform:
        if resize:
            preprocess = transforms.Compose([
                transforms.Resize((resize[0],resize[1])),
                transforms.ToTensor()])
        else:
            preprocess = transforms.Compose([
                transforms.ToTensor()])
    
    for k in range(total):
        if k%5==0:
            print(str(k)+"/"+str(total))
        studyuid_path = str(df.iloc[k]['FullPath'])
        series_list = os.listdir(studyuid_path)
        image_list, _, _ = data_loaders_utils.collect_images(studyuid_path,series_list)
        for j,img in enumerate(image_list):
            if transform:
                img=preprocess(img)
            means.append(torch.mean(img))
            stds.append(torch.std(img))
    
    mean_dataset = torch.mean(torch.tensor(means))
    std_dataset = torch.mean(torch.tensor(stds))
    return mean_dataset, std_dataset

def save_model(model, optimizer, epoch, loss, path_to_model):
    state = {
        'epoch': epoch+1,
        'state_dict': model.state_dict(),
        'optim_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(state, path_to_model)

def load_model(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optim_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch

def confusion_matrix_norm_func(conf_mat,fig_name,class_name):
    #class_name=['W','N1','N2','N3','REM']
    conf_mat_norm=np.empty((conf_mat.shape[0],conf_mat.shape[1]))
    #conf_mat=confusion_matrix(y_true, y_pred)
    for i in range(conf_mat.shape[0]):
        conf_mat_norm[i,:]=conf_mat[i,:]/sum(conf_mat[i,:])
    #print(conf_mat_norm)
    print_confusion_matrix(conf_mat_norm,class_name,fig_name)
    
def print_confusion_matrix(conf_mat_norm, class_names_y, class_names_x, fig_name, figsize = (2,2), fontsize=5):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    #sns.set()
    #grid_kws = {"height_ratios": (.9, .05), "hspace": .3}
    fig, ax = plt.subplots(figsize=figsize)
    #cbar_ax = fig.add_axes([.93, 0.1, 0.05, 0.77])
    #fig = plt.figure(figsize=figsize)
    heatmap=sns.heatmap(
        yticklabels=class_names_y,
        xticklabels=class_names_x,
        data=conf_mat_norm,
        ax=ax,
        cmap='YlGnBu',
        cbar=False,
        #cbar_ax=cbar_ax,
        annot=True,
        annot_kws={'size':fontsize},
        fmt="d",
        square=True
        #linewidths=0.75
        )
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='center', fontsize=fontsize)
    ax.set_ylabel('True',labelpad=0,fontsize=fontsize)
    ax.set_xlabel('Predicted',labelpad=3,fontsize=fontsize)
    #cbar_ax.tick_params(labelsize=fontsize) 
    #ax.get_yaxis().set_visible(False)
    #plt.tight_layout()
    #plt.show()
    #ax.set_title(fig_name)
    fig.savefig(fig_name, format='pdf', bbox_inches='tight')    

def crosscheck_view_collect_images(config_params, df):
    views_allowed = data_loaders_utils.views_allowed_dataset(config_params)
    #collect images for the model
    length=len(df.index)
    i=0
    for idx in df.index:
        print(i,'/',length)
        data=df.loc[idx] #loc is valid because I have reset_index in df_train, df_val and df_test. random data sampling returns index, but as our index is same as the relative position, bpth iloc and loc should work. I am using loc because of groupby view data sampler. 
        studyuid_path=str(data['FullPath'])
        series_list=os.listdir(studyuid_path)
        views_saved=[]
        series_list.sort()
        for series in series_list:
            series_path=studyuid_path+'/'+series
            img_list=os.listdir(series_path)
            for image in img_list:
                img_path = series_path+'/'+image
                img = Image.open(img_path)
                print(img.size)
                series_des=series.split('_')[0].upper().replace(" ","")
                if series_des in views_allowed and series_des not in views_saved:
                    views_saved.append(series_des)
                    #image_read_list.append(img)
                    #breast_side.append(series[0])
        if data['Views']!='+'.join(views_saved):
            print(idx)
            input('halt')
        i+=1

#fig_name = 'C:/Users/PathakS/OneDrive - Universiteit Twente/PhD/projects/radiology breast cancer/codes/breast-cancer-multiview-mammogram-codes/multiinstance results/results/NextSubmission/roi-diagnosis-esattside-confmat-vindr1.pdf'
#conf_mat = [[ 93 , 56, 20, 1], [ 53, 175, 9, 16]]
#conf_mat = [[ 79 , 67, 20, 3], [ 54, 178, 8, 15]] #silil
#conf_mat = [[ 54 , 30, 6, 1], [ 30, 63, 2, 1]] # esattside
#conf_mat = [[ 0 , 70, 0, 21], [ 0, 85, 0, 11]] # ismeanatt
#print_confusion_matrix(conf_mat, ['B-Case', 'M-Case'], ['B-Case + \nROI', 'M-Case +\nROI', 'B-Case + \n R\u0336O\u0336I\u0336', 'M-Case +\nNo-ROI'], fig_name, figsize = (5,5), fontsize=10)