#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 22:20:38 2021

@author: spathak
"""

from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torchvision import transforms
import pandas as pd
import os
import pickle
import torch
import math
import copy
#from skimage import io
from PIL import Image
from PIL import ImageOps
import numpy as np
import sys
import glob
import random
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
from sklearn import metrics, utils
from collections import Counter
import operator
from sklearn.model_selection import GroupShuffleSplit
from torch.autograd import Variable
from copy import deepcopy
import imageio
import re

from data_loading import augmentations, loading


groundtruth_dic = {'benign':0,'malignant':1}
inverted_groundtruth_dic = {0:'benign',1:'malignant'}
#views_allowed = ['LCC','LMLO','RCC','RMLO']
views_allowed_gmic = ['L-CC','L-MLO','R-CC','R-MLO']
cbis_view_dic = {'LEFT_CC': 'LCC', 'RIGHT_CC': 'RCC', 'LEFT_MLO': 'LMLO', 'RIGHT_MLO': 'RMLO'}
#optimizer_params_dic={'.mlo':0,'.cc':1,'_left.attention':2,'_right.attention':3,'_both.attention':4}
#cluster_data_path_prefix='/local/work/spathak'

class MyCrop:
    """Randomly crop the sides."""

    def __init__(self, left=100,right=100,top=100,bottom=100):
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom

    def __call__(self, x):
        width, height=x.size
        size_left = random.randint(0,self.left)
        size_right = random.randint(width-self.right,width)
        size_top = random.randint(0,self.top)
        size_bottom = random.randint(height-self.bottom,height)
        x = TF.crop(x,size_top,size_left,size_bottom,size_right)
        return x
    
class MyGammaCorrection:
    def __init__(self, factor=0.2):
        self.lb = 1-factor
        self.ub = 1+factor

    def __call__(self, x):
        gamma = random.uniform(self.lb,self.ub)
        return TF.adjust_gamma(x,gamma)

def myhorizontalflip(image,breast_side):
    if breast_side=='R':
        image = np.fliplr(image).copy() #R->L (following GMIC)
    return image

class MyHorizontalFlip:
    """Flip horizontally."""

    def __init__(self):
        pass

    def __call__(self, x, breast_side):
        #if breast_side=='L':
        #    return TF.hflip(x) #L->R
        if breast_side=='R':
            return TF.hflip(x) #R->L (following GMIC)
        else:
            return x

class MyPadding:
    def __init__(self, breast_side, max_height, max_width, height, width):
        self.breast_side = breast_side
        self.max_height=max_height
        self.max_width=max_width
        self.height=height
        self.width=width
          
    def __call__(self,img):
        print(img.shape)
        print(self.max_height-self.height)
        if self.breast_side=='L':
            image_padded=F.pad(img,(0,self.max_width-self.width,0,self.max_height-self.height,0,0),'constant',0)
        elif self.breast_side=='R':
            image_padded=F.pad(img,(self.max_width-self.width,0,0,self.max_height-self.height,0,0),'constant',0)
        print(image_padded.shape)
        return image_padded

class MyPaddingLongerSide:
    def __init__(self, resize):
        self.max_height=resize[0]
        self.max_width=resize[1]
        
    def __call__(self,img):#,breast_side):
        width=img.size[0]
        height=img.size[1]
        if height<self.max_height:
            diff=self.max_height-height
            img=TF.pad(img,(0,math.floor(diff/2),0,math.ceil(diff/2)),0,'constant')
        if width<self.max_width:
            diff=self.max_width-width
            img=TF.pad(img,(diff,0,0,0),0,'constant')
        return img
        
class BreastCancerDataset_generator(Dataset):
    def __init__(self, config_params, df, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df
        self.transform = transform
        self.config_params = config_params

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        if self.config_params['learningtype'] == 'SIL':
            data = self.df.iloc[idx]
            img = collect_images(self.config_params, data)
            if self.transform:
                img=self.transform(img)
            #print("after transformation:",img.shape)
            if self.config_params['channel'] == 1:
                img=img[0,:,:]
                img=img.unsqueeze(0).unsqueeze(1)
            elif self.config_params['channel'] == 3:
                img=img.unsqueeze(0)
            return idx, img, torch.tensor(groundtruth_dic[data['Groundtruth']]), data['Views']
        
        elif self.config_params['learningtype'] == 'MIL':
            flag = 0
            data = self.df.loc[idx] #loc is valid because I have reset_index in df_train, df_val and df_test. random data sampling returns index, but as our index is same as the relative position, bpth iloc and loc should work. I am using loc because of groupby view data sampler. 
            image_list, _, views_saved = collect_cases(self.config_params, data)
            for j,img in enumerate(image_list):
                if self.transform:
                    img=self.transform(img)
                if self.config_params['channel'] == 1:
                    img = img[0,:,:]
                    img = img.unsqueeze(0)
                img = img.unsqueeze(0)
                if flag==0:
                    image_tensor = img
                    flag = 1
                else:
                    image_tensor = torch.cat((image_tensor,img),0)
            return idx, image_tensor, torch.tensor(groundtruth_dic[data['Groundtruth']]), views_saved 

class CustomWeightedRandomSampler(Sampler):
    def __init__(self, df_train):
        self.df = df_train
    
    def __iter__(self):
        len_instances=self.df.shape[0]
        OriginalIndices=np.array(range(0,len_instances))
        class_count=stratified_class_count(self.df)
        class_count=class_count.to_dict()
        class_count[0] = class_count.pop('benign')
        class_count[1] = class_count.pop('malignant')
        max_class = max(class_count.items(), key=operator.itemgetter(1))
        diff_count_class=max_class[1]-np.array([class_count[0],class_count[1]])
        labels=self.df['Groundtruth'].replace(groundtruth_dic)
        #print(np.append(OriginalIndices,np.random.choice(np.where(labels==0)[0],5),axis=0))
        for i in range(0,2):
            if i!=max_class[0]:
                repeater=int(diff_count_class[i]/class_count[i])
                OriginalIndices=np.append(OriginalIndices,np.repeat(np.where(labels==i),repeater))
                leftover=diff_count_class[i]-(class_count[i]*repeater)
                OriginalIndices=np.append(OriginalIndices,np.random.choice(np.where(labels==i)[0],leftover),axis=0)
                #OriginalIndices=np.append(OriginalIndices,np.random.choice(np.where(labels==i)[0],diff_count_class[i]),axis=0)
        random.shuffle(OriginalIndices)
        iter_shuffledIndex=iter(OriginalIndices)
        return iter_shuffledIndex
    
    def __len__(self):
        class_count=stratified_class_count(self.df)
        class_count=class_count.to_dict()
        max_class=max(class_count.items(), key=operator.itemgetter(1))
        len_instances_oversample=max_class[1]*2
        #print("Weighted Sampler __len__:",len_instances_oversample)
        return len_instances_oversample

class CustomGroupbyViewRandomSampler(Sampler):
    def __init__(self, df_train, subset):
        self.df = df_train
        self.groups = self.df.groupby(by=['Views'])
        self.view_group = list(self.groups)
        self.subset = subset
    
    def __iter__(self):
        j=0
        for name, item in self.view_group:
            #print("Sampler:",name)
            indices=item.index.values
            #print("indices:",indices)
            if self.subset == 'train':
                random.shuffle(indices)
            #print("shuffled indices:",indices)
            if j==0:
                shuffled_indices = indices
            else:
                shuffled_indices = np.append(shuffled_indices,indices)
            j+=1
        #print("whole indices:",shuffled_indices)
        #input('halt')
        iter_shuffledIndices=iter(shuffled_indices)
        return iter_shuffledIndices
    
    def __viewlength__(self):
        view_group_length={}
        view_group_name = []
        for name, item in self.view_group:
           indices=item.index.values
           view_group_length[name]=len(indices)
           view_group_name.append(name)
        #print("__view_length__:",view_group_length)
        #print("__view_length__:",view_group_name)
        return view_group_length, view_group_name

class CustomGroupbyViewRandomBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, view_group_length_dic, view_group_name):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        self.sampler = sampler
        self.batch_size = batch_size
        self.view_group_length_dic = view_group_length_dic
        self.view_group_name = view_group_name
    
    def __iter__(self):
        batch = []
        len_tillCurrent=0
        file_current=0
        batch_no=0
        for idx in self.sampler:
            batch.append(idx)
            len_tillCurrent=len_tillCurrent+1
            if len(batch) == self.batch_size or len_tillCurrent==self.view_group_length_dic[self.view_group_name[file_current]]:
                #print("batch_sampler:",batch)
                batch_no=batch_no+1
                yield batch
                batch = []
            if len_tillCurrent==self.view_group_length_dic[self.view_group_name[file_current]]:
                len_tillCurrent = 0
                file_current=file_current+1
    
    def __len__(self):
        length=np.sum(np.ceil(np.array(list(self.view_group_length_dic.values()))/self.batch_size),dtype='int32')
        print("length in batch sampler:",length)
        return length

class CustomGroupbyViewFullRandomSampler(Sampler):
    def __init__(self, view_group_dic, batch_size, subset):
        self.view_group_dic = view_group_dic
        self.view_group_names = list(self.view_group_dic.keys())
        self.subset = subset
        self.batch_size = batch_size
    
    def __iter__(self):
        total_indices=[]
        view_group_name_dup = copy.deepcopy(self.view_group_names)
        view_group_dic_dup = copy.deepcopy(self.view_group_dic)
        #print(view_group_dic_dup)
        while view_group_name_dup:    
            key=random.sample(view_group_name_dup,1)[0]
            #print(key)
            if len(view_group_dic_dup[key])>self.batch_size:
                selected_index=random.sample(view_group_dic_dup[key],self.batch_size)
                insert_val=zip(selected_index,[key]*self.batch_size)
                total_indices.extend(insert_val)
                view_group_dic_dup[key] = [num for num in view_group_dic_dup[key] if num not in selected_index]
                #print(selected_index)
                #print(view_group_dic_dup[key])    
            else:
                total_indices.extend(zip(view_group_dic_dup[key],[key]*len(view_group_dic_dup[key])))
                view_group_dic_dup.pop(key,None)
                view_group_name_dup.pop(view_group_name_dup.index(key))
                view_group_name_dup=list(view_group_dic_dup.keys())
            #print(total_indices)
            #input('halt')
        #print(total_indices)
        
        #print(total_indices)
        #print("whole indices:",np.array(total_indices)[:,0])
        #input('halt')
        iter_shuffledIndices=iter(total_indices)
        return iter_shuffledIndices
    
    def __viewlength__(self):
        view_group_length={}
        for name, item in self.view_group_dic.items():
           view_group_length[name]=len(item)
        return view_group_length

class CustomGroupbyViewFullRandomBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, view_group_length_dic):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        self.sampler = sampler
        self.batch_size = batch_size
        self.view_group_length_dic = view_group_length_dic
    
    def __iter__(self):
        batch = []
        batch_no=0
        ct=0
        for idx,view_name in self.sampler:
            if ct==0:
                view_name_previous=view_name
            if len(batch) == self.batch_size or view_name!=view_name_previous: 
                #print("batch_sampler:",batch)
                batch_no+=1
                yield batch
                batch = []
            batch.append(idx)
            view_name_previous=view_name
            ct+=1
            if ct==sum(list(self.view_group_length_dic.values())):
                batch_no+=1
                yield batch
                batch = []
    
    def __len__(self):
        length=np.sum(np.ceil(np.array(list(self.view_group_length_dic.values()))/self.batch_size),dtype='int32')
        print("length in batch sampler:",length)
        return length

class CustomGroupbyViewWeightedRandomSampler(Sampler):
    def __init__(self, df_train):
        self.df = df_train
        self.view_group = list(self.df.groupby(by=['Views']))
    
    def __iter__(self):
        len_instances=self.df.shape[0]
        OriginalIndices=np.array(range(0,len_instances))
        class_count=stratified_class_count(self.df)
        class_count=class_count.to_dict()
        class_count[0] = class_count.pop('benign')
        class_count[1] = class_count.pop('malignant')
        max_class = max(class_count.items(), key=operator.itemgetter(1))
        diff_count_class=max_class[1]-np.array([class_count[0],class_count[1]])
        labels=self.df['Groundtruth'].replace(groundtruth_dic)
        for i in range(0,2):
            if i!=max_class[0]:
                repeater=int(diff_count_class[i]/class_count[i])
                OriginalIndices=np.append(OriginalIndices,np.repeat(np.where(labels==i),repeater))
                leftover=diff_count_class[i]-(class_count[i]*repeater)
                OriginalIndices=np.append(OriginalIndices,np.random.choice(np.where(labels==i)[0],leftover),axis=0)
        
        j=0
        for name, item in self.view_group:
            #print("Sampler:",name)
            indices = item.index.values
            print("indices:",indices)
            indices = OriginalIndices[np.where(np.isin(OriginalIndices,indices))]
            print("indices:",indices)
            random.shuffle(indices)
            self.view_group_length[name]=len(indices)
            self.view_group_name.append(name)
            #print("shuffled indices:",indices)
            if j==0:
                shuffled_indices = indices
            else:
                shuffled_indices = np.append(shuffled_indices,indices)
            j+=1
        #print("whole indices:",shuffled_indices)
        iter_shuffledIndices=iter(shuffled_indices)
        return iter_shuffledIndices
            
    def __viewlength__(self):
        view_group_length={}
        view_group_name = []
        len_instances=self.df.shape[0]
        OriginalIndices=np.array(range(0,len_instances))
        class_count=stratified_class_count(self.df)
        class_count=class_count.to_dict()
        class_count[0] = class_count.pop('benign')
        class_count[1] = class_count.pop('malignant')
        max_class = max(class_count.items(), key=operator.itemgetter(1))
        diff_count_class=max_class[1]-np.array([class_count[0],class_count[1]])
        labels=self.df['Groundtruth'].replace(groundtruth_dic)
        #print(np.append(OriginalIndices,np.random.choice(np.where(labels==0)[0],5),axis=0))
        for i in range(0,2):
            if i!=max_class[0]:
                repeater=int(diff_count_class[i]/class_count[i])
                OriginalIndices=np.append(OriginalIndices,np.repeat(np.where(labels==i),repeater))
                leftover=diff_count_class[i]-(class_count[i]*repeater)
                OriginalIndices=np.append(OriginalIndices,np.random.choice(np.where(labels==i)[0],leftover),axis=0)
        
        for name, item in self.view_group:
            #print("Sampler:",name)
            indices = item.index.values
            indices = OriginalIndices[np.where(np.isin(OriginalIndices,indices))]
            random.shuffle(indices)
            view_group_length[name]=len(indices)
            view_group_name.append(name)
        return view_group_length, view_group_name

def MyCollate(batch):
    i=0
    index=[]
    target=[]
    for item in batch:
        if i==0:
            data = batch[i][1]
            views_names = [item[3]]
        else:
            data=torch.cat((data,batch[i][1]),dim=0)
            views_names.append(item[3])
        index.append(item[0])
        target.append(item[2])
        i+=1
    index = torch.LongTensor(index)
    target = torch.LongTensor(target)
    return [index, data, target, views_names]

def MyCollateBreastWise(batch):
    i=0
    index=[]
    target=[]
    views_names_list=[]
    if batch[0][1].shape[1] == 3:
        data=torch.zeros(len(batch), len(batch[0][3]), batch[0][1].shape[1], batch[0][1].shape[2], batch[0][1].shape[3])
    elif batch[0][1].shape[1] == 1:
        data=torch.zeros(len(batch), len(batch[0][3]), batch[0][1].shape[2], batch[0][1].shape[3])
    for item in batch:
        views_names = item[3]
        if batch[i][1].shape[1]==3:
            data[i,:,:,:,:] = batch[i][1]
        elif batch[i][1].shape[1]==1:
            data[i,:,:,:] = batch[i][1].squeeze(1)
        index.append(item[0])
        target.append(item[2])
        views_names_list.append(item[3])
        i+=1
    index = torch.LongTensor(index)
    target = torch.LongTensor(target)
    return [index, data, target, list(np.unique(views_names_list))]

##--------------------------------------------------------------- collect image ----------------------------------------------------#
def view_extraction(series_list, views_allowed):
    series_list = list(map(lambda x: [x.split('_')[0].replace(' ', '').upper(), x], series_list))
    series_list = list(filter(lambda x: x[0] in views_allowed, series_list))
    series_list = sorted(series_list, key=lambda x: x[0])
    return series_list

def selecting_data(config_params, img_list):
    if config_params['dataset'] == 'zgt':
        if config_params['bitdepth'] == 8:
            img_list = list(filter(lambda x: re.search('_processed.png$', x), img_list))
        elif config_params['bitdepth'] == 12:
            img_list = list(filter(lambda x: re.search('_processed.npy$', x), img_list))
    return img_list

def views_allowed_dataset(config_params):
    if config_params['dataset'] == 'zgt' and config_params['viewsinclusion'] == 'all':
        views_allowed=['LCC', 'LLM', 'LML', 'LMLO', 'LXCCL', 'RCC', 'RLM', 'RML', 'RMLO', 'RXCCL']
    else:
        views_allowed = ['LCC','LMLO','RCC','RMLO']
    return views_allowed

def collect_cases(config_params, data):
    views_allowed = views_allowed_dataset(config_params)
    breast_side=[]
    image_read_list=[]
    views_saved=[]
    data1 = {}
    studyuid_path = str(data['FullPath'])
    series_list = os.listdir(studyuid_path)
    series_list = view_extraction(series_list, views_allowed)
    #series_list.sort()
    if series_list[0][1].split('.')[-1] == 'png':
        for series in series_list:
            img_path = studyuid_path + '/' + series[1]
            data1['FullPath'] = img_path
            data1['Views'] = series[0]
            img = collect_images(config_params, data1)
            if series[0] in views_allowed and series[0] not in views_saved:
                views_saved.append(series[0])
                image_read_list.append(img)
                breast_side.append(series[0][0])
    else:
        for series in series_list:
            series_path = studyuid_path+'/'+series[1]
            img_list = os.listdir(series_path)
            img_list = selecting_data(config_params, img_list)
            for image in img_list:
                img_path = series_path+'/'+image
                data1['FullPath'] = img_path
                data1['Views'] = series[0]
                img = collect_images(config_params, data1)
                if series[0] in views_allowed and series[0] not in views_saved:
                    if config_params['dataset']  == 'zgt' and config_params['viewsinclusion']=='all' and config_params['bitdepth']==12: #solve this in future
                        if series[0] in data['Views'].split('+'):
                            views_saved.append(series[0])
                            image_read_list.append(img)
                            breast_side.append(series[0][0])
                    else:
                        views_saved.append(series[0])
                        image_read_list.append(img)
                        breast_side.append(series[0][0])
        
        #if "+".join(sorted(data['Views'].split('+')))!="+".join(views_saved):
        #    print("Not matched:", data['FullPath'], flush=True)
        #    print(data['Views'], flush=True)
        #    print(views_saved, flush=True)
    return image_read_list, breast_side, views_saved

def collect_images(config_params, data):
    views_allowed = views_allowed_dataset(config_params)
    if config_params['bitdepth'] ==  8:
        img = collect_images_8bits(config_params, data, views_allowed)
    elif config_params['bitdepth'] == 16:
        if config_params['imagecleaning'] == 'gmic':
            img = collect_images_gmiccleaningcode(data)
        else:
            img = collect_images_16bits(config_params, data, views_allowed)
    elif config_params['bitdepth'] == 12:
        img = collect_images_12bits(config_params, data, views_allowed)
    return img
         
def collect_images_8bits(config_params, data, views_allowed):
    #collect images for the model
    if data['Views'] in views_allowed:
        img_path = str(data['FullPath'])
        img = Image.open(img_path)
        breast_side = data['Views'][0]
        if config_params['flipimage']:
            hflip_img = MyHorizontalFlip()
            img=hflip_img(img,breast_side)
        return img
    else:
        print('error in view')
        sys.exit()

def collect_images_16bits(config_params, data, views_allowed):
    #collect images for the model
    if data['Views'] in views_allowed:
        img_path = str(data['FullPath'])
        img = cv2.imread(img_path,-1)
        img_dtype = img.dtype
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB).astype(np.float32)
        breast_side = data['Views'][0]
        if img_dtype=='uint16':
            img/=65535
        img = torch.from_numpy(img.transpose((2, 0, 1))).contiguous()
        if config_params['flipimage']:
            hflip_img = MyHorizontalFlip()
            img = hflip_img(img,breast_side)
        return img
    else:
        print('error in view')
        sys.exit()

def collect_images_12bits(config_params, data, views_allowed):
    #collect images for the model
    #print(data['Views'])
    #print(views_allowed)
    if data['Views'] in views_allowed:
        img_path = str(data['FullPath'])
        img = np.load(img_path).astype(np.float32)
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        breast_side = data['Views'][0]
        img/=4095
        img = torch.from_numpy(img.transpose((2, 0, 1))).contiguous()
        if config_params['flipimage']:
            hflip_img = MyHorizontalFlip()
            img = hflip_img(img,breast_side)
        return img
    else:
        print(data['FullPath'])
        print('error in view')
        sys.exit()

def collect_images_gmiccleaningcode(data):
    breast_side = data['Views'][0]
    loaded_image = load_image(image_path=data['FullPath'], view=data['Views'], horizontal_flip='NO', breast_side=breast_side)
    gmic_pkl_filepath = '/projects/dso_mammovit/project_kushal/data/gmic-cleaningcode-pkl/data_restructured.pkl'
    f = open(gmic_pkl_filepath,'rb')
    pkl = pickle.load(f)
    imagename=data['ImageName'].replace('.','-')
    img = gmic_image_cleaning(loaded_image, data['Views'], pkl[imagename]['best_center'][0])
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    img/=65535
    img = torch.from_numpy(img.transpose((2, 0, 1))).contiguous()
    #print(data['FullPath'])
    #plt.imsave('./'+data['FullPath'].split('/')[-1], img, cmap="Greys_r")
    return img

def collect_images_gmic2(data):
    #another process to get the same result as collect_images_gmiccleaningcode; difference is in this function I use gmic's functions to load the image.
    with open("/projects/dso_mammovit/project_kushal/data/gmic-cleaningcode-pkl/data.pkl", "rb") as f:
        exam_list = pickle.load(f)
    for exam in exam_list:
        for view in views_allowed_gmic:
            if view in exam.keys() and exam[view][0]==data['FullPath'].split('/')[-1].strip('.png'):
                datum = exam
                view1 = view
                short_file_path = datum[view][0]
                print(datum)
    
    loaded_image = loading.load_image(image_path=data['FullPath'], view=view1, horizontal_flip=datum["horizontal_flip"])
    loaded_image = loading.process_image(loaded_image, view, datum["best_center"][view][0])
    print(loaded_image)
    print(loaded_image.shape)
    plt.imsave('./'+data['FullPath'].split('/')[-1], loaded_image, cmap="Greys_r")
    #input('halt')
    return loaded_image
##-------------------------------------------------------------------- collect images end ----------------------------------------------------------------------#

##------------------------------------------------------------------ different types of data augmentation functions --------------------------------------------# 
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def data_augmentation_train_kim(config_params, mean, std_dev):
    preprocess_train = transforms.Compose([
        transforms.ColorJitter(brightness=0.10, contrast=0.10),
        transforms.Resize((config_params['resize'][0], config_params['resize'][1])),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std_dev)
    ])
    return preprocess_train

def data_augmentation_train_shu(config_params, mean, std_dev):
    preprocess_train_list=[]
    if config_params['resize']:
        preprocess_train_list.append(transforms.Resize((config_params['resize'][0],config_params['resize'][1])))

    preprocess_train_list.append(transforms.ToTensor())

    preprocess_train_list=preprocess_train_list+[
        #MyCrop(),
        #transforms.Pad(100),
        transforms.RandomHorizontalFlip(p=0.5),
        #transforms.ColorJitter(brightness=0.20, contrast=0.20),
        transforms.ColorJitter(contrast=0.20, saturation=0.20),
        transforms.RandomRotation(30),
        #transforms.RandomAdjustSharpness(sharpness_factor=0.20),
        #MyGammaCorrection(0.20),
        #MyPaddingLongerSide(resize)
        AddGaussianNoise(mean=0, std=0.005)
        ]
    
    if config_params['datascaling']:
        if config_params['datascaling']!='standardizeperimage':
            preprocess_train_list.append(transforms.Normalize(mean=mean, std=std_dev))
    
    preprocess_train = transforms.Compose(preprocess_train_list)
    return preprocess_train

def data_augmentation_train_shen_gmic(config_params, mean, std_dev):
    preprocess_train_list=[]

    if config_params['imagecleaning']!='gmic':
        if config_params['resize']:
            preprocess_train_list.append(transforms.Resize((config_params['resize'][0], config_params['resize'][1])))
    
    preprocess_train_list=preprocess_train_list+[
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=(15),translate=(0.1,0.1),scale=(0.8,1.6),shear=(25)),
        AddGaussianNoise(mean=0, std=0.005)
        ]
    
    if config_params['datascaling']:
        if config_params['datascaling']!='standardizeperimage':
            preprocess_train_list.append(transforms.Normalize(mean=mean, std=std_dev))
    
    preprocess_train = transforms.Compose(preprocess_train_list)
    return preprocess_train

def data_augmentation_test_shen_gmic(config_params, mean, std_dev):
    preprocess_test_list=[]

    if config_params['imagecleaning']!='gmic':
        if config_params['resize']:
            preprocess_test_list.append(transforms.Resize((config_params['resize'][0],config_params['resize'][1])))
    
    if config_params['datascaling']:  
        if config_params['datascaling']!='standardizeperimage':
            preprocess_test_list.append(transforms.Normalize(mean=mean, std=std_dev))
    
    preprocess_test = transforms.Compose(preprocess_test_list)
    return preprocess_test

def data_augmentation(config_params):
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            utils.MyCrop(),
            transforms.Pad(100),
            transforms.RandomRotation(3),
            transforms.ColorJitter(brightness=0.20, contrast=0.20),
            transforms.RandomAdjustSharpness(sharpness_factor=0.20),
            utils.MyGammaCorrection(0.20),
            utils.MyPaddingLongerSide(config_params['resize']),
            transforms.Resize((config_params['resize'][0],config_params['resize'][1])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        #'train': transforms.Compose([
        #    transforms.Resize((resize,resize)),
        #    transforms.ToTensor(),
        #    transforms.RandomHorizontalFlip(p=0.5),
        #    transforms.ColorJitter(contrast=0.20, saturation=0.20),
        #    transforms.RandomRotation(30),
        #    AddGaussianNoise(mean=0, std=0.005),
        #    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #]),
        'val': transforms.Compose([
            transforms.Resize((config_params['resize'][0],config_params['resize'][1])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    return data_transforms

def data_augmentation_train(config_params, mean, std_dev):
    preprocess_train_list=[]

    preprocess_train_list=preprocess_train_list+[
        MyCrop(),
        #transforms.Pad(100),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(3),
        transforms.ColorJitter(brightness=0.20, contrast=0.20),
        transforms.RandomAdjustSharpness(sharpness_factor=0.20),
        MyGammaCorrection(0.20),
        MyPaddingLongerSide(config_params['resize'])]
    
    if config_params['resize']:
        preprocess_train_list.append(transforms.Resize((config_params['resize'][0],config_params['resize'][1])))
    
    preprocess_train_list.append(transforms.ToTensor())
    
    if config_params['datascaling']:
        if config_params['datascaling']!='standardizeperimage':
            preprocess_train_list.append(transforms.Normalize(mean=mean, std=std_dev))
    
    preprocess_train = transforms.Compose(preprocess_train_list)
    return preprocess_train

def data_augmentation_test(config_params, mean, std_dev):
    preprocess_test_list=[]

    if config_params['resize']:
        preprocess_test_list.append(transforms.Resize((config_params['resize'][0],config_params['resize'][1])))
    
    preprocess_test_list.append(transforms.ToTensor())
    
    if config_params['datascaling']:  
        if config_params['datascaling']!='standardizeperimage':
            preprocess_test_list.append(transforms.Normalize(mean=mean, std=std_dev))
    
    preprocess_test = transforms.Compose(preprocess_test_list)
    return preprocess_test

##------------------------------------------------------------------ different types of data augmentation functions end --------------------------------------------#

def fetch_groundtruth(df,acc_num,modality):
    col_names=df.filter(regex='Acc_'+modality+'.*').columns.tolist()
    acc_num=int(acc_num)
    groundtruth=df.loc[(df[col_names].astype('Int64')==acc_num).any(axis=1)]['final_gt']
    if groundtruth.empty:
        groundtruth=-1
    else:
        groundtruth=groundtruth.item()
    return groundtruth

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

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def removing_weight_decay_momentum(optimizer,layer_keyword,optimizer_params_dic):
    index=optimizer_params_dic[layer_keyword]
    optimizer.param_groups[index]['momentum']=0
    optimizer.param_groups[index]['weight_decay']=0
    return optimizer

def unfreeze_layers(model, layer_keyword):
    for name,param in model.named_parameters():
        if layer_keyword in name:
            param.requires_grad = True
    return model

def freeze_layers(model, layer_keyword):
    for name,param in model.named_parameters():
        if layer_keyword in name:
            param.requires_grad = False
    return model

def unfreeze_wt_update(optimizer, layer_keyword, optimizer_params_dic, previous_lr):
    index = optimizer_params_dic[layer_keyword]
    optimizer.param_groups[index]['lr'] = previous_lr
    return optimizer

def freeze_wt_update(optimizer, layer_keyword, optimizer_params_dic):
    index = optimizer_params_dic[layer_keyword]
    old_lr = optimizer.param_groups[index]['lr']
    optimizer.param_groups[index]['lr'] = 0
    return optimizer, old_lr

def return_optimstate(optimizer, layer_keyword, optimizer_params_dic):
    optim_state = {}
    count_params_end = 0
    index = optimizer_params_dic[layer_keyword]
    count_param_group = len(optimizer.param_groups[index]['params'])
    for id in range(0, index+1):
        count_params_end+= len(optimizer.param_groups[id]['params'])
    
    for id in range(count_params_end-count_param_group, count_params_end):
        if id in optimizer.state_dict()['state'].keys():
            optim_state[id] = deepcopy(optimizer.state_dict()['state'][id])
    return optim_state

def assign_previous_optimstate(optimizer, previous_state):
    if previous_state:
        state_dict = optimizer.state_dict()
        state_dict['state'].update(previous_state)
        optimizer.load_state_dict(state_dict)
    return optimizer

def dynamic_training(config_params, views_names, model, optimizer, previous_state, previous_lr, before_optimupdate):
    if config_params['attention'] == 'breastwise' and (config_params['milpooling'] == 'esatt' or config_params['milpooling'] == 'esgatt' or config_params['milpooling'] == 'isatt' or config_params['milpooling'] == 'isgatt'):
        optimizer_params_dic = {'img.attention':0, 'side.attention':1}
    elif config_params['attention'] == 'imagewise' and (config_params['milpooling'] == 'esatt' or config_params['milpooling'] == 'esgatt' or config_params['milpooling'] == 'isatt' or config_params['milpooling'] == 'isgatt'):
        optimizer_params_dic = {'img.attention':0}
    
    if config_params['milpooling'] == 'esatt' or config_params['milpooling'] == 'esgatt' or config_params['milpooling'] == 'isatt' or config_params['milpooling'] == 'isgatt':
        view_split = np.array([view[1:] for view in views_names])
        view_split = np.unique(view_split).tolist()
        breast_split = np.array([view[0] for view in views_names])
        breast_split = breast_split.tolist()

        #attention weighing switch off
        if config_params['attention'] == 'breastwise':
            #print("I am switching off perbreast.attention")
            if breast_split.count('L')<2 and breast_split.count('R')<2:
                if before_optimupdate:
                    model = freeze_layers(model, 'img.attention')
                    optimizer, previous_lr = freeze_wt_update(optimizer, 'img.attention',  optimizer_params_dic)
                    previous_state = return_optimstate(optimizer, 'img.attention',  optimizer_params_dic)
                else:
                    optimizer = assign_previous_optimstate(optimizer, previous_state)
                    model = unfreeze_layers(model, 'img.attention')
                    optimizer = unfreeze_wt_update(optimizer, 'img.attention',  optimizer_params_dic, previous_lr)
            
            #print("I am switching off both.attention")       
            if (breast_split.count('L')==0) or (breast_split.count('R')==0):
                if before_optimupdate:
                    model = freeze_layers(model, 'side.attention')
                    optimizer, previous_lr = freeze_wt_update(optimizer, 'side.attention',  optimizer_params_dic)
                    previous_state = return_optimstate(optimizer, 'side.attention',  optimizer_params_dic)
                else:
                    optimizer = assign_previous_optimstate(optimizer, previous_state)
                    model = unfreeze_layers(model, 'side.attention')
                    optimizer = unfreeze_wt_update(optimizer, 'side.attention',  optimizer_params_dic, previous_lr)
        
        elif config_params['attention'] == 'imagewise':
            if len(views_names)==1:
                if before_optimupdate:
                    model = freeze_layers(model, 'img.attention')
                    optimizer, previous_lr = freeze_wt_update(optimizer, 'img.attention',  optimizer_params_dic)
                    previous_state = return_optimstate(optimizer, 'img.attention',  optimizer_params_dic)
                else:
                    optimizer = assign_previous_optimstate(optimizer, previous_state)
                    model = unfreeze_layers(model, 'img.attention')
                    optimizer = unfreeze_wt_update(optimizer, 'img.attention',  optimizer_params_dic, previous_lr)     
        
    if before_optimupdate:
        return model, optimizer, previous_state, previous_lr
    else:
        return model, optimizer


def freeze_pipelines_kim(model,optimizer,views_names, attention, feature_extractor, activation):
    if activation=='softmax':
        if attention=='breastwise' and feature_extractor=='viewwise':
            optimizer_params_dic={'.mlo':0,'.cc':1,'both_b.attention':2,'both_m.attention':2,'perbreast_b.attention':3,'perbreast_m.attention':3}
        elif attention=='breastwise' and feature_extractor=='common':
            optimizer_params_dic={'both_b.attention':0,'both_m.attention':0,'perbreast_b.attention':1,'perbreast_m.attention':1}
        elif attention=='imagewise' and feature_extractor=='viewwise':
            optimizer_params_dic={'.mlo':0,'.cc':1,'model_attention_b.attention':2,'model_attention_m.attention':2}
        elif attention=='imagewise' and feature_extractor=='common':
            optimizer_params_dic={'model_attention_b.attention':0,'model_attention_m.attention':0}
        elif feature_extractor=='viewwise':
            optimizer_params_dic={'.mlo':0,'.cc':1}
    
    elif activation=='sigmoid':
        if attention=='breastwise' and feature_extractor=='viewwise':
            optimizer_params_dic={'.mlo':0,'.cc':1,'both.attention':2,'perbreast.attention':3}
        elif attention=='breastwise' and feature_extractor=='common':
            optimizer_params_dic={'both.attention':0,'perbreast.attention':1}
        elif attention=='imagewise' and feature_extractor=='viewwise':
            optimizer_params_dic={'.mlo':0,'.cc':1,'model_attention.attention':2}
        elif attention=='imagewise' and feature_extractor=='common':
            optimizer_params_dic={'model_attention.attention':0}
        elif feature_extractor=='viewwise':
            optimizer_params_dic={'.mlo':0,'.cc':1}
    
    #print(views_names)
    view_split=np.array([view[1:] for view in views_names])
    view_split=np.unique(view_split).tolist()
    breast_split=np.array([view[0] for view in views_names])
    breast_split=breast_split.tolist()
    #print(view_split,breast_split)
    
    #model switching on all weights
    for name,param in model.named_parameters():
        param.requires_grad=True
    
    for param_groups in optimizer.param_groups:
        param_groups['momentum']=0.9 
        param_groups['weight_decay']=0.0005
    
    if feature_extractor=='viewwise':
        #model switch off
        if view_split==['CC']:
            #print("I am switching off .mlo")
            model=freeze_layers(model, '.mlo')
            optimizer=removing_weight_decay_momentum(optimizer, '.mlo', optimizer_params_dic)
        elif view_split==['MLO']:
            #print("I am switching off .cc")
            model=freeze_layers(model,'.cc')
            optimizer=removing_weight_decay_momentum(optimizer, '.cc', optimizer_params_dic)
    
    #attention weighing switch off
    if attention=='breastwise':
        if activation=='softmax':
            if breast_split.count('L')<2 and breast_split.count('R')<2:
                #print("I am switching off _left.attention")
                model=freeze_layers(model,'perbreast_b.attention')
                model=freeze_layers(model,'perbreast_m.attention')
                optimizer=removing_weight_decay_momentum(optimizer, 'perbreast_b.attention', optimizer_params_dic)
                optimizer=removing_weight_decay_momentum(optimizer, 'perbreast_m.attention', optimizer_params_dic)
            if (breast_split.count('L')==0) or (breast_split.count('R')==0):
                #print("I am switching off both.attention")
                model=freeze_layers(model,'both_b.attention')
                model=freeze_layers(model,'both_m.attention')
                optimizer=removing_weight_decay_momentum(optimizer, 'both_b.attention', optimizer_params_dic)
                optimizer=removing_weight_decay_momentum(optimizer, 'both_m.attention', optimizer_params_dic)
        
        elif activation=='sigmoid':
            if breast_split.count('L')<2 and breast_split.count('R')<2:
                #print("I am switching off _left.attention")
                model=freeze_layers(model,'perbreast.attention')
                optimizer=removing_weight_decay_momentum(optimizer, 'perbreast.attention', optimizer_params_dic)
            if (breast_split.count('L')==0) or (breast_split.count('R')==0):
                #print("I am switching off both.attention")
                model=freeze_layers(model,'both.attention')
                optimizer=removing_weight_decay_momentum(optimizer, 'both.attention', optimizer_params_dic)
    
    elif attention=='imagewise':
        if activation=='softmax':
            if len(views_names)==1:
                model=freeze_layers(model,'model_attention_b.attention')
                model=freeze_layers(model,'model_attention_m.attention')
                optimizer=removing_weight_decay_momentum(optimizer, 'model_attention_b.attention', optimizer_params_dic)
                optimizer=removing_weight_decay_momentum(optimizer, 'model_attention_m.attention', optimizer_params_dic)
        elif activation=='sigmoid':
            if len(views_names)==1:
                model=freeze_layers(model,'model_attention.attention')
                optimizer=removing_weight_decay_momentum(optimizer, 'model_attention.attention', optimizer_params_dic)
    
    return model, optimizer

def views_distribution(config_params, df):
    views_allowed = views_allowed_dataset(config_params)
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

def plot(filename):
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

def class_distribution_weightedloss(df):
    df_groundtruth=df['Groundtruth'].map(groundtruth_dic)
    class_weight=utils.class_weight.compute_class_weight(class_weight = 'balanced', classes = np.array([0,1]), y = df_groundtruth)
    print(dict(Counter(df_groundtruth)))
    print(class_weight)
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

def stratifiedgroupsplit(df, rand_seed):
    groups = df.groupby('Groundtruth')
    all_train = []
    all_test = []
    all_val = []
    train_testsplit = GroupShuffleSplit(test_size=0.15, n_splits=2, random_state=rand_seed)
    train_valsplit = GroupShuffleSplit(test_size=0.10, n_splits=2, random_state=rand_seed)
    for group_id, group in groups:
        # if a group is already taken in test or train it must stay there
        group = group[~group['Patient_Id'].isin(all_train+all_val+all_test)]
        # if group is empty 
        if group.shape[0] == 0:
            continue
        train_inds1, test_inds = next(train_testsplit.split(group, groups=group['Patient_Id']))
        train_inds, val_inds = next(train_valsplit.split(group.iloc[train_inds1], groups=group.iloc[train_inds1]['Patient_Id']))
    
        all_train += group.iloc[train_inds1].iloc[train_inds]['Patient_Id'].tolist()
        all_val += group.iloc[train_inds1].iloc[val_inds]['Patient_Id'].tolist()
        all_test += group.iloc[test_inds]['Patient_Id'].tolist()
        
    train = df[df['Patient_Id'].isin(all_train)]
    val = df[df['Patient_Id'].isin(all_val)]
    test = df[df['Patient_Id'].isin(all_test)]
    
    '''form_train = set(train['Patient_Id'].tolist())
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
    print(test[test['Patient_Id'].isin(train['Patient_Id'].unique().tolist())])'''
    return train, val, test

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
        img, _ =collect_images(data)
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
        image_list, _, _ = collect_images(studyuid_path,series_list)
        for j,img in enumerate(image_list):
            if transform:
                img=preprocess(img)
            means.append(torch.mean(img))
            stds.append(torch.std(img))
    
    mean_dataset = torch.mean(torch.tensor(means))
    std_dataset = torch.mean(torch.tensor(stds))
    return mean_dataset, std_dataset

def performance_metrics(conf_mat,y_true,y_pred,y_prob):
    prec=metrics.precision_score(y_true,y_pred,pos_label=1)
    rec=metrics.recall_score(y_true,y_pred) #sensitivity, TPR
    if conf_mat is None:
        spec=np.sum((y_true==0) & (y_pred==0))/np.sum(y_true==0)
        print(y_true)
        print(y_pred)
        print(np.sum((y_true==0) & (y_pred==0)))
        print(np.sum(y_true==0))
        print(spec)
        
    else:
        spec=conf_mat[0,0]/np.sum(conf_mat[0,:]) #TNR
    
    f1=metrics.f1_score(y_true,y_pred)
    f1macro=metrics.f1_score(y_true,y_pred,average='macro')
    f1wtmacro=metrics.f1_score(y_true,y_pred,average='weighted')
    acc=metrics.accuracy_score(y_true,y_pred)
    bal_acc=(rec+spec)/2
    cohen_kappa=metrics.cohen_kappa_score(y_true,y_pred)
    if y_prob is None:
        auc=0.0
    else: 
        auc=metrics.roc_auc_score(y_true,y_prob)
    each_model_metrics=[prec,rec,spec,f1,f1macro,f1wtmacro,acc,bal_acc,cohen_kappa,auc]
    return each_model_metrics

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

def results_subgroups(df,true_labels,pred_labels,y_prob, sheet):
    breastden_A=df[df['BreastDensity']==1].index
    breastden_B=df[df['BreastDensity']==2].index
    breastden_C=df[df['BreastDensity']==3].index
    breastden_D=df[df['BreastDensity']==4].index
    
    breastden_A_metrics = performance_metrics(None,true_labels[breastden_A],pred_labels[breastden_A],y_prob[breastden_A])
    breastden_B_metrics = performance_metrics(None,true_labels[breastden_B],pred_labels[breastden_B],y_prob[breastden_B])
    breastden_C_metrics = performance_metrics(None,true_labels[breastden_C],pred_labels[breastden_C],y_prob[breastden_C])
    breastden_D_metrics = performance_metrics(None,true_labels[breastden_D],pred_labels[breastden_D],y_prob[breastden_D])
    
    sheet.append(['breast density A'])
    sheet.append(['Precision','Recall','Specificity','F1','Acc','Bal_Acc','Cohens Kappa','AUC'])
    sheet.append(breastden_A_metrics)
    sheet.append(['breast density B'])
    sheet.append(['Precision','Recall','Specificity','F1','Acc','Bal_Acc','Cohens Kappa','AUC'])
    sheet.append(breastden_B_metrics)
    sheet.append(['breast density C'])
    sheet.append(['Precision','Recall','Specificity','F1','Acc','Bal_Acc','Cohens Kappa','AUC'])
    sheet.append(breastden_C_metrics)
    sheet.append(['breast density D'])
    sheet.append(['Precision','Recall','Specificity','F1','Acc','Bal_Acc','Cohens Kappa','AUC'])
    sheet.append(breastden_D_metrics)
    
    birads_0=df[df['AssessmentMax']==0].index
    birads_1=df[df['AssessmentMax']==1].index
    birads_2=df[df['AssessmentMax']==2].index
    birads_3=df[df['AssessmentMax']==3].index
    birads_4=df[df['AssessmentMax']==4].index
    birads_5=df[df['AssessmentMax']==5].index
    birads_6=df[df['AssessmentMax']==6].index
    
    birads_0_metrics = performance_metrics(None,true_labels[birads_0],pred_labels[birads_0],y_prob[birads_0])
    if not birads_1.empty:
        try:
            print('I am in try!')
            birads_1_metrics = performance_metrics(None,true_labels[birads_1],pred_labels[birads_1],y_prob[birads_1])
        except:
            print('I am in except')
            pass
    birads_2_metrics = performance_metrics(None,true_labels[birads_2],pred_labels[birads_2],y_prob[birads_2])
    birads_3_metrics = performance_metrics(None,true_labels[birads_3],pred_labels[birads_3],y_prob[birads_3])
    birads_4_metrics = performance_metrics(None,true_labels[birads_4],pred_labels[birads_4],y_prob[birads_4])
    birads_5_metrics = performance_metrics(None,true_labels[birads_5],pred_labels[birads_5],y_prob[birads_5])
    if not birads_6.empty:
        try:
            print('I am in try!')
            birads_6_metrics = performance_metrics(None,true_labels[birads_6],pred_labels[birads_6],y_prob[birads_6])
        except:
            print('I am in except')
            pass
    
    sheet.append(['birads 0'])
    sheet.append(['Precision','Recall','Specificity','F1','Acc','Bal_Acc','Cohens Kappa','AUC'])
    sheet.append(birads_0_metrics)
    if not birads_1.empty:
        try:
            sheet.append(['birads 1'])
            sheet.append(['Precision','Recall','Specificity','F1','Acc','Bal_Acc','Cohens Kappa','AUC'])
            sheet.append(birads_1_metrics)
        except:
            pass
    
    sheet.append(['birads 2'])
    sheet.append(['Precision','Recall','Specificity','F1','Acc','Bal_Acc','Cohens Kappa','AUC'])
    sheet.append(birads_2_metrics)
    sheet.append(['birads 3'])
    sheet.append(['Precision','Recall','Specificity','F1','Acc','Bal_Acc','Cohens Kappa','AUC'])
    sheet.append(birads_3_metrics)
    sheet.append(['birads 4'])
    sheet.append(['Precision','Recall','Specificity','F1','Acc','Bal_Acc','Cohens Kappa','AUC'])
    sheet.append(birads_4_metrics)
    sheet.append(['birads 5'])
    sheet.append(['Precision','Recall','Specificity','F1','Acc','Bal_Acc','Cohens Kappa','AUC'])
    sheet.append(birads_5_metrics)
    if not birads_6.empty:
        try:
            sheet.append(['birads 6'])
            sheet.append(['Precision','Recall','Specificity','F1','Acc','Bal_Acc','Cohens Kappa','AUC'])
            sheet.append(birads_6_metrics)
        except:
            pass
    
    return sheet

def confusion_matrix_norm_func(conf_mat,fig_name,class_name):
    #class_name=['W','N1','N2','N3','REM']
    conf_mat_norm=np.empty((conf_mat.shape[0],conf_mat.shape[1]))
    #conf_mat=confusion_matrix(y_true, y_pred)
    for i in range(conf_mat.shape[0]):
        conf_mat_norm[i,:]=conf_mat[i,:]/sum(conf_mat[i,:])
    #print(conf_mat_norm)
    print_confusion_matrix(conf_mat_norm,class_name,fig_name)
    
def print_confusion_matrix(conf_mat_norm, class_names, fig_name, figsize = (2,2), fontsize=5):
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
        yticklabels=class_names,
        xticklabels=class_names,
        data=conf_mat_norm,
        ax=ax,
        cmap='YlGnBu',
        cbar=False,
        #cbar_ax=cbar_ax,
        annot=True,
        annot_kws={'size':fontsize},
        fmt=".2f",
        square=True
        #linewidths=0.75
        )
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    ax.set_ylabel('True label',labelpad=0,fontsize=fontsize)
    ax.set_xlabel('Predicted label',labelpad=0,fontsize=fontsize)
    #cbar_ax.tick_params(labelsize=fontsize) 
    #ax.get_yaxis().set_visible(False)
    #plt.tight_layout()
    #plt.show()
    ax.set_title(fig_name)
    fig.savefig(fig_name+'.pdf', format='pdf', bbox_inches='tight')    

def data_augmentation(img_path):
    img = Image.open(img_path)
    width, height = img.size
    #img.save('original_image.png')
    #print(angle)
    #saturation = random.uniform(0,2)
    #img1 = TF.adjust_saturation(img,0) #works only on coloured images, so we will ignore this
    #img1 = TF.adjust_hue(img,0.5) # works only on coloured image
    pad = random.randint(0,100)
    size_left = random.randint(0,100)
    size_right = random.randint(width-100,width)
    size_top = random.randint(0,100)
    size_bottom = random.randint(height-100,height)
    angle = random.randint(-3, 3)
    sharpness = random.uniform(0.8,1.2)
    gamma = random.uniform(0.8,1.2)
    brightness = random.uniform(0.8,1.2)
    contrast = random.uniform(0.8,1.2)
    #print(pad,size_left,size_right,size_top,size_bottom)
    img1 = TF.crop(img,size_top,size_left,size_bottom,size_right)
    img1 = TF.pad(img1,pad)
    
    #img1 = TF.rotate(img1,-3)
    img1 = TF.rotate(img1,3)
    img1 = TF.adjust_brightness(img1,1.2) #0 gives a black image, 1 gives the original image while 2 increases the brightness by a factor of 2
    img1.save('crop_pad_rotate_bright1.2.png')
    '''img1 = TF.adjust_contrast(img1,contrast) #0 gives a solid gray image, 1 gives the original image while 2 increases the contrast by a factor of 2
    img1 = TF.adjust_sharpness(img1,sharpness) #0 gives a blurred image, 1 gives the original image while 2 increases the sharpness by a factor of 2.
    img1 = TF.adjust_gamma(img1,gamma) #gamma larger than 1 make the shadows darker, while gamma smaller than 1 make dark regions lighter
    '''

def crosscheck_view_collect_images(config_params, df):
    views_allowed = views_allowed_dataset(config_params)
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



##------------------------------------------------------------- functions taken from gmic --------------------------------#
def standard_normalize_single_image(image):
    """
    Standardizes an image in-place 
    """
    image -= np.mean(image)
    image /= np.maximum(np.std(image), 10**(-5))

def read_image_png(file_name):
    image = np.array(imageio.imread(file_name))
    return image


def load_image(image_path, view, horizontal_flip, breast_side):
    """
    Loads a png or hdf5 image as floats and flips according to its view.
    """
    if image_path.endswith("png"):
        image = read_image_png(image_path)
    else:
        raise RuntimeError()
    image = image.astype(np.float32)
    image = myhorizontalflip(image, breast_side)
    return image

def gmic_image_cleaning(image, view, best_center):
    """
    Applies augmentation window with random noise in location and size
    and return normalized cropped image.
    """
    cropped_image, _ = augmentations.random_augmentation_best_center(
        image=image,
        input_size=(2944, 1920),
        random_number_generator=np.random.RandomState(0),
        best_center=best_center,
        view=view
    )
    cropped_image = cropped_image.copy()
    #standard_normalize_single_image(cropped_image)
    return cropped_image

def make_sure_in_range(val, min_val, max_val):
    """
    Function that make sure that min < val < max; otherwise return the limit value
    """
    if val < min_val:
        return min_val
    if val > max_val:
        return max_val
    return val

def crop_pytorch(original_img_pytorch, crop_shape, crop_position, out,
                 method="center", background_val="min"):
    """
    Function that take a crop on the original image.
    Use PyTorch to do this.
    :param original_img_pytorch: (N,C,H,W) PyTorch Tensor
    :param crop_shape: (h, w) integer tuple
    :param method: supported in ["center", "upper_left"]
    :return: (N, K, h, w) PyTorch Tensor
    """
    # retrieve inputs
    H, W = original_img_pytorch.shape
    crop_x, crop_y = crop_position
    x_delta, y_delta = crop_shape

    # locate the four corners
    if method == "center":
        min_x = int(np.round(crop_x - x_delta / 2))
        max_x = int(np.round(crop_x + x_delta / 2))
        min_y = int(np.round(crop_y - y_delta / 2))
        max_y = int(np.round(crop_y + y_delta / 2))
    elif method == "upper_left":
        min_x = int(np.round(crop_x))
        max_x = int(np.round(crop_x + x_delta))
        min_y = int(np.round(crop_y))
        max_y = int(np.round(crop_y + y_delta))

    # make sure that the crops are in range
    min_x = make_sure_in_range(min_x, 0, H)
    max_x = make_sure_in_range(max_x, 0, H)
    min_y = make_sure_in_range(min_y, 0, W)
    max_y = make_sure_in_range(max_y, 0, W)

    # somehow background is normalized to this number
    if background_val == "min":
        out[:, :] = original_img_pytorch.min()
    else:
        out[:, :] = background_val
    real_x_delta = max_x - min_x
    real_y_delta = max_y - min_y
    origin_x = crop_shape[0] - real_x_delta
    origin_y = crop_shape[1] - real_y_delta
    out[origin_x:, origin_y:] = original_img_pytorch[min_x:max_x, min_y:max_y]

def get_max_window(input_image, window_shape, pooling_logic="avg"):
    """
    Function that makes a sliding window of size window_shape over the
    input_image and return the UPPER_LEFT corner index with max sum
    :param input_image: N*C*H*W
    :param window_shape: h*w
    :return: N*C*2 tensor
    """
    N, C, H, W = input_image.size()
    if pooling_logic == "avg":
        # use average pooling to locate the window sums
        pool_map = torch.nn.functional.avg_pool2d(input_image, window_shape, stride=1)
    elif pooling_logic in ["std", "avg_entropy"]:
        # create sliding windows
        output_size = (H - window_shape[0] + 1, W - window_shape[1] + 1)
        sliding_windows = F.unfold(input_image, kernel_size=window_shape).view(N,C, window_shape[0]*window_shape[1], -1)
        # apply aggregation function on each sliding windows
        if pooling_logic == "std":
            agg_res = sliding_windows.std(dim=2, keepdim=False)
        elif pooling_logic == "avg_entropy":
            agg_res = -sliding_windows*torch.log(sliding_windows)-(1-sliding_windows)*torch.log(1-sliding_windows)
            agg_res = agg_res.mean(dim=2, keepdim=False)
        # merge back
        pool_map = F.fold(agg_res, kernel_size=(1, 1), output_size=output_size)
    _, _, _, W_map = pool_map.size()
    # transform to linear and get the index of the max val locations
    _, max_linear_idx = torch.max(pool_map.view(N, C, -1), -1)
    # convert back to 2d index
    #max_idx_x = max_linear_idx // W_map
    max_idx_x = torch.div(max_linear_idx, W_map, rounding_mode='trunc')
    max_idx_y = max_linear_idx - max_idx_x * W_map
    # put together the 2d index
    upper_left_points = torch.cat([max_idx_x.unsqueeze(-1), max_idx_y.unsqueeze(-1)], dim=-1)
    return upper_left_points

def generate_mask_uplft(input_image, window_shape, upper_left_points, gpu_number):
    """
    Function that generates mask that sets crops given upper_left
    corners to 0
    :param input_image:
    :param window_shape:
    :param upper_left_points:
    """
    N, C, H, W = input_image.size()
    window_h, window_w = window_shape
    # get the positions of masks
    mask_x_min = upper_left_points[:,:,0]
    mask_x_max = upper_left_points[:,:,0] + window_h
    mask_y_min = upper_left_points[:,:,1]
    mask_y_max = upper_left_points[:,:,1] + window_w
    # generate masks
    mask_x = Variable(torch.arange(0, H).view(-1, 1).repeat(N, C, 1, W))
    mask_y = Variable(torch.arange(0, W).view(1, -1).repeat(N, C, H, 1))
    if gpu_number is not None:
        device = torch.device("cuda:{}".format(gpu_number))
        mask_x = mask_x.cuda().to(device)
        mask_y = mask_y.cuda().to(device)
    x_gt_min = mask_x.float() >= mask_x_min.unsqueeze(-1).unsqueeze(-1).float()
    x_ls_max = mask_x.float() < mask_x_max.unsqueeze(-1).unsqueeze(-1).float()
    y_gt_min = mask_y.float() >= mask_y_min.unsqueeze(-1).unsqueeze(-1).float()
    y_ls_max = mask_y.float() < mask_y_max.unsqueeze(-1).unsqueeze(-1).float()

    # since logic operation is not supported for variable
    # I used * for logic ANd
    selected_x = x_gt_min * x_ls_max
    selected_y = y_gt_min * y_ls_max
    selected = selected_x * selected_y
    mask = 1 - selected.float()
    return mask

#---------------------------------------------- end of functions taken from gmic repository ----------------#