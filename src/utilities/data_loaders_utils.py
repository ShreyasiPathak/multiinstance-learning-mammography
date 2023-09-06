import re
import os
import cv2
import sys
import copy
import torch
import random
import pickle
import operator
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, Sampler

from utilities import utils, data_augmentation_utils, gmic_utils

views_allowed_gmic = ['L-CC','L-MLO','R-CC','R-MLO']

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
            return idx, img, torch.tensor(self.config_params['groundtruthdic'][data['Groundtruth']]), data['Views']
        
        elif self.config_params['learningtype'] == 'MIL' or self.config_params['learningtype'] == 'MV':
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
            #print(image_tensor.shape)
            return idx, image_tensor, torch.tensor(self.config_params['groundtruthdic'][data['Groundtruth']]), views_saved 

class CustomWeightedRandomSampler(Sampler):
    def __init__(self, config_params, df_train):
        self.df = df_train
        self.config_params = config_params
    
    def __iter__(self):
        len_instances=self.df.shape[0]
        OriginalIndices = np.array(range(0,len_instances))
        class_count = utils.stratified_class_count(self.df)
        class_count=class_count.to_dict()
        class_count[0] = class_count.pop('benign')
        class_count[1] = class_count.pop('malignant')
        max_class = max(class_count.items(), key=operator.itemgetter(1))
        diff_count_class=max_class[1]-np.array([class_count[0],class_count[1]])
        labels=self.df['Groundtruth'].replace(self.config_params['groundtruthdic'])
        #print(np.append(OriginalIndices,np.random.choice(np.where(labels==0)[0],5),axis=0))
        for i in range(0,2):
            if i!=max_class[0]:
                repeater=int(diff_count_class[i]/class_count[i])
                OriginalIndices=np.append(OriginalIndices,np.repeat(np.where(labels==i),repeater))
                leftover=diff_count_class[i]-(class_count[i]*repeater)
                OriginalIndices=np.append(OriginalIndices,np.random.choice(np.where(labels==i)[0],leftover),axis=0)
        random.shuffle(OriginalIndices)
        iter_shuffledIndex=iter(OriginalIndices)
        return iter_shuffledIndex
    
    def __len__(self):
        class_count = utils.stratified_class_count(self.df)
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
        while view_group_name_dup:    
            key=random.sample(view_group_name_dup,1)[0]
            if len(view_group_dic_dup[key])>self.batch_size:
                selected_index=random.sample(view_group_dic_dup[key],self.batch_size)
                insert_val=zip(selected_index,[key]*self.batch_size)
                total_indices.extend(insert_val)
                view_group_dic_dup[key] = [num for num in view_group_dic_dup[key] if num not in selected_index] 
            else:
                total_indices.extend(zip(view_group_dic_dup[key],[key]*len(view_group_dic_dup[key])))
                view_group_dic_dup.pop(key,None)
                view_group_name_dup.pop(view_group_name_dup.index(key))
                view_group_name_dup=list(view_group_dic_dup.keys())
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
    def __init__(self, config_params, df_train):
        self.df = df_train
        self.view_group = list(self.df.groupby(by=['Views']))
        self.config_params = config_params
    
    def __iter__(self):
        len_instances=self.df.shape[0]
        OriginalIndices=np.array(range(0,len_instances))
        class_count = utils.stratified_class_count(self.df)
        class_count=class_count.to_dict()
        class_count[0] = class_count.pop('benign')
        class_count[1] = class_count.pop('malignant')
        max_class = max(class_count.items(), key=operator.itemgetter(1))
        diff_count_class=max_class[1]-np.array([class_count[0],class_count[1]])
        labels=self.df['Groundtruth'].replace(self.config_params['groundtruthdic'])
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
        class_count = utils.stratified_class_count(self.df)
        class_count=class_count.to_dict()
        class_count[0] = class_count.pop('benign')
        class_count[1] = class_count.pop('malignant')
        max_class = max(class_count.items(), key=operator.itemgetter(1))
        diff_count_class=max_class[1]-np.array([class_count[0],class_count[1]])
        labels=self.df['Groundtruth'].replace(self.config_params['groundtruthdic'])
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

#--------------------------------------------------------------- collect image ----------------------------------------------------#
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
        '''
        #for CLAHE
        img = cv2.imread(img_path, 0)
        img_dtype = img.dtype
        if config_params['dataset']=='cbis-ddsm':
            clahe_create = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img = clahe_create.apply(img)
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB).astype(np.float32)
        breast_side = data['Views'][0]
        if img_dtype=='uint8':
            img/=255
        img = torch.from_numpy(img.transpose((2, 0, 1))).contiguous()
        '''
        img = Image.open(img_path)
        img= img.convert('RGB')
        breast_side = data['Views'][0]
        ### added later; have not checked if it works for shu or kim et al
        if config_params['dataaug'] == 'gmic':
            transformTen = transforms.ToTensor()
            img = transformTen(img)
        ### ends here ##
        if config_params['dataaug'] == 'kim':
            pad_longer_side = data_augmentation_utils.MyPaddingLongerSide(config_params['resize'])
            img = pad_longer_side(img, breast_side)
        if config_params['flipimage']:
            hflip_img = data_augmentation_utils.MyHorizontalFlip()
            img = hflip_img(img, breast_side)
        return img
    else:
        print('error in view')
        sys.exit()

def collect_images_16bits(config_params, data, views_allowed):
    #collect images for the model
    if data['Views'] in views_allowed:
        img_path = str(data['FullPath'])
        #print('img path:', img_path)
        img = cv2.imread(img_path,-1)
        img_dtype = img.dtype
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB).astype(np.float32)
        breast_side = data['Views'][0]
        if img_dtype=='uint16':
            img/=65535
        img = torch.from_numpy(img.transpose((2, 0, 1))).contiguous()
        if config_params['flipimage']:
            hflip_img = data_augmentation_utils.MyHorizontalFlip()
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
            hflip_img = data_augmentation_utils.MyHorizontalFlip()
            img = hflip_img(img,breast_side)
        return img
    else:
        print(data['FullPath'])
        print('error in view')
        sys.exit()

def collect_images_gmiccleaningcode(data):
    breast_side = data['Views'][0]
    loaded_image = gmic_utils.load_image(image_path=data['FullPath'], view=data['Views'], horizontal_flip='NO', breast_side=breast_side)
    gmic_pkl_filepath = '/projects/dso_mammovit/project_kushal/data/gmic-cleaningcode-pkl/data_restructured.pkl'
    f = open(gmic_pkl_filepath,'rb')
    pkl = pickle.load(f)
    imagename=data['ImageName'].replace('.','-')
    img = gmic_utils.gmic_image_cleaning(loaded_image, data['Views'], pkl[imagename]['best_center'][0])
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    img/=65535
    img = torch.from_numpy(img.transpose((2, 0, 1))).contiguous()
    #print(data['FullPath'])
    #plt.imsave('./'+data['FullPath'].split('/')[-1], img, cmap="Greys_r")
    return img
#---------------------------------------------------------end collect image ----------------------------------------------------#

#------------------------------------------------ collate functions ------------------------------------------------------------#
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
#---------------------------------------------------------- collate functions end -------------------------------------------------#