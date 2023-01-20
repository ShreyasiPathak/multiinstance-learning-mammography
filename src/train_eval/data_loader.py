import torch
import random
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader

from utilities import utils

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def input_normalization(config_params, df_train):
    if config_params['datascaling'] == 'scaling':
        if config_params['pretrained']:
            mean = [0.485, 0.456, 0.406]
            std_dev = [0.229, 0.224, 0.225]
            if config_params['channel']==1:
                mean_avg = sum(mean)/len(mean)
                std_dev_avg = sum(std_dev)/len(std_dev)
                mean = [mean_avg,mean_avg,mean_avg]
                std_dev = [std_dev_avg,std_dev_avg,std_dev_avg]
        else:
            mean = [0.5,0.5,0.5]
            std_dev = [0.5,0.5,0.5]
    
    elif config_params['datascaling'] == 'standardize':
        mean, std_dev = utils.calculate_dataset_mean_stddev(df_train, config_params['resize'], transform=True)
        #mean = [0.2016,0.2016,0.2016]
        #std_dev = [0.1953,0.1953,0.1953]
    print("mean, std dev of input normalization:",mean,std_dev)
    return mean, std_dev

def data_augmentation(config_params, df_train):
    mean, std_dev = input_normalization(config_params, df_train)
    
    if config_params['data_aug']=='kim':
        preprocess_train = transforms.Compose([
            transforms.ColorJitter(brightness=0.10, contrast=0.10),
            transforms.Resize((config_params['resize'][0],config_params['resize'][1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std_dev)
        ])
        preprocess_val=utils.data_augmentation_test(mean,std_dev,config_params)
        
    elif config_params['data_aug']=='gmic':
        preprocess_train = utils.data_augmentation_train_shen_gmic(mean,std_dev,config_params)
        preprocess_val = utils.data_augmentation_test_shen_gmic(mean,std_dev,config_params)
        
    elif config_params['data_aug']=='shu':
        preprocess_train=utils.data_augmentation_train_shu(mean,std_dev,config_params)
        preprocess_val=utils.data_augmentation_test(mean,std_dev,config_params)
        
    elif config_params['data_aug']=='own':
        preprocess_train=utils.data_augmentation_train(mean,std_dev,config_params)
        preprocess_val=utils.data_augmentation_test(mean,std_dev,config_params)
    
    return preprocess_train, preprocess_val

def dataloader(config_params, df_train, df_val, df_test, g):
    preprocess_train, preprocess_val = data_augmentation(config_params, df_train)
    
    batch_sampler=None
    batch_sampler_val=None
    batch_sampler_test=None
    shuffle=True
    sampler1=None
    if config_params['viewsinclusion'] == 'all':
        if config_params['class_imbalance'] == 'oversampling':
            sampler = utils.CustomGroupbyViewWeightedRandomSampler(df_train)
            sampler_val = utils.CustomGroupbyViewRandomSampler(df_val)
        else:
            sampler = utils.CustomGroupbyViewRandomSampler(df_train)
            sampler_val = utils.CustomGroupbyViewRandomSampler(df_val)
        
        sampler_test = utils.CustomGroupbyViewRandomSampler(df_test)
        
        view_group_length, view_group_name = sampler.__viewlength__()
        view_group_length_val, view_group_name_val = sampler_val.__viewlength__()
        view_group_length_test, view_group_name_test = sampler_test.__viewlength__()
        batch_sampler = utils.CustomGroupbyViewRandomBatchSampler(sampler, config_params['batch_size'], view_group_length, view_group_name)
        batch_sampler_val = utils.CustomGroupbyViewRandomBatchSampler(sampler_val, config_params['batch_size'], view_group_length_val, view_group_name_val)
        batch_sampler_test = utils.CustomGroupbyViewRandomBatchSampler(sampler_test, config_params['batch_size'], view_group_length_test, view_group_name_test)
        batch_size1 = 1
        shuffle = False
    
    elif config_params['viewsinclusion'] == 'standard': 
        if config_params['class_imbalance'] == 'oversampling':
            sampler1 = utils.CustomWeightedRandomSampler(df_train)
            shuffle = False
        batch_size1 = config_params['batch_size']
    
    dataset_gen_train = utils.BreastCancerDataset_generator(df_train, config_params, preprocess_train)
    dataloader_train = DataLoader(dataset_gen_train, batch_size=batch_size1, shuffle=shuffle, num_workers=config_params['numworkers'], collate_fn=utils.MyCollate, worker_init_fn=seed_worker, generator=g, sampler=sampler1, batch_sampler=batch_sampler)    
    
    if config_params['use_validation']:
        dataset_gen_val = utils.BreastCancerDataset_generator(df_val, config_params, preprocess_val)
        dataloader_val = DataLoader(dataset_gen_val, batch_size=batch_size1, shuffle=False, num_workers=config_params['numworkers'], collate_fn=utils.MyCollate, worker_init_fn=seed_worker, generator=g, batch_sampler=batch_sampler_val)
        #worker_init_fn=seed_worker, generator=g,
    
    dataset_gen_test = utils.BreastCancerDataset_generator(df_test, config_params, preprocess_val)
    dataloader_test = DataLoader(dataset_gen_test, batch_size=batch_size1, shuffle=False, num_workers=config_params['numworkers'], collate_fn=utils.MyCollate, worker_init_fn=seed_worker, generator=g, batch_sampler=batch_sampler_test)
    
    return dataloader_train, dataloader_val, dataloader_test