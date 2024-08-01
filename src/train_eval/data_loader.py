import torch
import random
import numpy as np
from torch.utils.data import DataLoader

from utilities import utils, data_augmentation_utils, data_loaders_utils

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def input_normalization(config_params, df_train):
    if config_params['datascaling'] == 'scaling':
        if config_params['pretrained']:
            mean = [0.485, 0.456, 0.406]
            std_dev = [0.229, 0.224, 0.225]
            if config_params['channel'] == 1:
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
    print("mean, std dev of input normalization:", mean, std_dev)
    return mean, std_dev

def data_augmentation(config_params, df_train):
    mean, std_dev = input_normalization(config_params, df_train)
    
    if config_params['dataaug']=='kim':
        preprocess_train = data_augmentation_utils.data_augmentation_train_kim(config_params, mean, std_dev)
        preprocess_val = data_augmentation_utils.data_augmentation_test(config_params, mean, std_dev)
        
    elif config_params['dataaug']=='gmic':
        preprocess_train = data_augmentation_utils.data_augmentation_train_shen_gmic(config_params, mean, std_dev)
        preprocess_val = data_augmentation_utils.data_augmentation_test_shen_gmic(config_params, mean, std_dev)
        
    elif config_params['dataaug']=='shu':
        preprocess_train = data_augmentation_utils.data_augmentation_train_shu(config_params, mean, std_dev)
        preprocess_val = data_augmentation_utils.data_augmentation_test(config_params, mean, std_dev)
        
    elif config_params['dataaug']=='own':
        preprocess_train = data_augmentation_utils.data_augmentation_train(config_params, mean, std_dev)
        preprocess_val = data_augmentation_utils.data_augmentation_test(config_params, mean, std_dev)
    
    elif config_params['dataaug']=='pipnet':
        preprocess_train = data_augmentation_utils.data_augmentation_train_pipnet(config_params, mean, std_dev)
        preprocess_val = data_augmentation_utils.data_augmentation_test_pipnet(config_params, mean, std_dev)
    
    return preprocess_train, preprocess_val

def dataloader(config_params, df_train, df_val, df_test, view_group_indices_train, g):
    preprocess_train, preprocess_val = data_augmentation(config_params, df_train)
    
    batch_sampler = None
    batch_sampler_val = None
    batch_sampler_test = None
    shuffle = True
    sampler1 = None
    if config_params['viewsinclusion'] == 'all' and config_params['learningtype'] == 'MIL':
        if config_params['classimbalance']=='oversampling':
            sampler = data_loaders_utils.CustomGroupbyViewWeightedRandomSampler(config_params, df_train)
            sampler_val = data_loaders_utils.CustomGroupbyViewRandomSampler(df_val, 'val')
        else:
            if config_params['extra'] == 'descendorder' or config_params['extra'] == 'ascendorder' or config_params['extra'] == 'xorder':
                sampler = data_loaders_utils.CustomGroupbyViewFullRandomSamplerOrderBased(view_group_indices_train, config_params['batchsize'], 'train', config_params['extra'])
            else:
                sampler = data_loaders_utils.CustomGroupbyViewFullRandomSampler(view_group_indices_train, config_params['batchsize'], 'train')
            sampler_val = data_loaders_utils.CustomGroupbyViewRandomSampler(df_val, 'val')
        
        sampler_test = data_loaders_utils.CustomGroupbyViewRandomSampler(df_test, 'test')
        
        view_group_length = sampler.__viewlength__()
        view_group_length_val, view_group_name_val = sampler_val.__viewlength__()
        view_group_length_test, view_group_name_test = sampler_test.__viewlength__()
        batch_sampler = data_loaders_utils.CustomGroupbyViewFullRandomBatchSampler(sampler, config_params['batchsize'], view_group_length)
        batch_sampler_val = data_loaders_utils.CustomGroupbyViewRandomBatchSampler(sampler_val, config_params['batchsize'], view_group_length_val, view_group_name_val)
        batch_sampler_test = data_loaders_utils.CustomGroupbyViewRandomBatchSampler(sampler_test, config_params['batchsize'], view_group_length_test, view_group_name_test)
        batch_size1=1
        shuffle=False
    
    else:
        if config_params['classimbalance'] == 'oversampling':
            sampler1 = data_loaders_utils.CustomWeightedRandomSampler(config_params, df_train)
            shuffle = False
        batch_size1 = config_params['batchsize']
    
    if config_params['learningtype'] == 'SIL':
        dataset_gen_train = data_loaders_utils.BreastCancerDataset_generator(config_params, df_train, preprocess_train)
        dataloader_train = DataLoader(dataset_gen_train, batch_size=batch_size1, shuffle=shuffle, num_workers=config_params['numworkers'], collate_fn=data_loaders_utils.MyCollate, worker_init_fn=seed_worker, generator=g, sampler=sampler1, batch_sampler=batch_sampler)    
        
        if config_params['usevalidation']:
            dataset_gen_val = data_loaders_utils.BreastCancerDataset_generator(config_params, df_val, preprocess_val)
            dataloader_val = DataLoader(dataset_gen_val, batch_size=batch_size1, shuffle=False, num_workers=config_params['numworkers'], collate_fn=data_loaders_utils.MyCollate, worker_init_fn=seed_worker, generator=g, batch_sampler=batch_sampler_val)
        
        dataset_gen_test = data_loaders_utils.BreastCancerDataset_generator(config_params, df_test, preprocess_val)
        dataloader_test = DataLoader(dataset_gen_test, batch_size=batch_size1, shuffle=False, num_workers=config_params['numworkers'], collate_fn=data_loaders_utils.MyCollate, worker_init_fn=seed_worker, generator=g, batch_sampler=batch_sampler_test)
    
    elif config_params['learningtype'] == 'MIL' or config_params['learningtype'] == 'MV':
        dataset_gen_train = data_loaders_utils.BreastCancerDataset_generator(config_params, df_train, preprocess_train)
        dataloader_train = DataLoader(dataset_gen_train, batch_size=batch_size1, shuffle=shuffle, num_workers=config_params['numworkers'], collate_fn=data_loaders_utils.MyCollateBreastWise, worker_init_fn=seed_worker, generator=g, sampler=sampler1, batch_sampler=batch_sampler)    
        
        if config_params['usevalidation']:
            dataset_gen_val = data_loaders_utils.BreastCancerDataset_generator(config_params, df_val, preprocess_val)
            dataloader_val = DataLoader(dataset_gen_val, batch_size=batch_size1, shuffle=False, num_workers=config_params['numworkers'], collate_fn=data_loaders_utils.MyCollateBreastWise, worker_init_fn=seed_worker, generator=g, batch_sampler=batch_sampler_val)
        
        dataset_gen_test = data_loaders_utils.BreastCancerDataset_generator(config_params, df_test, preprocess_val)
        dataloader_test = DataLoader(dataset_gen_test, batch_size=batch_size1, shuffle=False, num_workers=config_params['numworkers'], collate_fn=data_loaders_utils.MyCollateBreastWise, worker_init_fn=seed_worker, generator=g, batch_sampler=batch_sampler_test)

    if config_params['usevalidation']:
        return dataloader_train, dataloader_val, dataloader_test
    else:
        return dataloader_train, dataloader_test