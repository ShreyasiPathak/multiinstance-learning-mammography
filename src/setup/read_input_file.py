import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utilities import utils

def input_file_creation(config_params):
    if config_params['learningtype'] == 'SIL':
        if config_params['datasplit'] == 'officialtestset':
            csv_file_path = config_params['SIL_csvfilepath']
            df_modality = pd.read_csv(csv_file_path, sep=';')
            print("df modality shape:",df_modality.shape)
            df_modality = df_modality[~df_modality['Views'].isnull()]
            print("df modality no null view:",df_modality.shape)
            df_modality['FullPath'] = config_params['preprocessed_imagepath']+'/'+df_modality['ShortPath']
            df_modality['Groundtruth'] = df_modality['ImageLabel']
            
            df_train = df_modality[df_modality['ImageName'].str.contains('Training')]
            if config_params['usevalidation']:
                df_train, df_val = train_test_split(df_train, test_size=0.10, shuffle=True, stratify=df_train['Groundtruth'])
            df_test = df_modality[df_modality['ImageName'].str.contains('Test')]

        elif config_params['datasplit'] == 'casebasedtestset':
            csv_file_path = config_params['MIL_csvfilepath']
            df_modality = pd.read_csv(csv_file_path,sep=';')
            print("df modality shape:",df_modality.shape)
            df_modality = df_modality[~df_modality['Views'].isnull()]
            print("df modality no null view:",df_modality.shape)

            #bags with exactly 4 views
            df_modality1  = df_modality[df_modality['Views'].str.split('+').str.len()==4.0]
            print("df_modality 4 views:", df_modality1.shape)
            df_train, df_val, df_test = utils.stratifiedgroupsplit(df_modality1, config_params['randseeddata'])
            
            #bags with views!=4
            df_modality2 = df_modality[df_modality['Views'].str.split('+').str.len()!=4.0]
            print("df_modality views<4:", df_modality2.shape)
            df_train = pd.concat([df_train, df_modality2[df_modality2['Patient_Id'].isin(df_train['Patient_Id'].unique().tolist())]])
            df_val = pd.concat([df_val, df_modality2[df_modality2['Patient_Id'].isin(df_val['Patient_Id'].unique().tolist())]])
            df_test = pd.concat([df_test, df_modality2[df_modality2['Patient_Id'].isin(df_test['Patient_Id'].unique().tolist())]])
            df_modality2 = df_modality2[~df_modality2['Patient_Id'].isin(df_train['Patient_Id'].unique().tolist()+df_val['Patient_Id'].unique().tolist()+df_test['Patient_Id'].unique().tolist())]
            df_train1, df_val1, df_test1 = utils.stratifiedgroupsplit(df_modality2, config_params['randseeddata'])
            
            df_train = pd.concat([df_train, df_train1])
            df_val = pd.concat([df_val, df_val1])
            df_test = pd.concat([df_test, df_test1])
            
            print("Check starting between perfect transfer of patients from case based to single instance based")
            train_check = df_train['FolderName'].unique().tolist()
            val_check = df_val['FolderName'].unique().tolist()
            test_check = df_test['FolderName'].unique().tolist()
            train_check.sort()
            val_check.sort()
            test_check.sort()
            print(len(train_check))
            print(len(val_check))
            print(len(test_check))

            if config_params['labeltouse'] == 'imagelabel':
                df_instances = pd.read_csv(config_params['SIL_csvfilepath'], sep=';')
                df_instances['Groundtruth'] = df_instances['ImageLabel']
                df_instances['FullPath'] = config_params['preprocessed_imagepath']+'/'+df_instances['ShortPath']
                #pd.read_csv('/projects/dso_mammovit/project_kushal/data/MG_training_files_cbis-ddsm_singleinstance_groundtruth.csv', sep=';')

            elif config_params['labeltouse'] == 'caselabel':
                df_instances = pd.read_csv(config_params['SIL_csvfilepath'], sep=';')
                df_instances['Groundtruth'] = df_instances['CaseLabel']
                df_instances['FullPath'] = config_params['preprocessed_imagepath']+'/'+df_instances['ShortPath']
                #df_instances=pd.read_csv('/projects/dso_mammovit/project_kushal/data/MG_training_files_cbis-ddsm_singleinstance_caselabel_groundtruth.csv', sep=';')

            #taking the case-level patient rows from the SIL csv path 
            df_train = df_instances[df_instances['ImageName'].str.split('_').str[:3].str.join('_').isin(df_train['FolderName'].tolist())]
            df_val = df_instances[df_instances['ImageName'].str.split('_').str[:3].str.join('_').isin(df_val['FolderName'].tolist())]
            df_test = df_instances[df_instances['ImageName'].str.split('_').str[:3].str.join('_').isin(df_test['FolderName'].tolist())]

            print('check continues...')
            train_check1=df_train['ImageName'].str.split('_').str[:3].str.join('_').unique().tolist()
            val_check1=df_val['ImageName'].str.split('_').str[:3].str.join('_').unique().tolist()
            test_check1=df_test['ImageName'].str.split('_').str[:3].str.join('_').unique().tolist()
            train_check1.sort()
            val_check1.sort()
            test_check1.sort()
            print(len(train_check1))
            print(len(val_check1))
            print(len(test_check1))
            if train_check==train_check1:
                print('identical train')
            if val_check==val_check1:
                print('identical val')
            if test_check==test_check1:
                print('identical test')
            print("check end!")

    elif config_params['learningtype']=='MIL':
        csv_file_path = config_params['MIL_csvfilepath']
        df_modality=pd.read_csv(csv_file_path,sep=';')
        print("df modality shape:",df_modality.shape)
        df_modality=df_modality[~df_modality['Views'].isnull()]
        print("df modality no null view:",df_modality.shape)
        df_modality['FullPath'] = config_params['preprocessed_imagepath']+'/'+df_modality['ShortPath']

        #bags with exactly 4 views
        df_modality1=df_modality[df_modality['Views'].str.split('+').str.len()==4.0]
        print("df_modality 4 views:", df_modality1.shape)
        df_train, df_val, df_test = utils.stratifiedgroupsplit(df_modality1, config_params['randseeddata'])
        
        #bags with views!=4
        df_modality2=df_modality[df_modality['Views'].str.split('+').str.len()!=4.0]
        print("df_modality views<4:",df_modality2.shape)
        df_train = pd.concat([df_train, df_modality2[df_modality2['Patient_Id'].isin(df_train['Patient_Id'].unique().tolist())]])
        df_val = pd.concat([df_val, df_modality2[df_modality2['Patient_Id'].isin(df_val['Patient_Id'].unique().tolist())]])
        df_test = pd.concat([df_test, df_modality2[df_modality2['Patient_Id'].isin(df_test['Patient_Id'].unique().tolist())]])
        df_modality2 = df_modality2[~df_modality2['Patient_Id'].isin(df_train['Patient_Id'].unique().tolist()+df_val['Patient_Id'].unique().tolist()+df_test['Patient_Id'].unique().tolist())]
        df_train1, df_val1, df_test1 = utils.stratifiedgroupsplit(df_modality2, config_params['randseeddata'])
        
        df_train = pd.concat([df_train,df_train1])
        df_val = pd.concat([df_val,df_val1])
        df_test = pd.concat([df_test,df_test1])
    
    #df_train = df_train[100:130]
    #df_val = df_val[:5]
    #df_test = df_test[:10]
    total_instances = df_modality.shape[0]
    print("Total instances:",total_instances)

    #reset index     
    df_train = df_train.reset_index()
    train_instances = df_train.shape[0]
    print("Train:", utils.stratified_class_count(df_train))
    print("training instances:", train_instances)
    if config_params['usevalidation']:
        df_val = df_val.reset_index()
        val_instances = df_val.shape[0]
        print("Val:",utils.stratified_class_count(df_val))
        print("Validation instances:", val_instances)
    df_test = df_test.reset_index()
    test_instances = df_test.shape[0]
    print("Test:", utils.stratified_class_count(df_test)) 
    print("Test instances:", test_instances) 
            
    if config_params['viewsinclusion'] == 'all':
        #group by view
        view_group_indices, view_group_names_train = utils.groupby_view_train(df_train)
        print(view_group_names_train)
        
        df_val, view_group_names_val = utils.groupby_view_test(df_val)
        print(view_group_names_val)
        
        df_test, view_group_names_test = utils.groupby_view_test(df_test)
        print(view_group_names_test)
        #print(df_test)
    elif config_params['viewsinclusion'] == 'standard':
        view_group_indices = None

    #calculate number of batches
    if config_params['viewsinclusion'] == 'all':
        numbatches_train = int(sum(np.ceil(np.array(list(view_group_names_train.values()))/config_params['batchsize'])))
        numbatches_val = int(sum(np.ceil(np.array(list(view_group_names_val.values()))/config_params['batchsize'])))
        numbatches_test = int(sum(np.ceil(np.array(list(view_group_names_test.values()))/config_params['batchsize'])))
    else:
        #if config_params['class_imbalance']=='oversampling':
        #    batches_train=int(math.ceil(sampler1.__len__()/config_params['batch_size']))
        #else:
        numbatches_train = int(math.ceil(train_instances/config_params['batchsize']))
        
        if config_params['usevalidation']:
            numbatches_val = int(math.ceil(val_instances/config_params['batchsize']))
        
        numbatches_test = int(math.ceil(test_instances/config_params['batchsize']))
    
    if config_params['usevalidation']:
        return df_train, df_val, df_test, numbatches_train, numbatches_val, numbatches_test, view_group_indices
    else:
        return df_train, df_test, numbatches_train, numbatches_test, view_group_indices