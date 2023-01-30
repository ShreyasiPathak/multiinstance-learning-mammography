# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 16:43:24 2021

@author: PathakS
"""

import re
import os
import math
import torch
import datetime
import argparse
import random

import glob
import numpy as np
import pandas as pd
from sklearn import metrics

import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.ops.focal_loss import sigmoid_focal_loss

from train_eval import test, optimization, loss_function, evaluation, data_loader
from models import sil_mil_model
from utilities import pytorchtools, utils
from setup import read_config_file, read_input_file, output_files_setup

#import tensorboard_log

#pd.set_option('display.max_columns', None)  # or 1000
#pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199

def set_random_seed(config_params):
    #random state initialization of the code - values - 8, 24, 30
    torch.manual_seed(config_params['randseedother']) 
    torch.cuda.manual_seed(config_params['randseedother'])
    torch.cuda.manual_seed_all(config_params['randseedother'])
    np.random.seed(config_params['randseeddata'])
    random.seed(config_params['randseeddata'])
    g = torch.Generator()
    g.manual_seed(config_params['randseedother'])
    torch.backends.cudnn.deterministic = True
    return g

def model_initialization(config_params):
    if config_params['learningtype'] == 'SIL':
        model = sil_mil_model.SILmodel(config_params)
    elif config_params['learningtype'] == 'MIL':
        model = sil_mil_model.MILmodel(config_params)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    #print(model)
    model.to(config_params['device'])
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total model parameters:", pytorch_total_params)

    return model, pytorch_total_params

def model_checkpoint(config_params, path_to_model):
    if config_params['patienceepochs']:
        modelcheckpoint = pytorchtools.EarlyStopping(path_to_model=path_to_model, patience=config_params['patienceepochs'], verbose=True)
    elif config_params['usevalidation']:
        modelcheckpoint = pytorchtools.ModelCheckpoint(path_to_model=path_to_model, verbose=True)
    return modelcheckpoint

def train(config_params, model, path_to_model, data_iterator_train, data_iterator_val, batches_train, batches_val, df_train):
    '''Training'''
    if config_params['usevalidation']:
        modelcheckpoint = model_checkpoint(config_params, path_to_model)
    optimizer = optimization.optimizer_fn(config_params, model)
    scheduler = optimization.select_lr_scheduler(config_params, optimizer)
    class_weights_train = loss_function.class_imbalance(config_params, df_train)

    if os.path.isfile(path_to_model):
        model, _, start_epoch = utils.load_model(model, optimizer, path_to_model)
        print("start epoch:",start_epoch)
        print("lr:",optimizer.param_groups[0]['lr'])
    else:
        start_epoch = 0
        
    if config_params['femodel'] == 'gmic_resnet18':
        bcelogitloss, bceloss = loss_function.loss_fn_gmic_initialize(config_params, class_weights_train, test_bool=False)
    else:
        if config_params['activation'] == 'softmax':
            lossfn = loss_function.loss_fn_crossentropy(config_params, class_weights_train, test_bool=False)
        elif config_params['activation'] == 'sigmoid':
            lossfn = loss_function.loss_fn_bce(config_params, class_weights_train, test_bool=False)
    
    for epoch in range(start_epoch,config_params['maxepochs']):
        model.train()
        loss_train=0.0
        correct_train=0
        conf_mat_train=np.zeros((2,2))
        total_images_train=0
        batch_no=0

        if config_params['trainingmethod'] == 'multisteplr1':
            model = utils.layer_selection_for_training(model,epoch, config_params['trainingmethod'], epoch_step=5)
        
        for train_idx, train_batch, train_labels, views_names in data_iterator_train:
            train_batch = train_batch.to(config_params['device'])
            train_labels = train_labels.to(config_params['device'])
            train_labels = train_labels.view(-1)
            print("train batch:",train_batch.shape)

            if config_params['viewsinclusion'] == 'all' and config_params['extra'] == 'dynamic_training':
                model, optimizer, state_before_optim, lr_before_optim = utils.dynamic_training(config_params, views_names, model, optimizer, None, None, True)
            
            if config_params['femodel'] == 'gmic_resnet18':
                if config_params['learningtype'] == 'SIL':
                    output_batch_local, output_batch_global, output_batch_fusion, saliency_map = model(train_batch) # compute model output, loss and total train loss over one epoch
                elif config_params['learningtype'] == 'MIL':
                    output_batch_local, output_batch_global, output_batch_fusion, saliency_map = model(train_batch, views_names)
                output_batch_local = output_batch_local.view(-1)
                output_batch_global = output_batch_global.view(-1)
                output_batch_fusion = output_batch_fusion.view(-1)
                train_labels = train_labels.float()
                pred = torch.ge(torch.sigmoid(output_batch_fusion), torch.tensor(0.5)).float()
                loss = loss_function.loss_fn_gmic(config_params, bcelogitloss, bceloss, output_batch_local, output_batch_global, output_batch_fusion, saliency_map, train_labels, class_weights_train, test_bool=False)
            
            else:
                if config_params['learningtype'] == 'SIL':
                    output_batch = model(train_batch)
                elif config_params['learningtype'] == 'MIL':
                    output_batch = model(train_batch, views_names)
                
                if config_params['activation'] == 'sigmoid':
                    output_batch = output_batch.squeeze(1)
                    output_batch = output_batch.view(-1)                                                                          
                    train_labels = train_labels.float()
                    pred = torch.ge(torch.sigmoid(output_batch), torch.tensor(0.5)).float()
                    if config_params['classimbalance'] == 'focalloss':
                        loss = sigmoid_focal_loss(output_batch, train_labels, alpha=-1, reduction='mean')
                    else:
                        loss = lossfn(output_batch, train_labels)
                
                elif config_params['activation'] == 'softmax':
                    pred = output_batch.argmax(dim=1, keepdim=True)
                    loss = lossfn(output_batch, train_labels)

            loss_train+=(train_labels.size()[0]*loss.item())

            '''weights_before_backprop = []
            parameter_name=[]

            for name, param in model.named_parameters(): # loop the weights in the model before updating and store them
                parameter_name.append(name)
                weights_before_backprop.append(param.clone())
            '''

            optimizer.zero_grad()  # clear previous gradients, compute gradients of all variables wrt loss
            loss.backward()
            optimizer.step() # performs updates using calculated gradients
            batch_no=batch_no+1

            if config_params['viewsinclusion'] == 'all' and config_params['extra'] == 'dynamic_training':
                model, optimizer = utils.dynamic_training(config_params, views_names, model, optimizer, state_before_optim, lr_before_optim, False)

            '''weights_after_backprop = [] # weights after backprop
            for name, param in model.named_parameters():
                weights_after_backprop.append(param.clone()) # only layer1's weight should update, layer2 is not used
            
            for i in zip(parameter_name, weights_before_backprop, weights_after_backprop):
                if torch.equal(i[1],i[2]):
                    print(i[0], torch.equal(i[1],i[2]))
            input('halt')
            '''

            #performance metrics of training dataset
            correct_train, total_images_train, conf_mat_train, _ = evaluation.conf_mat_create(pred, train_labels, correct_train, total_images_train, conf_mat_train, config_params['classes'])
            print('Train: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, config_params['maxepochs'], batch_no, batches_train, loss.item()))
        
        if scheduler!=None:
            current_lr=scheduler.get_last_lr()[0]
        else:
            current_lr=optimizer.param_groups[0]['lr']
        print("current lr:",current_lr)
        
        running_train_loss = loss_train/total_images_train

        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        
        if config_params['usevalidation']:
            correct_test, total_images_val, loss_val, conf_mat_val, auc_val = validation(config_params, model, data_iterator_val, batches_val, df_val, epoch)
            valid_loss = loss_val/total_images_val
            evaluation.results_store_excel(True, True, False, None, correct_train, total_images_train, loss_train, correct_test, total_images_val, loss_val, epoch, conf_mat_train, conf_mat_val, current_lr, auc_val, path_to_results_xlsx, path_to_results_text)
        
        if config_params['patienceepochs']:
            modelcheckpoint(valid_loss, model, optimizer, epoch, conf_mat_train, conf_mat_val, running_train_loss, auc_val)
            if modelcheckpoint.early_stop:
                print("Early stopping",epoch+1)
                break
        else:
            if config_params['usevalidation']:
                modelcheckpoint(valid_loss, model, optimizer, epoch, conf_mat_train, conf_mat_val, running_train_loss, auc_val)
            else:
                utils.save_model(model, optimizer, epoch, running_train_loss, path_to_model)
                per_model_metrics, conf_mat_test = test(config_params, model, path_to_model, data_iterator_val, batches_val, df_test)
                evaluation.results_store_excel(True, False, True, per_model_metrics, correct_train, total_images_train, loss_train, None, None, None, epoch, conf_mat_train, None, current_lr, None, path_to_results_xlsx, path_to_results_text)
                evaluation.write_results_xlsx_confmat(conf_mat_test, path_to_results_xlsx, 'confmat_train_val_test')
                evaluation.write_results_xlsx(per_model_metrics, path_to_results_xlsx, 'test_results')

        if scheduler!=None: 
            scheduler.step()
    
    if config_params['usevalidation']:
        evaluation.write_results_xlsx_confmat(modelcheckpoint.conf_mat_train_best, path_to_results_xlsx, 'confmat_train_val_test')
        evaluation.write_results_xlsx_confmat(modelcheckpoint.conf_mat_test_best, path_to_results_xlsx, 'confmat_train_val_test')
   
    print('Finished Training')
    
def validation(config_params, model, data_iterator_val, batches_val, df_val, epoch):
    """Validation"""
    model.eval()
    total_images=0
    val_loss = 0
    correct = 0
    s=0
    batch_val_no=0
    conf_mat_val=np.zeros((2,2))

    class_weights_val = loss_function.class_imbalance(config_params, df_val)

    if config_params['femodel'] == 'gmic_resnet18':
        bcelogitloss_val, bceloss_val = loss_function.loss_fn_gmic_initialize(config_params, class_weights_val, test_bool=False)
    else:
        if config_params['activation'] == 'softmax':
            lossfn1 = loss_function.loss_fn_crossentropy(config_params, class_weights_val, test_bool=False)
        elif config_params['activation'] == 'sigmoid':
            lossfn1 = loss_function.loss_fn_bce(config_params, class_weights_val, test_bool=False)
    
    with torch.no_grad():   
        for val_idx, val_batch, val_labels, views_names in data_iterator_val:
            val_batch, val_labels = val_batch.to(config_params['device']), val_labels.to(config_params['device'])
            val_labels = val_labels.view(-1)#.float()
            if config_params['femodel'] == 'gmic_resnet18':
                if config_params['learningtype'] == 'SIL':
                    output_batch_local_val, output_batch_global_val, output_batch_fusion_val, saliency_map_val = model(val_batch) # compute model output, loss and total train loss over one epoch
                elif config_params['learningtype'] == 'MIL':
                    output_batch_local_val, output_batch_global_val, output_batch_fusion_val, saliency_map_val = model(val_batch, views_names)
                
                output_batch_local_val = output_batch_local_val.view(-1)
                output_batch_global_val = output_batch_global_val.view(-1)
                output_batch_fusion_val = output_batch_fusion_val.view(-1)
                val_labels = val_labels.float()
                val_pred = torch.ge(torch.sigmoid(output_batch_fusion_val), torch.tensor(0.5)).float()
                loss1 = loss_function.loss_fn_gmic(config_params, bcelogitloss_val, bceloss_val, output_batch_local_val, output_batch_global_val, output_batch_fusion_val, saliency_map_val, val_labels, class_weights_val, test_bool=False).item()
                output_val = output_batch_fusion_val
            else:
                if config_params['learningtype'] == 'SIL':
                    output_val = model(val_batch)
                if config_params['learningtype'] == 'MIL':
                    output_val = model(val_batch, views_names)
                if config_params['activation'] == 'sigmoid':
                    output_val = output_val.squeeze(1)
                    output_val = output_val.view(-1)                                                 
                    val_labels=val_labels.float()
                    val_pred = torch.ge(torch.sigmoid(output_val), torch.tensor(0.5)).float()
                    if config_params['classimbalance']=='focalloss':
                        loss1 = sigmoid_focal_loss(output_val, val_labels, alpha=-1, reduction='mean').item()
                    else:
                        loss1 = lossfn1(output_val, val_labels).item()
                elif config_params['activation'] == 'softmax':
                    val_pred = output_val.argmax(dim=1, keepdim=True)
                    loss1 = lossfn1(output_val, val_labels).item()
            
            if batch_val_no==0:
                val_pred_all = val_pred
                val_labels_all = val_labels
                print(output_val.data.shape)
                if config_params['activation'] == 'sigmoid':
                    output_all_ten = torch.sigmoid(output_val.data)
                elif config_params['activation'] == 'softmax':
                    output_all_ten = F.softmax(output_val.data,dim=1)
                    output_all_ten = output_all_ten[:,1]
            else:
                val_pred_all = torch.cat((val_pred_all,val_pred),dim=0)
                val_labels_all = torch.cat((val_labels_all,val_labels),dim=0)
                if config_params['activation'] == 'sigmoid':
                    output_all_ten = torch.cat((output_all_ten,torch.sigmoid(output_val.data)),dim=0)
                elif config_params['activation'] == 'softmax':
                    output_all_ten = torch.cat((output_all_ten,F.softmax(output_val.data,dim=1)[:,1]),dim=0)

            s = s+val_labels.shape[0]    
            val_loss += val_labels.size()[0]*loss1 # sum up batch loss
            correct, total_images, conf_mat_val, _ = evaluation.conf_mat_create(val_pred, val_labels, correct, total_images, conf_mat_val, config_params['classes'])
            
            batch_val_no+=1
            print('Val: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, config_params['maxepochs'], batch_val_no, batches_val, loss1))
    
    print("conf_mat_val:",conf_mat_val)
    print("total_images:",total_images)
    print("s:",s)
    print('\nVal set: total val loss: {:.4f}, Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Epoch:{}\n'.format(
        val_loss, val_loss/total_images, correct, total_images,
        100. * correct / total_images,epoch+1))
    
    auc = metrics.roc_auc_score(val_labels_all.cpu().numpy(), output_all_ten.cpu().numpy())
    return correct, total_images, val_loss, conf_mat_val, auc

if __name__=='__main__':
    #read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file_path",
        type=str,
        help="full path where the config.ini file containing the parameters to run this code is stored",
    )

    parser.add_argument(
        "--num_config_start",
        type=int,
        help="file number of hyperparameter combination to start with; one config file corresponds to one hyperparameter combination",
    )

    parser.add_argument(
        "--num_config_end",
        type=int,
        help="file number of hyperparameter combination to end with; one config file corresponds to one hyperparameter combination",
    )
    args = parser.parse_args()

    num_config_start = args.num_config_start
    num_config_end = args.num_config_end

    #read all instructed config files
    config_file_names = glob.glob(args.config_file_path+'/config*')
    config_file_names = sorted(config_file_names, key=lambda x: int(re.search(r'\d+$', x.split('.')[-2]).group()))
    print("config files to be read:",config_file_names[num_config_start:num_config_end])
    
    for config_file in config_file_names[num_config_start:num_config_end]:
        begin_time = datetime.datetime.now()
        
        print("config file reading:",config_file)
        config_params = read_config_file.read_config_file(config_file)
        
        g = set_random_seed(config_params)
        
        if config_params['usevalidation']:
            path_to_model, path_to_results_xlsx, path_to_results_text, path_to_learning_curve, path_to_log_file, path_to_hyperparam_search = output_files_setup.output_files(config_file, config_params, num_config_start, num_config_end)
            df_train, df_val, df_test, batches_train, batches_val, batches_test, view_group_indices_train = read_input_file.input_file_creation(config_params)
            dataloader_train, dataloader_val, dataloader_test = data_loader.dataloader(config_params, df_train, df_val, df_test, view_group_indices_train, g)
        else:
            path_to_model, path_to_results_xlsx, path_to_results_text, path_to_learning_curve, path_to_log_file = output_files_setup.output_files(config_file, config_params, num_config_start, num_config_end)
            df_train, df_test, batches_train, batches_test, view_group_indices_train = read_input_file.input_file_creation(config_params)
            dataloader_train, dataloader_test = data_loader.dataloader(config_params, df_train, None, df_test, view_group_indices_train, g)
        
        model, total_params = model_initialization(config_params)

        #training the model
        if config_params['usevalidation']:
            train(config_params, model, path_to_model, dataloader_train, dataloader_val, batches_train, batches_val, df_train)
        else:
            train(config_params, model, path_to_model, dataloader_train, dataloader_test, batches_train, batches_test, df_train)

        #hyperparameter results
        if config_params['usevalidation']:
            per_model_metrics_val, _ = test.run_test(config_params, model, path_to_model, dataloader_val, batches_val, df_val)
            hyperparam_details = [config_file.split('/')[-1], config_params['lr'], config_params['wtdecay'], config_params['sm_reg_param'], config_params['trainingmethod'], config_params['optimizer'], config_params['patienceepochs'], config_params['batchsize']] + per_model_metrics_val
            evaluation.write_results_xlsx(hyperparam_details, path_to_hyperparam_search, 'hyperparam_results')

        #test the model
        per_model_metrics_test, conf_mat_test = test.run_test(config_params, model, path_to_model, dataloader_test, batches_test, df_test)
        evaluation.write_results_xlsx_confmat(conf_mat_test, path_to_results_xlsx, 'confmat_train_val_test')
        evaluation.write_results_xlsx(per_model_metrics_test, path_to_results_xlsx, 'test_results')

        '''except Exception as err:
            print("Exception encountered:",err)
            #save the results
            wb.save(path_to_results)
        
            #plot the training and validation loss and accuracy
            #df=pd.read_excel(path_to_results)
            #results_plot(df,file_name)
        '''
        
        f = open(path_to_log_file,'w')
        f.write("Model parameters:"+str(total_params/math.pow(10,6))+'\n')
        f.write("Start time:"+str(begin_time)+'\n')
        f.write("End time:"+str(datetime.datetime.now())+'\n')
        f.write("Execution time:"+str(datetime.datetime.now() - begin_time)+'\n')
        f.close()