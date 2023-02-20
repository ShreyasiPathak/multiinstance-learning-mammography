# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 16:23:57 2022

@author: PathakS
"""

import torch
import numpy as np
import torch.nn.functional as F

from utilities import utils
from train_eval import loss_function, evaluation

def load_model_for_testing(model,path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    print("checkpoint epoch and loss:", checkpoint['epoch'], checkpoint['loss'])
    return model 

def test(config_params, model, dataloader_test, batches_test, df_test):
    """Testing"""
    model.eval()
    total_images=0
    test_loss = 0
    correct = 0
    s=0
    batch_test_no=0
    conf_mat_test=np.zeros((2,2))
    
    if config_params['femodel'] == 'gmic_resnet18':
        bcelogitloss, bceloss = loss_function.loss_fn_gmic_initialize(config_params, None, test_bool=True)
    else:
        if config_params['activation'] == 'softmax':
            lossfn1 = loss_function.loss_fn_crossentropy(config_params, None, test_bool=True)
        elif config_params['activation'] == 'sigmoid':
            lossfn1 = loss_function.loss_fn_bce(config_params, None, test_bool=True)
    
    with torch.no_grad():
        for test_idx, test_batch, test_labels, views_names in dataloader_test:
            test_batch, test_labels = test_batch.to(config_params['device']), test_labels.to(config_params['device'])
            test_labels = test_labels.view(-1)
            
            if config_params['femodel'] == 'gmic_resnet18':
                if config_params['learningtype'] == 'SIL':
                    output_batch_local, output_batch_global, output_batch_fusion, saliency_map = model(test_batch) # compute model output, loss and total train loss over one epoch
                elif config_params['learningtype'] == 'MIL':
                    output_batch_local, output_batch_global, output_batch_fusion, saliency_map = model(test_batch, views_names)
                output_batch_local = output_batch_local.view(-1)
                output_batch_global = output_batch_global.view(-1)
                output_batch_fusion = output_batch_fusion.view(-1)
                test_labels = test_labels.float()
                test_pred = torch.ge(torch.sigmoid(output_batch_fusion), torch.tensor(0.5)).float()
                loss1 = loss_function.loss_fn_gmic(config_params, bcelogitloss, bceloss, output_batch_local, output_batch_global, output_batch_fusion, saliency_map, test_labels, None, test_bool=True).item()
                output_test = output_batch_fusion
            
            else:
                if config_params['learningtype'] == 'SIL':
                    output_test = model(test_batch)
                elif config_params['learningtype'] == 'MIL':
                    output_test = model(test_batch, views_names)
                
                if config_params['activation']=='sigmoid':
                    output_test = output_test.squeeze(1)
                    output_test = output_test.view(-1)                                                 
                    test_labels = test_labels.float()
                    test_pred = torch.ge(torch.sigmoid(output_test), torch.tensor(0.5)).float()
                    loss1 = lossfn1(output_test, test_labels).item()
                elif config_params['activation']=='softmax':
                    test_pred = output_test.argmax(dim=1, keepdim=True)
                    loss1 = lossfn1(output_test, test_labels).item()
            
            if batch_test_no==0:
                test_pred_all=test_pred
                test_labels_all=test_labels
                print(output_test.data.shape, flush=True)
                if config_params['activation']=='sigmoid':
                    output_all_ten=torch.sigmoid(output_test.data)
                elif config_params['activation']=='softmax':
                    output_all_ten=F.softmax(output_test.data,dim=1)
                    output_all_ten=output_all_ten[:,1]
            else:
                test_pred_all=torch.cat((test_pred_all,test_pred),dim=0)
                test_labels_all=torch.cat((test_labels_all,test_labels),dim=0)
                if config_params['activation']=='sigmoid':
                    output_all_ten=torch.cat((output_all_ten,torch.sigmoid(output_test.data)),dim=0)
                elif config_params['activation']=='softmax':
                    output_all_ten=torch.cat((output_all_ten,F.softmax(output_test.data,dim=1)[:,1]),dim=0)
            
            test_loss += test_labels.size()[0]*loss1 # sum up batch loss
            correct, total_images, conf_mat_test, conf_mat_batch = evaluation.conf_mat_create(test_pred, test_labels, correct, total_images, conf_mat_test, config_params['classes'])
            
            #count_dic_viewwise, conf_mat_viewwise = evaluation.save_viewwise_count(views_names, test_batch, conf_mat_batch, batch_test_no)
            
            batch_test_no+=1
            s=s+test_labels.shape[0]
            print ('Test: Step [{}/{}], Loss: {:.4f}'.format(batch_test_no, batches_test, loss1), flush=True)
    
    running_loss = test_loss/total_images
    print("conf_mat_test:",conf_mat_test, flush=True)
    print("total_images:",total_images, flush=True)
    print("s:",s, flush=True)
    print('\nTest set: total test loss: {:.4f}, Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) \n'.format(
        test_loss, running_loss, correct, total_images, 100. * correct / total_images), flush=True)
    
    per_model_metrics = utils.performance_metrics(conf_mat_test,test_labels_all.cpu().numpy(),test_pred_all.cpu().numpy(), output_all_ten.cpu().numpy())
    per_model_metrics = [running_loss] + per_model_metrics
    print(per_model_metrics, flush=True)
    
    #val_stats_viewwise = evaluation.calc_viewwise_metric(count_dic_viewwise, conf_mat_viewwise)
    #sheet4 = evaluation.results_viewwise(sheet4, val_stats_viewwise)
    
    if config_params['learningtype'] == 'SIL':
        per_model_metrics_caselevel = evaluation.case_label_from_SIL(config_params, df_test, test_labels_all.cpu().numpy(), test_pred_all.cpu().numpy())
        return per_model_metrics, conf_mat_test, per_model_metrics_caselevel
    else:
        return per_model_metrics, conf_mat_test

def run_test(config_params, model, path_to_model, dataloader_test, batches_test, df_test):
    path_to_trained_model = path_to_model
    model1 = load_model_for_testing(model, path_to_trained_model)
    if config_params['learningtype'] == 'SIL':
        per_model_metrics_test, conf_mat_test, per_model_metrics_caselevel = test(config_params, model1, dataloader_test, batches_test,  df_test)
        return per_model_metrics_test, conf_mat_test, per_model_metrics_caselevel
    else:
        per_model_metrics_test, conf_mat_test = test(config_params, model1, dataloader_test, batches_test,  df_test)
        return per_model_metrics_test, conf_mat_test