# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 16:23:57 2022

@author: PathakS
"""

import torch
import numpy as np
import torch.nn.functional as F

from train_eval import loss_function, evaluation

#import mlflow
#from mlflow.types import Schema, TensorSpec
#from mlflow.models import ModelSignature

def load_model_for_testing(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    print("checkpoint epoch and loss:", checkpoint['epoch'], checkpoint['loss'])
    #print(model.four_view_resnet.feature_extractor.ds_net.bn1.running_mean)
    #print(model.four_view_resnet.feature_extractor.ds_net.bn1.running_var)
    '''ar_mean = []
    ar_std = []
    ar_layer = []
    for name, layer in model.named_modules():
        if 'bn' in name:
            print(name)
            #print(layer.running_mean.mean())
            #print(layer.running_var.mean())
            ar_layer.append(name)
            ar_mean.append(layer.running_mean.mean().item())
            ar_std.append(layer.running_var.mean().item())
            #print(model+'.'+'.'.join(name.split('.')[:-1])+'.running_mean')
            #print(params)
    print(ar_layer)
    print(ar_mean)
    print(ar_std)'''
    return model 

def test(config_params, model, dataloader_test, batches_test, df_test, path_to_results_xlsx, sheetname, epoch):
    """Testing"""
    model.eval()
    total_images=0
    test_loss = 0
    correct = 0
    s=0
    batch_test_no=0
    count_dic_viewwise={}
    eval_subgroup = False
    eval_mode = True
    conf_mat_test=np.zeros((config_params['numclasses'],config_params['numclasses']))
    views_standard=['LCC', 'LMLO', 'RCC', 'RMLO']

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
            #print(df_test.loc[test_idx.item()])
            if config_params['femodel'] == 'gmic_resnet18':
                if config_params['learningtype'] == 'SIL':
                    output_batch_local, output_batch_global, output_batch_fusion, saliency_map, _, _, _, _ = model(test_batch, eval_mode) # compute model output, loss and total train loss over one epoch
                    output_patch_test = None
                elif config_params['learningtype'] == 'MIL':
                    output_batch_local, output_batch_global, output_batch_fusion, saliency_map, _, _, _, _, output_patch_test, _ = model(test_batch, views_names, eval_mode)
                
                if config_params['activation'] == 'sigmoid':
                    output_batch_local = output_batch_local.view(-1)
                    output_batch_global = output_batch_global.view(-1)
                    output_batch_fusion = output_batch_fusion.view(-1)
                    test_labels = test_labels.float()
                    if config_params['milpooling']=='isatt' or config_params['milpooling']=='isgatt' or config_params['milpooling']=='ismean' or config_params['milpooling']=='ismax':
                        test_pred = torch.ge(output_batch_fusion, torch.tensor(0.5)).float()
                    else:
                        test_pred = torch.ge(torch.sigmoid(output_batch_fusion), torch.tensor(0.5)).float()
                
                elif config_params['activation'] == 'softmax':
                    test_pred = output_batch_fusion.argmax(dim=1, keepdim=True)

                loss1 = loss_function.loss_fn_gmic(config_params, bcelogitloss, bceloss, output_batch_local, output_batch_global, output_batch_fusion, saliency_map, test_labels, None, output_patch_test, test_bool=True).item()
                output_test = output_batch_fusion
            
            else:
                if config_params['learningtype'] == 'SIL':
                    output_test = model(test_batch, eval_mode)
                elif config_params['learningtype'] == 'MIL':
                    output_test = model(test_batch, views_names, eval_mode)
                elif config_params['learningtype'] == 'MV':
                    output_test = model(test_batch, views_names, eval_mode)
                
                if config_params['activation']=='sigmoid':
                    if len(output_test.shape)>1:
                        output_test = output_test.squeeze(1)
                    output_test = output_test.view(-1)                                                 
                    test_labels = test_labels.float()
                    if config_params['milpooling']=='isatt' or config_params['milpooling']=='isgatt' or config_params['milpooling']=='ismean' or config_params['milpooling']=='ismax':
                        test_pred = torch.ge(output_test, torch.tensor(0.5)).float()
                    else:
                        test_pred = torch.ge(torch.sigmoid(output_test), torch.tensor(0.5)).float()
                    loss1 = lossfn1(output_test, test_labels).item()
                elif config_params['activation']=='softmax':
                    test_pred = output_test.argmax(dim=1, keepdim=True)
                    loss1 = lossfn1(output_test, test_labels).item()
            
            if batch_test_no==0:
                test_pred_all=test_pred
                test_labels_all=test_labels
                loss_all = torch.tensor([loss1])
                print(output_test.data.shape, flush=True)
                if config_params['activation']=='sigmoid':
                    if config_params['milpooling']=='isatt' or config_params['milpooling']=='isgatt' or config_params['milpooling']=='ismean' or config_params['milpooling']=='ismax':
                        output_all_ten=output_test.data
                    else:
                        output_all_ten=torch.sigmoid(output_test.data)
                elif config_params['activation']=='softmax':
                    output_all_ten=F.softmax(output_test.data,dim=1)
                    if config_params['numclasses'] < 3:
                        output_all_ten=output_all_ten[:,1]
            else:
                test_pred_all=torch.cat((test_pred_all,test_pred),dim=0)
                test_labels_all=torch.cat((test_labels_all,test_labels),dim=0)
                loss_all = torch.cat((loss_all, torch.tensor([loss1])),dim=0)
                if config_params['activation']=='sigmoid':
                    if config_params['milpooling']=='isatt' or config_params['milpooling']=='isgatt' or config_params['milpooling']=='ismean' or config_params['milpooling']=='ismax':
                        output_all_ten=torch.cat((output_all_ten,output_test.data),dim=0)
                    else:
                        output_all_ten=torch.cat((output_all_ten,torch.sigmoid(output_test.data)),dim=0)
                elif config_params['activation']=='softmax':
                    if config_params['numclasses'] < 3:
                        output_all_ten=torch.cat((output_all_ten,F.softmax(output_test.data,dim=1)[:,1]),dim=0)
                    else:
                        output_all_ten=torch.cat((output_all_ten,F.softmax(output_test.data,dim=1)),dim=0)
            
            test_loss += test_labels.size()[0]*loss1 # sum up batch loss
            correct, total_images, conf_mat_test, conf_mat_batch = evaluation.conf_mat_create(test_pred, test_labels, correct, total_images, conf_mat_test, config_params['classes'])
            
            if config_params['viewsinclusion'] == 'all' and config_params['learningtype'] == 'MIL' and (config_params['dataset'] == 'zgt' or config_params['dataset'] == 'cbis-ddsm'):
                count_dic_viewwise = evaluation.calc_viewwise_metric_newplot(views_names, views_standard, count_dic_viewwise, test_labels, test_pred, output_test)
                #print("count_dic_viewwise:", count_dic_viewwise, flush=True)
            batch_test_no+=1
            s=s+test_labels.shape[0]
            print('Test: Step [{}/{}], Loss: {:.4f}'.format(batch_test_no, batches_test, loss1), flush=True)
    
    #predicted_labels = "/".join(path_to_results_xlsx.split('/')[:-1])+'/'+str(config_params['milpooling'])+'_'+str(config_params['attention'])+'_'+str(config_params['dataset'])+'_'+str(config_params['randseeddata'])+"_predlabels.npy"
    #f = open(predicted_labels,'wb')
    #np.save(f, test_pred_all.cpu().numpy())

    #true_labels = "/".join(path_to_results_xlsx.split('/')[:-1])+'/'+str(config_params['milpooling'])+'_'+str(config_params['attention'])+'_'+str(config_params['dataset'])+'_'+str(config_params['randseeddata'])+"_truelabels.npy"
    #f1 = open(true_labels,'wb')
    #np.save(f1, test_labels_all.cpu().numpy())

    #loss_file = "/".join(path_to_results_xlsx.split('/')[:-1])+'/'+"loss.npy"
    #f = open(loss_file,'wb')
    #np.save(f, loss_all.cpu().numpy())

    #prob_file = "/".join(path_to_results_xlsx.split('/')[:-1])+'/'+"probability.npy"
    #f1 = open(prob_file,'wb')
    #np.save(f1, output_all_ten.cpu().numpy())

    running_loss = test_loss/total_images
    print("conf_mat_test:",conf_mat_test, flush=True)
    print("total_images:",total_images, flush=True)
    print("s:",s, flush=True)
    print('\nTest set: total test loss: {:.4f}, Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) \n'.format(
        test_loss, running_loss, correct, total_images, 100. * correct / total_images), flush=True)
    
    per_model_metrics = evaluation.aggregate_performance_metrics(config_params, test_labels_all.cpu().numpy(),test_pred_all.cpu().numpy(), output_all_ten.cpu().numpy())
    metrics_dic = {"Test precision positive class": per_model_metrics[0], "Test precision micro": per_model_metrics[1], "Test precision macro": per_model_metrics[2], "Test recall positive class": per_model_metrics[3], "Test recall micro": per_model_metrics[4], "Test recall macro": per_model_metrics[5], "Test f1 positive class": per_model_metrics[6], "Test f1 micro": per_model_metrics[7], "Test f1 macro": per_model_metrics[8], "Test f1 wt macro": per_model_metrics[9], "Test accuracy": per_model_metrics[10], "Test cohen kappa": per_model_metrics[11], "Test AUC": per_model_metrics[12], "Test AUC wt macro": per_model_metrics[13]}
    #mlflow.log_metrics(metrics_dic, step=epoch)
    per_model_metrics = [epoch, running_loss] + per_model_metrics
    print(per_model_metrics, flush=True)

    if sheetname == 'hyperparam_results':
        hyperparam_details = [config_params['config_file'], config_params['lr'], config_params['wtdecay'], config_params['sm_reg_param'], config_params['trainingmethod'], config_params['optimizer'], config_params['patienceepochs'], config_params['batchsize']] + per_model_metrics
        evaluation.write_results_xlsx(hyperparam_details, config_params['path_to_hyperparam_search'], 'hyperparam_results')
    else:
        evaluation.write_results_xlsx_confmat(config_params, conf_mat_test, path_to_results_xlsx, 'confmat_train_val_test')
        evaluation.write_results_xlsx(per_model_metrics, path_to_results_xlsx, 'test_results')
        evaluation.classspecific_performance_metrics(config_params, test_labels_all.cpu().numpy(),test_pred_all.cpu().numpy(), output_all_ten.cpu().numpy(), path_to_results_xlsx, 'test_results')
    
        if config_params['viewsinclusion'] == 'all' and config_params['learningtype'] == 'MIL' and (config_params['dataset'] == 'zgt' or config_params['dataset'] == 'cbis-ddsm'):
            evaluation.write_results_viewwise(config_params, path_to_results_xlsx, 'metrics_view_wise', count_dic_viewwise)
        
        if eval_subgroup:
            evaluation.results_breastdensity(config_params, df_test, test_labels_all.cpu().numpy(), test_pred_all.cpu().numpy(), output_all_ten.cpu().numpy(), path_to_results_xlsx)
            evaluation.results_birads(config_params, df_test, test_labels_all.cpu().numpy(), test_pred_all.cpu().numpy(), output_all_ten.cpu().numpy(), path_to_results_xlsx)
            #evaluation.results_abnormality(config_params, df_test, test_labels_all.cpu().numpy(), test_pred_all.cpu().numpy(), output_all_ten.cpu().numpy(), path_to_results_xlsx)

        if config_params['learningtype'] == 'SIL':
            evaluation.case_label_from_SIL(config_params, df_test, test_labels_all.cpu().numpy(), test_pred_all.cpu().numpy(), path_to_results_xlsx)
            #evaluation.write_results_xlsx(per_model_metrics_caselevel, path_to_results_xlsx, 'test_results')

def run_test(config_params, model, path_to_model, dataloader_test, batches_test, df_test, path_to_results_xlsx, sheetname, epoch):
    path_to_trained_model = path_to_model
    model1 = load_model_for_testing(model, path_to_trained_model)
    
    #load model from mlflow to check if the model saved in mlflow works
    #path_to_trained_model = '/homes/spathak/multiview_mammogram/multiinstance-learning-mammography/src/mlartifacts/386620200319207163/c75cb1a8951f4fefa88047209e8734e5/artifacts/model_8.tar'
    #model1 = load_model_for_testing(model, path_to_trained_model)

    #save the final model for testing in mlflow
    #input_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 428, 28))])
    #output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 10))])
    #signature = ModelSignature(inputs=input_schema, outputs=output_schema)
    #mlflow.pytorch.log_model(model1, "final_model")
    
    test(config_params, model1, dataloader_test, batches_test,  df_test, path_to_results_xlsx, sheetname, epoch)