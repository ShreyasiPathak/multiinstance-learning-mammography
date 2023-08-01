# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 10:45:34 2022

@author: PathakS
"""

import torch
import pickle
import matplotlib.pyplot as plt

from train_eval import loss_function, test

activation_feature = {}
def get_activation(name):
    def hook(model, input1, output):
        if name=='baglevel_fusionfeature':
            print(input1[0].shape)
            activation_feature[name] = input1[0]
            print(output.shape)
        else:
            print(name, output.shape)
            activation_feature[name] = output.detach()
    return hook

def feature_vector(config_params, model, dataloader_test, batches_test, hook):
    """Testing"""
    model.eval()
    total_images=0
    test_loss = 0
    s=0
    batch_test_no=0
    activation_feature_tensor={'malignant':{},'benign':{}}
    
    if config_params['femodel'] == 'gmic_resnet18':
        bcelogitloss, bceloss = loss_function.loss_fn_gmic_initialize(config_params, None, test_bool=True)
    else:
        if config_params['activation'] == 'softmax':
            lossfn1 = loss_function.loss_fn_crossentropy(config_params, None, test_bool=True)
        elif config_params['activation'] == 'sigmoid':
            lossfn1 = loss_function.loss_fn_bce(config_params, None, test_bool=True)
    
    if hook == 'baglevel_fusionfeature':
        model.milpooling_block.classifier_fusion.register_forward_hook(get_activation(hook))
    
    elif hook == 'ds_net.layer4.conv2':
        model.four_view_resnet.feature_extractor.ds_net.layer4[1].conv2.register_forward_hook(get_activation(hook))
    
    elif hook == 'ds_net.layer3.conv2':
        model.four_view_resnet.feature_extractor.ds_net.layer3[1].conv2.register_forward_hook(get_activation(hook))

    elif hook == 'ds_net.conv1':
        model.four_view_resnet.feature_extractor.ds_net.conv1.register_forward_hook(get_activation(hook))
    
    elif hook == 'dn_resnet.conv1':
        model.four_view_resnet.feature_extractor.dn_resnet.conv1.register_forward_hook(get_activation(hook))

    elif hook == 'dn_resnet.layer3.conv2':
        model.four_view_resnet.feature_extractor.dn_resnet.layer3[1].conv2.register_forward_hook(get_activation(hook))
    
    elif hook == 'dn_resnet.layer4.conv2':
        model.four_view_resnet.feature_extractor.dn_resnet.layer4[1].conv2.register_forward_hook(get_activation(hook))

    activation_feature_tensor={}
    with torch.no_grad():
        for test_idx, test_batch, test_labels, views_names in dataloader_test:
            test_batch, test_labels = test_batch.to(config_params['device']), test_labels.to(config_params['device'])
            test_labels = test_labels.view(-1)
            print("test batch:", test_batch.shape)
            print("Views:", views_names)
            if config_params['femodel'] == 'gmic_resnet18':
                if config_params['learningtype'] == 'SIL':
                    output_batch_local, output_batch_global, output_batch_fusion, saliency_map = model(test_batch) # compute model output, loss and total train loss over one epoch
                elif config_params['learningtype'] == 'MIL':
                    output_batch_local, output_batch_global, output_batch_fusion, saliency_map, _, _, _, _ = model(test_batch, views_names)
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
            
            print("activation feature hook:", activation_feature[hook].shape)
            if activation_feature[hook].shape[0]>1:
                for roi_id in range(activation_feature[hook].shape[0]):
                    print(roi_id)
                    act = activation_feature[hook][roi_id, :, :, :].cpu().numpy()
                    #print(act[0,:,:])
                    #plt.imshow(act[0,:,:], cmap='gray')
                    #print(act.shape)
                    #plt.show()
                    fig, axarr = plt.subplots(8, 8)
                    for idx, ax in enumerate(axarr.flatten()):
                        ax.imshow(act[idx, :, :], cmap = 'gray')
                    plt.setp(axarr, xticks=[], yticks=[], frame_on=False)
                    plt.tight_layout(h_pad=0.2, w_pad=0.01)
                    plt.show()
            else:
                act = activation_feature[hook].squeeze().cpu().numpy()
                #print(act[0,:,:])
                #plt.imshow(act[0,:,:], cmap='gray')
                #print(act.shape)
                #plt.show()
                fig, axarr = plt.subplots(8, 8)
                for idx, ax in enumerate(axarr.flatten()):
                    ax.imshow(act[idx, :, :], cmap = 'gray')
                plt.setp(axarr, xticks=[], yticks=[], frame_on=False)
                plt.tight_layout(h_pad=0.2, w_pad=0.01)
                plt.show()

            input('halt')

            activation_feature_tensor[test_idx.item()] = [activation_feature[hook].detach().squeeze(1).cpu(), test_labels, test_pred, torch.sigmoid(output_test), loss1]
            
            test_loss += test_labels.size()[0]*loss1 # sum up batch loss
            
            batch_test_no+=1
            s=s+test_labels.shape[0]
            print ('Test: Step [{}/{}], Loss: {:.4f}'.format(batch_test_no, batches_test, loss1))
    
    return activation_feature, activation_feature_tensor

def save_featurevector(config_params, model, path_to_model, dataloader_test, batches_test, df_test, path_to_attentionwt):
    path_to_trained_model = path_to_model
    model1 = test.load_model_for_testing(model, path_to_trained_model)
    hook = 'baglevel_fusionfeature'
    activation_feature, test_set_details = feature_vector(config_params, model1, dataloader_test, batches_test,  df_test, path_to_attentionwt, hook)
    featurevec_path = path_to_attentionwt+'/'+str(config_params['milpooling'])+'_'+str(config_params['attention'])+'_'+str(config_params['dataset'])+'_'+str(config_params['randseeddata'])+"_"+str(hook)+".pkl"
    with open(featurevec_path, 'wb') as handle:
        pickle.dump(test_set_details, handle, protocol=pickle.HIGHEST_PROTOCOL)

def visualize_feature_maps(config_params, model, path_to_model, dataloader_test, batches_test, df_test, path_to_attentionwt):
    path_to_trained_model = path_to_model
    model1 = test.load_model_for_testing(model, path_to_trained_model)
    hook = 'dn_resnet.layer3.conv2'
    activation_feature, test_set_details = feature_vector(config_params, model1, dataloader_test, batches_test, hook)
    print("visualize:", activation_feature[hook].shape)
    act = activation_feature[hook].squeeze()
    print(act.shape)
    fig, axarr = plt.subplots(act.size(0))
    for idx in range(act.size(0)):
        axarr[idx].imshow(act[idx])
    plt.show()