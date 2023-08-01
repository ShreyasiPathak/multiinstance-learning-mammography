import torch
import numpy as np
import pandas as pd
import pickle
import torch.nn.functional as F

from train_eval import loss_function, test

activation_feature = {}
def get_activation(name):
    def hook(model, input, output):
        output = output[0].detach()
        output = torch.transpose(output, 2, 1)  # KxN 10,2,1->10,1,2 #Nx4x1->Nx1x4
        output = F.softmax(output, dim=2)
        #print(output)
        #if activation_feature=={}:
        activation_feature[name] = output
        #else:
        #    activation_feature[name] = torch.cat((activation_feature.get(name),output),dim=2)
        #print(activation_feature[name])
    return hook

def mixed_labels_within_case(df):
    #cbis
    df['FolderName'] = df['ShortPath'].str.split('/').str[0]
    conflict_grp = df.groupby(by=['FolderName']).filter(lambda x: len(np.unique(x['ImageLabel']))>1)
    print(conflict_grp.groupby('FolderName').size())

def attention_weights(config_params, model, dataloader_test, batches_test, df_test, path_to_output):
    """Testing"""
    model.eval()
    total_images=0
    test_loss = 0
    s=0
    batch_test_no=0
    image_label_percase = []
    groundtruth_dic = {'benign':0,'malignant':1}
    
    image_csv_file_path = config_params['SIL_csvfilepath']
    df_image = pd.read_csv(image_csv_file_path, sep=';')

    activation_feature_tensor={'malignant':{},'benign':{}}
    if config_params['femodel'] == 'gmic_resnet18':
        bcelogitloss, bceloss = loss_function.loss_fn_gmic_initialize(config_params, None, test_bool=True)
    else:
        if config_params['activation'] == 'softmax':
            lossfn1 = loss_function.loss_fn_crossentropy(config_params, None, test_bool=True)
        elif config_params['activation'] == 'sigmoid':
            lossfn1 = loss_function.loss_fn_bce(config_params, None, test_bool=True)
    
    if config_params['attention']=='breastwise':
        model.milpooling_block.model_attention_fusion_img.register_forward_hook(get_activation('model_attention_perbreast'))
        model.milpooling_block.model_attention_fusion_side.register_forward_hook(get_activation('model_attention_both'))
    elif config_params['attention']=='imagewise':
        model.milpooling_block.model_attention_fusion_img.register_forward_hook(get_activation('model_attention'))
    
    
    with torch.no_grad():
        for test_idx, test_batch, test_labels, views_names in dataloader_test:
            test_batch, test_labels = test_batch.to(config_params['device']), test_labels.to(config_params['device'])
            test_labels = test_labels.view(-1)
            print("test batch:", test_batch.shape)
            print("Views:", views_names)
            '''flag = 0
            for id_case in test_idx:
                print(id_case.item())
                df_image['FolderName'] = df_image['ShortPath'].str.split('/').str[0]
                df_image_case = df_image[df_image['FolderName'] == df_test.loc[id_case.item(), 'ShortPath']]
                #for case_grp in df_image_batch.groupby(by='FolderName'):
                df_image_case = df_image_case.sort_values(by='Views')
                if df_image_case['Views'].tolist()!= views_names:
                    input('halt')
                image_label_percase.append(df_image_case['ImageLabel'].map(groundtruth_dic).tolist()) 
                if len(np.unique(df_image_case['ImageLabel'].map(groundtruth_dic)))>1:
                    print(batch_test_no)
                    print(id_case)
                    print(df_image_case)
                    flag = 1
                    input('halt')
                #print(df_test.loc[id_case.item(), 'ShortPath'])
                #print(df_image_case)
                #print(image_label_percase)
                #input('halt2')
            '''
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
            
            breast_split = np.array([view[0] for view in views_names])
            breast_split = breast_split.tolist()
            if batch_test_no==0:
                if len(views_names)>1:
                    #if flag:
                    #    print(activation_feature['model_attention'])
                    for view_index, view in enumerate(views_names):
                        if config_params['attention']=='imagewise':
                            activation_feature_tensor['malignant'][view]=activation_feature['model_attention'][:,:,view_index].squeeze(1)[test_labels==1]
                            activation_feature_tensor['benign'][view]=activation_feature['model_attention'][:,:,view_index].squeeze(1)[test_labels==0]
                            #activation_feature_tensor_avg[view]=torch.pow(activation_feature['model_attention'][:,:,view_index].squeeze(1)-(1/len(views_names)),2)
                            print("malignant:",view,activation_feature_tensor['malignant'][view].shape)
                            print("benign:", view, activation_feature_tensor['benign'][view].shape)
                        elif config_params['attention']=='breastwise':
                            if view[0]=='L':
                                print(activation_feature['model_attention_both'][:,:,0].squeeze(1))
                                print(activation_feature['model_attention_perbreast'][:,:,int(view_index/2)].squeeze(1))
                                if (breast_split.count('L')==1) and (breast_split.count('R')==1):
                                    final_att_wt_L = activation_feature['model_attention_both'][:,:,0].squeeze(1)
                                elif (breast_split.count('L')==0) or (breast_split.count('R')==0):
                                    final_att_wt_L = activation_feature['model_attention_perbreast'][:,:,int(view_index/2)].squeeze(1)
                                else:
                                    final_att_wt_L=activation_feature['model_attention_perbreast'][:,:,int(view_index/2)].squeeze(1)*activation_feature['model_attention_both'][:,:,0].squeeze(1)
                                print(final_att_wt_L)
                                activation_feature_tensor['malignant'][view]=final_att_wt_L[test_labels==1]
                                activation_feature_tensor['benign'][view]=final_att_wt_L[test_labels==0]
                            elif view[0]=='R':
                                print(activation_feature['model_attention_both'][:,:,1].squeeze(1))
                                print(activation_feature['model_attention_perbreast'][:,:,int(view_index/2)].squeeze(1))
                                if (breast_split.count('L')==1) and (breast_split.count('R')==1):
                                    final_att_wt_R=activation_feature['model_attention_both'][:,:,1].squeeze(1)
                                elif (breast_split.count('L')==0) or (breast_split.count('R')==0):
                                    final_att_wt_R=activation_feature['model_attention_perbreast'][:,:,int(view_index/2)].squeeze(1)
                                else:
                                    final_att_wt_R=activation_feature['model_attention_perbreast'][:,:,int(view_index/2)].squeeze(1)*activation_feature['model_attention_both'][:,:,1].squeeze(1)
                                print(final_att_wt_R)
                                activation_feature_tensor['malignant'][view]=final_att_wt_R[test_labels==1]
                                activation_feature_tensor['benign'][view]=final_att_wt_R[test_labels==0]
                            print("malignant:",view,activation_feature_tensor['malignant'][view].shape)
                            print("benign:",view,activation_feature_tensor['benign'][view].shape)
                            #print(activation_feature['model_attention_perbreast'].shape)
                            #print(activation_feature['model_attention_both'].shape)
            else:
                if len(views_names)>1:
                    #print("malignant:",test_labels[test_labels==1].shape)
                    #print("benign:",test_labels[test_labels==0].shape)
                    #if flag:
                    #    print(activation_feature['model_attention'])
                    for view_index, view in enumerate(views_names):
                        if config_params['attention']=='imagewise':
                            if view in activation_feature_tensor['malignant'].keys():
                                activation_feature_tensor['malignant'][view]=torch.cat((activation_feature_tensor['malignant'][view],activation_feature['model_attention'][:,:,view_index].squeeze(1)[test_labels==1]),dim=0)
                            else:
                                activation_feature_tensor['malignant'][view]=activation_feature['model_attention'][:,:,view_index].squeeze(1)[test_labels==1]
                            if view in activation_feature_tensor['benign'].keys():    
                                activation_feature_tensor['benign'][view]=torch.cat((activation_feature_tensor['benign'][view],activation_feature['model_attention'][:,:,view_index].squeeze(1)[test_labels==0]),dim=0)
                            else:
                                activation_feature_tensor['benign'][view]=activation_feature['model_attention'][:,:,view_index].squeeze(1)[test_labels==0]
                                #activation_feature_tensor_avg[view]=torch.cat((activation_feature_tensor[view],torch.pow(activation_feature['model_attention'][:,:,view_index].squeeze(1)-(1/len(views_names)),2)),dim=0)
                            print("malignant:",view,activation_feature_tensor['malignant'][view].shape)
                            print("benign:", view, activation_feature_tensor['benign'][view].shape)
                        
                        elif config_params['attention']=='breastwise':
                            if view[0]=='L':
                                if (breast_split.count('L')==1) and (breast_split.count('R')==1):
                                    final_att_wt_L=activation_feature['model_attention_both'][:,:,0].squeeze(1)
                                elif (breast_split.count('L')==0) or (breast_split.count('R')==0):
                                    final_att_wt_L=activation_feature['model_attention_perbreast'][:,:,int(view_index/2)].squeeze(1)
                                else:
                                    final_att_wt_L=activation_feature['model_attention_perbreast'][:,:,int(view_index/2)].squeeze(1)*activation_feature['model_attention_both'][:,:,0].squeeze(1)    
                                if view in activation_feature_tensor['malignant'].keys():
                                    activation_feature_tensor['malignant'][view]=torch.cat((activation_feature_tensor['malignant'][view],final_att_wt_L[test_labels==1]),dim=0)
                                else:
                                    activation_feature_tensor['malignant'][view]=final_att_wt_L[test_labels==1]
                                if view in activation_feature_tensor['benign'].keys():
                                    activation_feature_tensor['benign'][view]=torch.cat((activation_feature_tensor['benign'][view],final_att_wt_L[test_labels==0]),dim=0)
                                else:
                                    activation_feature_tensor['benign'][view]=final_att_wt_L[test_labels==0]
                            elif view[0]=='R':
                                if (breast_split.count('L')==1) and (breast_split.count('R')==1):
                                    final_att_wt_R=activation_feature['model_attention_both'][:,:,1].squeeze(1)
                                elif (breast_split.count('L')==0) or (breast_split.count('R')==0):
                                    final_att_wt_R=activation_feature['model_attention_perbreast'][:,:,int(view_index/2)].squeeze(1)
                                else:
                                    final_att_wt_R=activation_feature['model_attention_perbreast'][:,:,int(view_index/2)].squeeze(1)*activation_feature['model_attention_both'][:,:,1].squeeze(1)
                                
                                if view in activation_feature_tensor['malignant'].keys():    
                                    activation_feature_tensor['malignant'][view]=torch.cat((activation_feature_tensor['malignant'][view],final_att_wt_R[test_labels==1]),dim=0)
                                else:
                                    activation_feature_tensor['malignant'][view]=final_att_wt_R[test_labels==1]
                                if view in activation_feature_tensor['benign'].keys():
                                    activation_feature_tensor['benign'][view]=torch.cat((activation_feature_tensor['benign'][view],final_att_wt_R[test_labels==0]),dim=0)
                                else:
                                    activation_feature_tensor['benign'][view]=final_att_wt_R[test_labels==0]
                            print("malignant:",view,activation_feature_tensor['malignant'][view].shape)
                            print("benign:", view, activation_feature_tensor['benign'][view].shape)
            
            test_loss += test_labels.size()[0]*loss1 # sum up batch loss
            batch_test_no+=1
            s=s+test_labels.shape[0]
            print ('Test: Step [{}/{}], Loss: {:.4f}'.format(batch_test_no, batches_test, loss1))
    
    attention_wt_path_np = path_to_output+'/'+str(config_params['milpooling'])+'_'+str(config_params['attention'])+'_'+str(config_params['dataset'])+'_'+str(config_params['randseeddata'])+".pkl"
    
    for key1 in activation_feature_tensor.keys():
        for key2 in activation_feature_tensor[key1].keys():
            activation_feature_tensor[key1][key2]=activation_feature_tensor[key1][key2].cpu()
    
    with open(attention_wt_path_np, 'wb') as handle:
        pickle.dump(activation_feature_tensor, handle, protocol=pickle.HIGHEST_PROTOCOL)

def save_attentionwt(config_params, model, path_to_model, dataloader_test, batches_test, df_test, path_to_attentionwt):
    path_to_trained_model = path_to_model
    model1 = test.load_model_for_testing(model, path_to_trained_model)
    attention_weights(config_params, model1, dataloader_test, batches_test,  df_test, path_to_attentionwt)
