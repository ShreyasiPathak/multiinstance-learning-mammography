import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
import cv2
import matplotlib.cm as cm
import torchvision
from scipy import stats
import math
import openpyxl as op
from sklearn.metrics import confusion_matrix

#from utilities import utils
#from train_eval import test, mask_roi_match, evaluation

def imglabel_f1_plot(meanF1, stddev):
    fig = plt.figure(figsize=(14,16))
    #plt.rcParams['font.size'] = 20
    x = np.arange(6)
    labels = ['IS-Mean$^{img}$','IS-Att$^{img}$', 'IS-GAtt$^{img}$', 'ES-Att$^{img}$', 'ES-GAtt$^{img}$', 'ES-Att$^{side}$']
    #plt.errorbar(x, meanF1, yerr=stddev, fmt = 'o')
    plt.bar(x, meanF1, yerr=stddev, capsize = 3, color ='orange',)
    plt.ylim(0.2)
    plt.xticks(x, labels, rotation=30, fontsize=25)
    plt.xlabel('MIL models', fontsize=25)
    plt.ylabel('F1 score of image-level prediction using MIL models', fontsize=25)
    plt.yticks(fontsize=25)
    plt.savefig('C:/Users/PathakS/OneDrive - Universiteit Twente/PhD/projects/radiology breast cancer/breast-cancer-multiview-mammogram-codes/multiinstance results/results/IEEE-TMI/f1_attmodel_gt_imglabel.pdf', format='pdf')

def match_imagelevel_groundtruth_withattnwt(config_params, exam_name, attwt):
    #print(exam_name)
    df_image = pd.read_csv(config_params['SIL_csvfilepath'], sep=';')
    df_grp = df_image[df_image['StudyInstanceUID']==exam_name].sort_values(by='Views')
    #if np.unique(df_grp['CaseLabel'])[0] == 'malignant':
    df_grp['TrueImageLabel'] = df_grp['ImageLabel'].map({'benign':0, 'malignant':1})
    df_grp['ModelAttwt'] = (np.array(list(attwt[0]))>0.25).astype(int)
    #print(df_grp[['StudyInstanceUID','Views','ImageLabel', 'CaseLabel','ModelAttwt']])
    y_true = df_grp['TrueImageLabel'].values.tolist()
    y_pred = df_grp['ModelAttwt'].values.tolist()
    #y_pred = [1,1,1,1]
    '''
    if len(np.unique(df_grp['ImageLabel']))!=1:
        corr, pval = stats.pointbiserialr(df_grp['TrueImageLabel'].values.tolist(), df_grp['ModelAttwt'].values.tolist())
    else:
        corr = None 
        pval = None 
    '''
    return y_true, y_pred
    
def extract_img_attn_wts(config_params, img_attns):
    if config_params['attention'] == 'imagewise':
        A_all = img_attns.view(img_attns.shape[0], -1).data.cpu().numpy()
    elif config_params['attention'] == 'breastwise':
        A_left = img_attns[0]
        A_right = img_attns[1]
        A_both = img_attns[2]
        '''print(A_left)
        print(A_left.shape)
        print(A_right)
        print(A_right.shape)
        print(A_both)
        print(A_both.shape)
        '''

        if (A_left is not None) and (A_right is not None) and (A_both is not None):
            A_left_both = torch.mul(A_left, A_both[:,:,0])
            A_right_both = torch.mul(A_right, A_both[:,:,1])
            A_all = torch.cat((A_left_both, A_right_both), dim=2)
            A_all = A_all.view(1, -1).data.cpu().numpy()
        #print(A_all)
        #input('halt')
    return A_all

def model_output(config_params, model, dataloader_test, df_test, path_to_results):
    model.eval()
    eval_mode = True
    count = 0
    pval_sum =0
    corr_sum =0
    y_true_all = []
    y_pred_all = []
    with torch.no_grad():
        for test_idx, test_batch, test_labels, views_names in dataloader_test:
            test_batch, test_labels = test_batch.to(config_params['device']), test_labels.to(config_params['device'])
            test_labels = test_labels.view(-1)
            print("test idx:", test_idx.item())
            print("test batch:", test_batch.shape)
            if config_params['femodel'] == 'gmic_resnet18':
                if config_params['learningtype'] == 'SIL':
                    loaded_image = utils.collect_images(config_params, df_test.loc[test_idx.item()])
                    _, _, output_batch_fusion, saliency_map, patch_locations, patch_imgs, patch_attns, _ = model(test_batch, eval_mode) # compute model output, loss and total train loss over one epoch
                    loaded_image = loaded_image[np.newaxis,:,:,:]
                    patch_locations = patch_locations[:,np.newaxis,:]
                    patch_imgs = patch_imgs[:,np.newaxis,:]
                    patch_attns = patch_attns[:,np.newaxis,:]
                    saliency_map = saliency_map[:, np.newaxis, :, :, :]
                    img_attns = None
                    exam_name = df_test.loc[test_idx.item(), 'ImageName']

                elif config_params['learningtype'] == 'MIL':
                    if config_params['dataset'] == 'cbis-ddsm':
                        exam_name = df_test.loc[test_idx.item(), 'FolderName']
                    elif config_params['dataset'] == 'vindr':
                        exam_name = df_test.loc[test_idx.item(), 'StudyInstanceUID']
                    loaded_image, _, _ = utils.collect_cases(config_params, df_test.loc[test_idx.item()])
                    _, _, output_batch_fusion, saliency_map, patch_locations, patch_imgs, patch_attns, img_attns, _ = model(test_batch, views_names, eval_mode)
            
            test_pred = torch.ge(torch.sigmoid(output_batch_fusion), torch.tensor(0.5)).float()
            saliency_maps = saliency_map.cpu().numpy()
            patch_attentions = patch_attns[0, :, :].data.cpu().numpy()

            if test_labels.item() == 1:
                img_attns = extract_img_attn_wts(config_params, img_attns)
                y_true, y_pred = match_imagelevel_groundtruth_withattnwt(config_params, exam_name, img_attns)
                y_true_all.extend(y_true)
                y_pred_all.extend(y_pred)
                count+=1
                #print("corr, pval:", corr, pval)
                '''if (corr is not None) and (not math.isnan(corr)):
                    print(corr)
                    corr_sum = corr_sum + corr
                    pval_sum = pval_sum + pval
                    count+=1
                '''
    print("count:", count)
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    conf_mat = confusion_matrix(y_true_all, y_pred_all)
    print(conf_mat)
    evaluation.write_results_xlsx_confmat(config_params, conf_mat, path_to_results, 'imglabel_confmat')
    print(count)
    wb = op.load_workbook(path_to_results)
    sheet = wb['imglabel_confmat']
    sheet.append(['PrecisionBin','PrecisionMicro','PrecisionMacro','RecallBin','RecallMicro','RecallMacro','F1Bin','F1Micro','F1macro','F1wtmacro','Acc','Cohens Kappa','AUC'])
    wb.save(path_to_results)
    per_model_metric = evaluation.aggregate_performance_metrics(config_params, y_true_all, y_pred_all, None)
    print(per_model_metric)
    evaluation.write_results_xlsx(per_model_metric, path_to_results, 'imglabel_confmat')
    
    '''print(corr_sum)
    print(pval_sum)
    
    corr_sum = corr_sum/count
    pval_sum = pval_sum/count 
    print(corr_sum, pval_sum)'''

def run_imagelabel_attwt_match(config_params, model, path_to_model, dataloader_test, df_test, path_to_results):
    path_to_trained_model = path_to_model
    #model1 = test.load_model_for_testing(model, path_to_trained_model)
    #model_output(config_params, model1, dataloader_test, df_test, path_to_results)


meanF1 = [0.52, 0.75, 0.75, 0.75, 0.72, 0.82]
stddev = [0, 0.03, 0.02, 0.01, 0, 0.04]
imglabel_f1_plot(meanF1, stddev)