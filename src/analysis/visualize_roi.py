import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
import cv2
import pickle
import matplotlib.cm as cm
import torchvision

from utilities import data_loaders_utils, gmic_utils
from train_eval import test
from analysis import mask_roi_match, imagelabel_attwt_match

def visualize_example(config_params, input_img_case, saliency_maps_case, true_segs,
                      patch_locations_case, patch_img_case, patch_attentions_case,
                      img_attention_case, save_dir, parameters, views_names, true_label, pred_label):
    """
    Function that visualizes the saliency maps for an example
    """

    groundtruth_dic={0:'benign', 1:'malignant'}

    # set up colormaps for benign and malignant
    alphas = np.abs(np.linspace(0, 0.95, 259))
    alpha_green = plt.cm.get_cmap('Greens')
    alpha_green._init()
    alpha_green._lut[:, -1] = alphas
    alpha_red = plt.cm.get_cmap('Reds')
    alpha_red._init()
    alpha_red._lut[:, -1] = alphas

    # create visualization template
    total_num_subplots = 3 + parameters["K"]
    figure = plt.figure(figsize=(30, 12))
    figure.suptitle('Groundtruth:' + groundtruth_dic[true_label]+', Predicted:' + groundtruth_dic[pred_label], fontsize=14, y = 0.94)
    
    for view_id, view_name in enumerate(views_names):
        try:
            input_img = input_img_case[view_id][np.newaxis,:,:,:]
        except:
            input_img = input_img_case[view_id][np.newaxis, np.newaxis,:,:]
        
        input_img = torchvision.transforms.Resize((config_params['resize'][0], config_params['resize'][1]))(input_img)
        saliency_maps = saliency_maps_case[:,view_id,:,:,:]
        patch_locations = patch_locations_case[:,view_id,:,:]
        patch_img = patch_img_case[:,view_id,:,:,:]
        patch_attentions = patch_attentions_case[view_id,:]
        try:
            img_attention = img_attention_case[:, view_id]
        except:
            pass
        
        # colormap lists
        _, _, h, w = saliency_maps.shape
        _, _, H, W = input_img.shape

        # input image + segmentation map
        subfigure = figure.add_subplot(len(views_names), total_num_subplots, view_id*9 + 1)
        subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
        benign_seg, malignant_seg = true_segs
        if benign_seg is not None:
            cm.Greens.set_under('w', alpha=0)
            subfigure.imshow(benign_seg, alpha=0.85, cmap=cm.Greens, clim=[0.9, 1])
        if malignant_seg is not None:
            cm.OrRd.set_under('w', alpha=0)
            subfigure.imshow(malignant_seg, alpha=0.85, cmap=cm.OrRd, clim=[0.9, 1])
        try:
            # uncomment the below line for visualization of attention pooling based models
            subfigure.set_title(views_names[view_id][0]+'-'+views_names[view_id][1:]+ ", $a_{\mathrm{"+views_names[view_id][0]+'-'+views_names[view_id][1:]+"}}"+f"= ${img_attention[0]:.2f}")
            #uncomment the below line for visualizetion of mean pooling models
            #subfigure.set_title(views_names[view_id][0]+'-'+views_names[view_id][1:])
            subfigure.axis('off')
        except:
            pass

        # class activation maps
        subfigure = figure.add_subplot(len(views_names), total_num_subplots, view_id*9 + 2)
        subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
        resized_cam_malignant = cv2.resize(saliency_maps[0,0,:,:], (W, H))
        subfigure.imshow(resized_cam_malignant, cmap=alpha_red, clim=[0.0, 1.0])
        subfigure.set_title("Saliency Map")
        subfigure.axis('off')
        
        # patch map
        subfigure = figure.add_subplot(len(views_names), total_num_subplots, view_id*9 + 3)
        subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
        #cm.YlGnBu.set_under('w', alpha=0)
        cm.spring.set_under('w', alpha=0)
        crop_mask = gmic_utils.get_crop_mask(
            patch_locations[0, np.arange(parameters["K"]), :],
            parameters["crop_shape"], (H, W),
            "upper_left")
        #subfigure.imshow(crop_mask, alpha=0.7, cmap=cm.YlGnBu, clim=[0.9, 1])
        subfigure.imshow(crop_mask, alpha=0.7, cmap=cm.spring, clim=[0.9, 1])
        if benign_seg is not None:
            cm.Greens.set_under('w', alpha=0)
            subfigure.imshow(benign_seg, alpha=0.85, cmap=cm.Greens, clim=[0.9, 1])
        if malignant_seg is not None:
            cm.OrRd.set_under('w', alpha=0)
            subfigure.imshow(malignant_seg, alpha=0.85, cmap=cm.OrRd, clim=[0.9, 1])
        subfigure.set_title("patch map")
        subfigure.axis('off')
        
        # crops
        for crop_idx in range(parameters["K"]):
            subfigure = figure.add_subplot(len(views_names), total_num_subplots, view_id*9 + 4 + crop_idx)
            subfigure.imshow(patch_img[0, crop_idx, :, :], cmap='gray', alpha=0.8, interpolation='nearest',
                            aspect='equal')
            subfigure.axis('off')
            # crops_attn can be None when we only need the left branch + visualization
            subfigure.set_title("$\\alpha_{0} = ${1:.2f}".format(crop_idx, patch_attentions[crop_idx]))
    
    print(save_dir)
    plt.savefig(save_dir, bbox_inches='tight', format="pdf", dpi=500)
    plt.close()

def seg_evaluation(config_params, input_img_case, img_path_list, patch_locations_case, patch_img_case, patch_attentions_case, img_attention_case, views_names, exam_name, df_score, seg_eval_metric, pred_label, true_label, roi_diagnosis_conf_mat_all):    
    iou_any_roi_highestattnwt = []
    iou_any_roi_max = []
    iou_all_roi_mean = []
    fig, ax = plt.subplots(1, len(views_names))
    if len(views_names) == 1:
        ax = [ax]

    #filtering rows with empty rois
    if config_params['dataset'] == 'vindr':
        df_roi = pd.read_csv('/deepstore/datasets/dmb/medical/breastcancer/mammography/vindr/vindr_singleinstance_imgpreprocessing_size_withroiloc.csv', sep=';')
        #print(df_roi[df_roi['split']=='test']['ImageName'].unique().shape) #357
        #df_roi = pd.read_csv('/groups/dso/spathak/vindr/vindr_singleinstance_imgpreprocessing_size_withroiloc.csv', sep=';')
        if config_params['learningtype'] == 'MIL': 
            #image_list = os.listdir(config_params['preprocessed_imagepath']+'/'+exam_name)
            roi_folder_df = df_roi[df_roi['study_id']==exam_name]
        elif config_params['learningtype'] == 'SIL': 
            roi_folder_df = df_roi[df_roi['ImageName']==exam_name]
        #print("Exam name:", exam_name)
        #print("df roi matched:", roi_folder_df)
        not_allow = roi_folder_df.empty
        #print("not allow", not_allow)
    
    elif config_params['dataset'] == 'cbis-ddsm':
        view_name_for_cbis = {'LCC': 'LEFT_CC', 'RCC': 'RIGHT_CC', 'LMLO': 'LEFT_MLO', 'RMLO': 'RIGHT_MLO'}
        not_allow = False
    
    elif config_params['dataset'] == 'zgt':
        #zgt_roi_annotation = '/home/pathaks/PhD/case-level-breast-cancer/multiview_mammogram/input_data/extracted_annotations_withrle_corrected.csv'
        zgt_roi_annotation = '/homes/spathak/multiview_mammogram/input_data/extracted_annotations_withrle_corrected.csv'
        df_roi_zgt = pd.read_csv(zgt_roi_annotation, sep=';')
        #print("exam name:", exam_name)
        #print(df_roi_zgt[df_roi_zgt['image'].str.split('/').str[-1]==exam_name])
        #print(df_roi_zgt['image'].str.split('/').str[-1])
        if config_params['learningtype'] == 'MIL':
            not_allow = df_roi_zgt[df_roi_zgt['image'].str.split('/').str[-3]==exam_name]['rle_val'].isnull().all()
        elif config_params['learningtype'] == 'SIL':
            not_allow = df_roi_zgt[df_roi_zgt['image'].str.split('/').str[-1]==exam_name]['rle_val'].isnull().all()
        print(not_allow)

    if not not_allow:
        roi_diagnosis_conf_mat_percase = np.zeros((2,4))
        flag = 0
        for view_id, view_name in enumerate(views_names):
            input_img = input_img_case[view_id]
            input_img = torchvision.transforms.Resize((config_params['resize'][0], config_params['resize'][1]))(input_img)
            patch_locations = patch_locations_case[:,view_id,:,:]
            patch_img = patch_img_case[:,view_id,:,:,:]
            patch_attentions = patch_attentions_case[view_id,:]
            try:
                img_attention = img_attention_case[:, view_id]
            except:
                pass
            
            if config_params['learningtype'] == 'MIL':
                if config_params['dataset'] == 'cbis-ddsm':
                    image_name = exam_name + '_' + view_name_for_cbis[view_name]
                    print("image_name from the for loop of view names:", image_name)
                    accnum = 'NULL'
                    case_name = exam_name 
                    print("case name:", case_name)
                    #roi_folder_df.loc[roi_folder_df['ImageName'] == img_path_list[view_id].split('/')[-1].split('.png')[0].split('_')[-1], 'study_id']
                    series_name = 'NULL'
                    print(img_path_list[view_id])
                    print("series name:", series_name)
                    img_name = img_path_list[view_id].split('/')[-1].split('_1-1.png')[0]
                    print("image name from the case:", img_name) 
                    assert image_name == img_name
                
                elif config_params['dataset'] == 'vindr':
                    print("img path list:", img_path_list[view_id])
                    image_name = img_path_list[view_id].split('/')[-1].split('.')[0].split('_')[1]
                    #print(roi_folder_df.loc[roi_folder_df['ImageName'] == image_name])
                    if roi_folder_df.loc[roi_folder_df['ImageName'] == image_name].empty:
                        continue
                    #image_name = [image_name.split('.')[0].split('_')[1] for image_name in image_list if view_name in image_name]
                    print('image_name', image_name)
                    #if len(image_name) == 1:
                    #    image_name = image_name[0]
                    #else:
                    #    input('stop')
                    accnum = 'NULL'
                    case_name = exam_name 
                    print("case name:", case_name)
                    #roi_folder_df.loc[roi_folder_df['ImageName'] == img_path_list[view_id].split('/')[-1].split('.png')[0].split('_')[-1], 'study_id']
                    series_name = roi_folder_df.loc[roi_folder_df['ImageName'] == image_name, 'series_id']
                    #print(series_name)
                    if len(series_name):
                        series_name = series_name.unique().item()
                    print("series name:", series_name)
                    img_name = img_path_list[view_id].split('/')[-1]  
                
                elif config_params['dataset'] == 'zgt':
                    image_name = img_path_list[view_id] #complete path of the image
                    accnum = image_name.split('/')[-3].split('_')[-1]
                    case_name = image_name.split('/')[-3]
                    series_name = image_name.split('/')[-2]
                    img_name = image_name.split('/')[-1]
                    #print('image name:', image_name)
                    #print(view_name)
                    #print(img_path_list[view_id])
                    assert img_path_list[view_id].split('/')[-2].split('_')[0].replace(' ','') == view_name
                
            elif config_params['learningtype'] == 'SIL':
                if config_params['dataset'] == 'zgt':
                    image_name = img_path_list # complete path of the image
                    #print(image_name, flush=True)
                    #print(img_path_list)
                    accnum = img_path_list.split('/')[-3].split('_')[-1]
                    case_name = img_path_list.split('/')[-3]
                    series_name = img_path_list.split('/')[-2]
                    img_name = img_path_list.split('/')[-1]
                elif config_params['dataset'] == 'vindr':
                    image_name = exam_name 
                    print("exam name:",exam_name)
                    accnum = 'NULL'
                    case_name = img_path_list.split('/')[-2]
                    print("case name:", case_name)
                    series_name = roi_folder_df.loc[roi_folder_df['ImageName'] == image_name, 'series_id']
                    if len(series_name):
                        series_name = series_name.unique().item()
                    print("series name:", series_name)
                    img_name = img_path_list.split('/')[-1].split('.')[0].split('_')[1]
                elif config_params['dataset'] == 'cbis-ddsm':
                    image_name = exam_name
                    print("image_name:", image_name)
                    print("image path:", img_path_list)
                    accnum = 'NULL'
                    case_name = img_path_list.split('/')[-3] 
                    print("case name:", case_name)
                    #roi_folder_df.loc[roi_folder_df['ImageName'] == img_path_list[view_id].split('/')[-1].split('.png')[0].split('_')[-1], 'study_id']
                    series_name = img_path_list.split('/')[-2] 
                    
                    img_name = img_path_list.split('/')[-1]
                #print("in visualize roi:", image_name)

            if config_params['dataset'] == 'cbis-ddsm':
                iou_view_each_roi, _, iou_highestattnwt_each_roi, fig, ax = mask_roi_match.match_to_mask_images_cbis(config_params, input_img, image_name, patch_attentions, patch_locations, seg_eval_metric, view_id, views_names, fig, ax)
            elif config_params['dataset'] == 'vindr':
                #iou_view_each_roi, _, iou_highestattnwt_each_roi, fig, ax = mask_roi_match.match_to_mask_images_vindr(config_params, input_img, image_name, patch_attentions, patch_locations, seg_eval_metric, view_id, views_names, fig, ax)
                iou_view_each_roi, _, iou_highestattnwt_each_roi, fig, ax, roi_diagnosis_conf_mat_percase = mask_roi_match.roi_diagnosis_images_vindr(config_params, input_img, image_name, patch_attentions, patch_locations, seg_eval_metric, view_id, views_names, fig, ax, pred_label, true_label, roi_diagnosis_conf_mat_percase)
            elif config_params['dataset'] == 'zgt':
                iou_view_each_roi, _, iou_highestattnwt_each_roi, fig, ax = mask_roi_match.match_to_mask_images_zgt(config_params, input_img, image_name, patch_attentions, patch_locations, seg_eval_metric, view_id, views_names, fig, ax, img_attention)
            if iou_view_each_roi!=[]:
                #iou_any_roi_max.append(max(iou_view_each_roi)) # max IOU for any of the ROIs
                #iou_all_roi_mean = iou_all_roi_mean + iou_view_each_roi #average for all ROIs in one view 
                #iou_any_roi_highestattnwt.append(max(iou_highestattnwt_each_roi)) #V, R
                print("max score:", max(iou_view_each_roi))
                df_score[image_name]=[accnum, case_name, series_name, img_name, iou_view_each_roi, max(iou_view_each_roi), np.mean(iou_view_each_roi), max(iou_highestattnwt_each_roi)]
        
        for u in range(roi_diagnosis_conf_mat_percase.shape[0]):
            for v in range(roi_diagnosis_conf_mat_percase.shape[1]):
                if roi_diagnosis_conf_mat_percase[u,v] >= 1:
                    roi_diagnosis_conf_mat_all[u,v]+=1
                    flag = 1
        if flag == 0:
            if true_label == 0 and pred_label==0:
                roi_diagnosis_conf_mat_all[0,2]+=1  
            elif true_label == 0 and pred_label==1:
                roi_diagnosis_conf_mat_all[0,3]+=1   
            elif true_label == 1 and pred_label==0:
                roi_diagnosis_conf_mat_all[1,2]+=1  
            elif true_label == 1 and pred_label==1:
                roi_diagnosis_conf_mat_all[1,3]+=1  

                
    #bounding box 
    '''if iou_any_roi_max!=[]:
        for ax1 in ax:
            if not ax1.get_images(): 
                ax1.set_visible(False)
        plt.savefig(os.path.join(config_params['path_to_output'], "bounding-box", "{0}.png".format(exam_name)), bbox_inches='tight', format="png")
    '''
    plt.close()
    '''if iou_any_roi_max!=[]:
        iou_any_roi_max = max(iou_any_roi_max) #max IOU over all ROIs
        iou_all_roi_mean = np.mean(np.array(iou_all_roi_mean)) #axis = 0 is across rows; average IOU over each ROI
        iou_any_roi_highestattnwt = max(iou_any_roi_highestattnwt) #max IOU over all ROIs
    return iou_any_roi_max, iou_all_roi_mean, iou_any_roi_highestattnwt'''
    return df_score, roi_diagnosis_conf_mat_all


def modelpatch_roi_match(config_params, model, dataloader_test, df_test):
    model.eval()
    eval_mode = True
    iou_sum_any_roi = 0
    iou_sum_all_roi = 0
    iou_sum_any_roi_hattnwt = 0
    df_iou = {}
    dsc_sum_any_roi = 0
    dsc_sum_all_roi = 0
    dsc_sum_any_roi_hattnwt = 0
    df_dsc = {}
    dic_pred = {}
    roi_diagnosis_conf_mat_iou_all = np.zeros((2,4))
    roi_diagnosis_conf_mat_dsc_all = np.zeros((2,4))

    #if config_params['dataset'] == 'zgt':
    #    zgt_roi_annotation = '/homes/spathak/multiview_mammogram/input_data/extracted_annotations_withrle_corrected.csv'
    #    df_roi_zgt = pd.read_csv(zgt_roi_annotation, sep=';')
    #    print(list(df_roi_zgt['AccessionNum'].unique())) 
    # #245 is the number of images, but one is removed due to blank binary mask. So, the resulting numbr is 244.
    
    '''if config_params['dataset'] == 'cbis-ddsm':
        print(df_test['FolderName'])
        mask_path = '/deepstore/datasets/dmb/medical/breastcancer/mammography/cbis-ddsm/roi-images/'
        roi_folders = os.listdir(mask_path)
        print(roi_folders)
        count=0
        filtered_list = []
        for roi_folder in roi_folders:
            df_filter = df_test[df_test['FolderName']=='_'.join(roi_folder.split('_')[:3])]
            if not df_filter.empty:
                count+=1
                if roi_folder.split('_')[:5] not in filtered_list:
                    filtered_list.append(roi_folder.split('_')[:5])
        print(count)
        print(len(filtered_list)) #455
        input('halt')
    '''

    with torch.no_grad():
        for test_idx, test_batch, test_labels, views_names in dataloader_test:
            test_batch, test_labels = test_batch.to(config_params['device']), test_labels.to(config_params['device'])
            test_labels = test_labels.view(-1)
            #print("test idx:", test_idx.item(), flush=True)
            #print("test batch:", test_batch.shape, flush=True)
            #print("Accession Num:", df_test.loc[test_idx.item(), 'AccessionNum'], flush=True)
            #if df_test.loc[test_idx.item(), 'AccessionNum'] not in list(df_roi_zgt['AccessionNum'].unique()):
                #print(type(df_test.loc[test_idx.item(), 'AccessionNum']))
                #print(type(list(df_roi_zgt['AccessionNum'].unique())[0]))
            #    continue
            if config_params['femodel'] == 'gmic_resnet18':
                print("test idx:", test_idx.item(), flush=True)
                print("test batch:", test_batch.shape, flush=True)
                
                if config_params['learningtype'] == 'SIL':
                    loaded_image = data_loaders_utils.collect_images(config_params, df_test.loc[test_idx.item()])
                    _, _, output_batch_fusion, saliency_map, patch_locations, patch_imgs, patch_attns, _ = model(test_batch, eval_mode) # compute model output, loss and total train loss over one epoch
                    loaded_image = loaded_image[np.newaxis,:,:,:]
                    patch_locations = patch_locations[:,np.newaxis,:]
                    patch_imgs = patch_imgs[:,np.newaxis,:]
                    patch_attns = patch_attns[:,np.newaxis,:]
                    saliency_map = saliency_map[:, np.newaxis, :, :, :]
                    img_attns = None
                    if config_params['dataset'] == 'cbis-ddsm':
                        exam_name = df_test.loc[test_idx.item(), 'ImageName']
                        img_path_list = df_test.loc[test_idx.item(), 'FullPath']
                    elif config_params['dataset'] == 'vindr':
                        exam_name = df_test.loc[test_idx.item(), 'ImageName']
                        img_path_list = df_test.loc[test_idx.item(), 'FullPath']
                        #print("exam name:", exam_name)
                    elif config_params['dataset'] == 'zgt':
                        print("Accession Num:", df_test.loc[test_idx.item(), 'AccessionNum'], flush=True)
                        img_path_list = df_test.loc[test_idx.item(), 'FullPath']
                        exam_name = df_test.loc[test_idx.item(), 'ShortPath'].split('/')[-1]


                elif config_params['learningtype'] == 'MIL':
                    if config_params['dataset'] == 'cbis-ddsm':
                        exam_name = df_test.loc[test_idx.item(), 'FolderName']
                    elif config_params['dataset'] == 'vindr':
                        exam_name = df_test.loc[test_idx.item(), 'StudyInstanceUID']
                    elif config_params['dataset'] == 'zgt':
                        exam_name = df_test.loc[test_idx.item(), 'ShortPath'].split('/')[-1]
                    loaded_image, _, _, img_path_list = data_loaders_utils.collect_cases(config_params, df_test.loc[test_idx.item()])
                    _, _, output_batch_fusion, saliency_map, patch_locations, patch_imgs, patch_attns, img_attns, _, _ = model(test_batch, views_names, eval_mode)
                    patch_locations = patch_locations.cpu()
                    patch_imgs = patch_imgs.cpu()
            
            #print(img_path_list)
            test_pred = torch.ge(torch.sigmoid(output_batch_fusion), torch.tensor(0.5)).float()
            print(test_labels)
            print(test_pred)
            #dic_pred[exam_name] = [test_labels.item(), test_pred.item()]
            
            #if test_labels.item() == 0 and test_pred.item() == 1:
            #    print("Accession Num:", df_test.loc[test_idx.item(), 'AccessionNum'], flush=True)
            #    input('halt')

            saliency_maps = saliency_map.cpu().numpy()
            try:
                img_attns = imagelabel_attwt_match.extract_img_attn_wts(config_params, img_attns)
            except:
                pass
            print("img attns:", img_attns)
            patch_attentions = patch_attns[0, :, :].data.cpu().numpy()
        
            #for calculating the intersection over union and dice similarity score of ROI candidates with the groundtruth ROI.
            #iou_any_roi_max, iou_all_roi_mean, iou_any_roi_max_highestattnwt = seg_evaluation(config_params, loaded_image, img_path_list, patch_locations, patch_imgs, patch_attentions, img_attns, views_names, exam_name, 'IOU')
            #dsc_any_roi_max, dsc_all_roi_mean, dsc_any_roi_max_highestattnwt = seg_evaluation(config_params, loaded_image, img_path_list, patch_locations, patch_imgs, patch_attentions, img_attns, views_names, exam_name, 'DSC')
            df_iou, roi_diagnosis_conf_mat_iou_all = seg_evaluation(config_params, loaded_image, img_path_list, patch_locations, patch_imgs, patch_attentions, img_attns, views_names, exam_name, df_iou, 'IOU', int(test_pred.item()), int(test_labels.item()), roi_diagnosis_conf_mat_iou_all)
            df_dsc, roi_diagnosis_conf_mat_dsc_all = seg_evaluation(config_params, loaded_image, img_path_list, patch_locations, patch_imgs, patch_attentions, img_attns, views_names, exam_name, df_dsc,'DSC', int(test_pred.item()), int(test_labels.item()), roi_diagnosis_conf_mat_dsc_all)
            
            '''if iou_any_roi_max!=[]:
                df_iou[exam_name] = [exam_name, iou_any_roi_max, iou_all_roi_mean, iou_any_roi_max_highestattnwt]
                iou_sum_any_roi = iou_sum_any_roi + iou_any_roi_max
                iou_sum_all_roi = iou_sum_all_roi + iou_all_roi_mean
                iou_sum_any_roi_hattnwt = iou_sum_any_roi_hattnwt + iou_any_roi_max_highestattnwt
            print("iou exam any roi iou:", iou_any_roi_max)
            print("iou exam all roi iou:", iou_all_roi_mean)
            print("iou exam any roi dsc:", iou_any_roi_max)
            print("iou exam all roi dsc:", iou_all_roi_mean)'''

            '''if dsc_any_roi_max!=[]:
                df_dsc[exam_name] = [exam_name, dsc_any_roi_max, dsc_all_roi_mean, dsc_any_roi_max_highestattnwt]
                dsc_sum_any_roi = dsc_sum_any_roi + dsc_any_roi_max
                dsc_sum_all_roi = dsc_sum_all_roi + dsc_all_roi_mean
                dsc_sum_any_roi_hattnwt = iou_sum_any_roi_hattnwt + dsc_any_roi_max_highestattnwt
            print("dsc exam any roi:", dsc_any_roi_max)
            #print("dsc exam all roi:", dsc_all_roi_mean)
            #print("dsc exam any roi:", dsc_any_roi_max)
            #print("dsc exam all roi:", dsc_all_roi_mean)'''
            print(test_labels, test_pred)
            #input('halt')
    
    #dbfile = open(os.path.join(config_params['path_to_output'], "label_pred_"+str(config_params['randseedother']) +'_'+ str(config_params['randseeddata'])+".pkl"), 'ab')
    #pickle.dump(dic_pred, dbfile)
    #dbfile.close()
    print("roi diagnosis conf mat iou:")
    print(roi_diagnosis_conf_mat_iou_all)
    print("roi diagnosis conf mat dsc:")
    print(roi_diagnosis_conf_mat_dsc_all)
    '''
    df_iou_dsc = dict()
    for key in df_iou.keys():
        df_iou_dsc[key] = df_iou[key] + df_dsc[key][4:8]

    #df_img_iou = pd.DataFrame.from_dict(df_iou, orient='index', columns=['AccessionNum', 'CaseName', 'SeriesName', 'ImageName', 'iou_any_roi', 'iou_any_roi_max', 'iou_all_roi_mean', 'iou_any_roi_max_highestattnwt'])
    df_img_iou_dsc = pd.DataFrame.from_dict(df_iou_dsc, orient='index', columns=['AccessionNum', 'CaseName', 'SeriesName', 'ImageName', 'iou_any_roi', 'iou_any_roi_max', 'iou_all_roi_mean', 'iou_any_roi_max_highestattnwt', 'dsc_any_roi', 'dsc_any_roi_max', 'dsc_all_roi_mean', 'dsc_any_roi_max_highestattnwt'])
    mean_index = len(df_img_iou_dsc.index)
    mean_1 = round(df_img_iou_dsc['iou_any_roi_max'].mean(), 2)
    mean_2 = round(df_img_iou_dsc['iou_all_roi_mean'].mean(), 2)
    mean_3 = round(df_img_iou_dsc['iou_any_roi_max_highestattnwt'].mean(), 2)
    mean_4 = round(df_img_iou_dsc['dsc_any_roi_max'].mean(), 2)
    mean_5 = round(df_img_iou_dsc['dsc_all_roi_mean'].mean(), 2)
    mean_6 = round(df_img_iou_dsc['dsc_any_roi_max_highestattnwt'].mean(), 2)

    df_img_iou_dsc.loc[mean_index, 'iou_any_roi_max'] = mean_1
    df_img_iou_dsc.loc[mean_index, 'iou_all_roi_mean'] = mean_2
    df_img_iou_dsc.loc[mean_index, 'iou_any_roi_max_highestattnwt'] = mean_3
    df_img_iou_dsc.loc[mean_index, 'dsc_any_roi_max'] = mean_4
    df_img_iou_dsc.loc[mean_index, 'dsc_all_roi_mean'] = mean_5
    df_img_iou_dsc.loc[mean_index, 'dsc_any_roi_max_highestattnwt'] = mean_6

    df_img_iou_dsc.to_csv(os.path.join(config_params['path_to_output'], "iou_dsc_score_test_set_"+str(config_params['randseedother']) +'_'+ str(config_params['randseeddata'])+".csv"), sep=';',na_rep='NULL',index=False)
    '''
    '''iou_avg_any_roi = iou_sum_any_roi/df_img_iou.shape[0]
    iou_avg_all_roi = iou_sum_all_roi/df_img_iou.shape[0]
    iou_avg_any_roi_hattnwt = iou_sum_any_roi_hattnwt/df_img_iou.shape[0]
    print("iou avg any roi:", iou_avg_any_roi)
    print("iou avg all roi:", iou_avg_all_roi)
    print("iou avg any roi hattnwt:", iou_avg_any_roi_hattnwt)'''

    #df_img_dsc = pd.DataFrame.from_dict(df_dsc, orient='index', columns=['AccessionNum', 'CaseName', 'SeriesName', 'ImageName', 'iou_any_roi', 'iou_any_roi_max', 'iou_all_roi_mean', 'iou_any_roi_max_highestattnwt'])
    #df_img_dsc.to_csv(os.path.join(config_params['path_to_output'], "dsc_score_test_set_"+str(config_params['randseedother']) +'_'+ str(config_params['randseeddata'])+".csv"), sep=';',na_rep='NULL',index=False)
    '''dsc_avg_any_roi = dsc_sum_any_roi/df_img_dsc.shape[0]
    dsc_avg_all_roi = dsc_sum_all_roi/df_img_dsc.shape[0]
    dsc_avg_any_roi_hattnwt = dsc_sum_any_roi_hattnwt/df_img_dsc.shape[0]
    print("dsc avg any roi:", dsc_avg_any_roi)
    print("dsc avg all roi:", dsc_avg_all_roi)
    print("dsc avg any roi hattnwt:", dsc_avg_any_roi_hattnwt)'''
    
def output_visualize(config_params, model, dataloader_test, df_test):
    model.eval()
    eval_mode = True
    if config_params['dataset'] == 'zgt':
        zgt_roi_annotation = '/homes/spathak/multiview_mammogram/input_data/extracted_annotations_withrle_corrected.csv'
        df_roi_zgt = pd.read_csv(zgt_roi_annotation, sep=';')
    with torch.no_grad():
        for test_idx, test_batch, test_labels, views_names in dataloader_test:
            test_batch, test_labels = test_batch.to(config_params['device']), test_labels.to(config_params['device'])
            test_labels = test_labels.view(-1)
            print("test idx:", test_idx.item(), flush=True)
            print("test batch:", test_batch.shape, flush=True)
            print("Accession Num:", df_test.loc[test_idx.item(), 'AccessionNum'], flush=True)
            if config_params['dataset'] == 'zgt':
                not_allow = df_roi_zgt[df_roi_zgt['AccessionNum']==df_test.loc[test_idx.item(), 'AccessionNum']]['rle_val'].isnull().all()
                if not_allow:
                    continue
            #if df_test.loc[test_idx.item(), 'AccessionNum'] == 6003332157: #6003020666: #6002394481: #6003020666: #6002466686: #6003145061: #6002626200:
            #    print("I am in visualize", flush=True)
            if config_params['femodel'] == 'gmic_resnet18':
                if config_params['learningtype'] == 'SIL':
                    loaded_image = data_loaders_utils.collect_images(config_params, df_test.loc[test_idx.item()])
                    _, _, output_batch_fusion, saliency_map, patch_locations, patch_imgs, patch_attns, _ = model(test_batch, eval_mode) # compute model output, loss and total train loss over one epoch
                    #print(output_batch_fusion.shape)
                    #print(saliency_map.shape)
                    loaded_image = loaded_image[np.newaxis,:,:,:]
                    patch_locations = patch_locations[:,np.newaxis,:]
                    patch_imgs = patch_imgs[:,np.newaxis,:]
                    patch_attns = patch_attns[:,np.newaxis,:]
                    saliency_map = saliency_map[:, np.newaxis, :, :, :]
                    img_attns = None
                    if config_params['dataset'] == 'cbis-ddsm':
                        exam_name = df_test.loc[test_idx.item(), 'FolderName']
                    elif config_params['dataset'] == 'vindr':
                        exam_name = df_test.loc[test_idx.item(), 'StudyInstanceUID']
                    elif config_params['dataset'] == 'zgt':
                        exam_name = df_test.loc[test_idx.item(), 'StudyInstanceUID']
                    #exam_name = df_test.loc[test_idx.item(), 'ImageName']

                elif config_params['learningtype'] == 'MIL':
                    if config_params['dataset'] == 'cbis-ddsm':
                        exam_name = df_test.loc[test_idx.item(), 'FolderName']
                    elif config_params['dataset'] == 'vindr':
                        exam_name = df_test.loc[test_idx.item(), 'StudyInstanceUID']
                    elif config_params['dataset'] == 'zgt':
                        exam_name = df_test.loc[test_idx.item(), 'StudyInstanceUID']
                    loaded_image, _, _, _ = data_loaders_utils.collect_cases(config_params, df_test.loc[test_idx.item()])
                    _, _, output_batch_fusion, saliency_map, patch_locations, patch_imgs, patch_attns, img_attns, _, _ = model(test_batch, views_names, eval_mode)
                    patch_locations = patch_locations.cpu()
                    patch_imgs = patch_imgs.cpu()

            test_pred = torch.ge(torch.sigmoid(output_batch_fusion), torch.tensor(0.5)).float()
            #print(loaded_image)
            #if test_labels.item() == test_pred.item() == 1:
            saliency_maps = saliency_map.cpu().numpy()
            #print(saliency_maps[0,0,0,:,:])
            #print(saliency_maps.shape) #(1, 4, 1, 92, 60)
            #print(patch_locations.shape) #1, 4, 6, 2
            #print(patch_imgs.shape) #1, 4, 6, 256, 256
            #print(patch_attns.shape) #1, 4, 6
            #print("loaded image:", loaded_image[0].shape)
            #print(loaded_image[1].shape)
            #print(img_attns.shape) #1,4,1
            #input('wait')
            try:
                img_attns = imagelabel_attwt_match.extract_img_attn_wts(config_params, img_attns)
            except:
                pass
            patch_attentions = patch_attns[0, :, :].data.cpu().numpy()
            
            #filename = view_name+'_'+df_test.loc[test_idx.item(), 'ShortPath'].split('/')[-1]
            filename = df_test.loc[test_idx.item(), 'ShortPath'].split('/')[-1]
            if not os.path.exists(os.path.join(config_params['path_to_output'], "visualization")):
                os.mkdir(os.path.join(config_params['path_to_output'], "visualization"))
            if not os.path.exists(os.path.join(config_params['path_to_output'], "visualization", str(test_labels.item())+'-'+str(test_pred.item()))):
                os.mkdir(os.path.join(config_params['path_to_output'], "visualization", str(test_labels.item())+'-'+str(test_pred.item())))
            save_dir = os.path.join(config_params['path_to_output'], "visualization", str(test_labels.item())+'-'+str(test_pred.item()), "{0}.pdf".format(filename))
            
            #for visualization the case and ROI candidates; comment the line below if you don't want to visualize it.
            visualize_example(config_params, loaded_image, saliency_maps, [None, None], patch_locations, patch_imgs, patch_attentions, img_attns, save_dir, config_params['gmic_parameters'], views_names, test_labels.item(), test_pred.item())
            #input('halt')
            #print("exam name:", exam_name)
            
def run_visualization_pipeline(config_params, model, path_to_model, dataloader_test, df_test):
    path_to_trained_model = path_to_model
    model1 = test.load_model_for_testing(model, path_to_trained_model)
    #output_visualize(config_params, model1, dataloader_test, df_test)
    modelpatch_roi_match(config_params, model, dataloader_test, df_test)
