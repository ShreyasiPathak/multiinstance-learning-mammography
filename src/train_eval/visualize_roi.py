import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
import cv2
import matplotlib.cm as cm
import torchvision

from utilities import utils
from train_eval import test, mask_roi_match, imagelabel_attwt_match

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
            subfigure.set_title("input image: $a_{"+views_names[view_id]+"}"+f"= ${img_attention[0]:.2f}")
            subfigure.axis('off')
        except:
            pass

        # class activation maps
        subfigure = figure.add_subplot(len(views_names), total_num_subplots, view_id*9 + 2)
        subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
        resized_cam_malignant = cv2.resize(saliency_maps[0,0,:,:], (W, H))
        #print(alpha_red)
        #print(resized_cam_malignant)
        #input('wait1')
        subfigure.imshow(resized_cam_malignant, cmap=alpha_red, clim=[0.0, 1.0])
        subfigure.set_title("SM: malignant")
        subfigure.axis('off')
        '''
        subfigure = figure.add_subplot(1, total_num_subplots, 3)
        subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
        resized_cam_benign = cv2.resize(saliency_maps[0,0,:,:], (W, H))
        subfigure.imshow(resized_cam_benign, cmap=alpha_green, clim=[0.0, 1.0])
        subfigure.set_title("SM: benign")
        subfigure.axis('off')
        '''
        
        # patch map
        subfigure = figure.add_subplot(len(views_names), total_num_subplots, view_id*9 + 3)
        subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
        cm.YlGnBu.set_under('w', alpha=0)
        crop_mask = utils.get_crop_mask(
            patch_locations[0, np.arange(parameters["K"]), :],
            parameters["crop_shape"], (H, W),
            "upper_left")
        subfigure.imshow(crop_mask, alpha=0.7, cmap=cm.YlGnBu, clim=[0.9, 1])
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
    
    
    plt.savefig(save_dir, bbox_inches='tight', format="pdf", dpi=500)
    plt.close()

def seg_evaluation(config_params, input_img_case, patch_locations_case, patch_img_case, patch_attentions_case, img_attention_case, views_names, exam_name, seg_eval_metric):    
    iou_any_roi_highestattnwt = []
    iou_any_roi_max = []
    iou_all_roi_mean = []
    view_name_for_cbis = {'LCC': 'LEFT_CC', 'RCC': 'RIGHT_CC', 'LMLO': 'LEFT_MLO', 'RMLO': 'RIGHT_MLO'}
    fig, ax = plt.subplots(1, len(views_names))
    if len(views_names) == 1:
        ax = [ax]

    if config_params['dataset'] == 'vindr':
        df_roi = pd.read_csv('/groups/dso/spathak/vindr/vindr_singleinstance_imgpreprocessing_size_withroiloc.csv', sep=';')
        if config_params['learningtype'] == 'MIL': 
            image_list = os.listdir(config_params['preprocessed_imagepath']+'/'+exam_name)
            roi_folder_df = df_roi[df_roi['study_id']==exam_name]
        elif config_params['learningtype'] == 'SIL': 
            roi_folder_df = df_roi[df_roi['ImageName']==exam_name]

        not_allow = roi_folder_df.empty
    elif config_params['dataset'] == 'cbis-ddsm':
        not_allow = False

    if not not_allow:
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
                elif config_params['dataset'] == 'vindr':
                    image_name = [image_name.split('.')[0].split('_')[1] for image_name in image_list if view_name in image_name]
                    if len(image_name) == 1:
                        image_name = image_name[0]
                    else:
                        input('stop')
            elif config_params['learningtype'] == 'SIL':
                image_name = exam_name
                print("in visualize roi:", image_name)

            if config_params['dataset'] == 'cbis-ddsm':
                iou_view_each_roi, _, iou_highestattnwt_each_roi, fig, ax = mask_roi_match.match_to_mask_images(config_params, input_img, image_name, patch_attentions, patch_locations, seg_eval_metric, view_id, views_names, fig, ax)
            elif config_params['dataset'] == 'vindr':
                iou_view_each_roi, _, iou_highestattnwt_each_roi, fig, ax = mask_roi_match.match_to_mask_images_vindr(config_params, input_img, image_name, patch_attentions, patch_locations, seg_eval_metric, view_id, views_names, fig, ax)
            if iou_view_each_roi!=[]:
                iou_any_roi_max.append(max(iou_view_each_roi)) # max IOU for any of the ROIs
                iou_all_roi_mean = iou_all_roi_mean + iou_view_each_roi #average for all ROIs in one view 
                iou_any_roi_highestattnwt.append(max(iou_highestattnwt_each_roi)) #V, R
    
    '''if iou_any_roi_max!=[]:
        for ax1 in ax:
            if not ax1.get_images(): 
                ax1.set_visible(False)
        plt.savefig(os.path.join(config_params['path_to_output'], "bounding-box", "{0}.png".format(exam_name)), bbox_inches='tight', format="png")
    '''
    plt.close()
    if iou_any_roi_max!=[]:
        iou_any_roi_max = max(iou_any_roi_max) #max IOU over all ROIs
        iou_all_roi_mean = np.mean(np.array(iou_all_roi_mean)) #axis = 0 is across rows; average IOU over each ROI
        iou_any_roi_highestattnwt = max(iou_any_roi_highestattnwt) #max IOU over all ROIs
    return iou_any_roi_max, iou_all_roi_mean, iou_any_roi_highestattnwt

def model_output(config_params, model, dataloader_test, df_test):
    model.eval()
    #print(df_test.loc[df_test['ShortPath']=='Mass-Test_P_00116'])
    #print(df_test['ShortPath'])
    #input('halt')
    eval_mode = True
    iou_sum_any_roi = 0
    iou_sum_all_roi = 0
    iou_sum_any_roi_hattnwt = 0
    df_iou = {}
    dsc_sum_any_roi = 0
    dsc_sum_all_roi = 0
    dsc_sum_any_roi_hattnwt = 0
    df_dsc = {}
    with torch.no_grad():
        for test_idx, test_batch, test_labels, views_names in dataloader_test:
            #if 'Mass-Test_P_00942' in df_test.loc[test_idx.item(),'ShortPath']:
            if '1.2.826.0.1.3680043.2.526.11.40.1608721769240233.1231594.1756724_6002394481' == df_test.loc[test_idx.item(),'ShortPath'].split('/')[-1]:
                test_batch, test_labels = test_batch.to(config_params['device']), test_labels.to(config_params['device'])
                test_labels = test_labels.view(-1)
                print("test idx:", test_idx.item())
                print("test batch:", test_batch.shape)
                if config_params['femodel'] == 'gmic_resnet18':
                    if config_params['learningtype'] == 'SIL':
                        loaded_image = utils.collect_images(config_params, df_test.loc[test_idx.item()])
                        _, _, output_batch_fusion, saliency_map, patch_locations, patch_imgs, patch_attns, _ = model(test_batch, eval_mode) # compute model output, loss and total train loss over one epoch
                        #print(output_batch_fusion.shape)
                        #print(saliency_map.shape)
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
                    #if config_params['attention'] == 'imagewise':
                    #    img_attns = img_attns.view(img_attns.shape[0], -1).data.cpu().numpy()
                    img_attns = imagelabel_attwt_match.extract_img_attn_wts(config_params, img_attns)
                except:
                    pass
                patch_attentions = patch_attns[0, :, :].data.cpu().numpy()
                
                #filename = view_name+'_'+df_test.loc[test_idx.item(), 'ShortPath'].split('/')[-1]
                filename = df_test.loc[test_idx.item(), 'ShortPath'].split('/')[-1]
                if not os.path.exists(os.path.join(config_params['path_to_output'], "visualization")):
                    os.mkdir(os.path.join(config_params['path_to_output'], "visualization"))
                if not os.path.exists(os.path.join(config_params['path_to_output'], "visualization", "malignant-to-malignant")):
                    os.mkdir(os.path.join(config_params['path_to_output'], "visualization", "malignant-to-malignant"))
                save_dir = os.path.join(config_params['path_to_output'], "visualization", "malignant-to-malignant", "{0}.pdf".format(filename))
                #print(save_dir)
                visualize_example(config_params, loaded_image, saliency_maps, [None, None], patch_locations, patch_imgs, patch_attentions, img_attns, save_dir, config_params['gmic_parameters'], views_names, test_labels.item(), test_pred.item())
                input('halt')
                print("exam name:", exam_name)
                iou_any_roi_max, iou_all_roi_mean, iou_any_roi_max_highestattnwt = seg_evaluation(config_params, loaded_image, patch_locations, patch_imgs, patch_attentions, img_attns, views_names, exam_name, 'IOU')
                dsc_any_roi_max, dsc_all_roi_mean, dsc_any_roi_max_highestattnwt = seg_evaluation(config_params, loaded_image, patch_locations, patch_imgs, patch_attentions, img_attns, views_names, exam_name, 'DSC')
                if iou_any_roi_max!=[]:
                    df_iou[exam_name] = [exam_name, iou_any_roi_max, iou_all_roi_mean, iou_any_roi_max_highestattnwt]
                    iou_sum_any_roi = iou_sum_any_roi + iou_any_roi_max
                    iou_sum_all_roi = iou_sum_all_roi + iou_all_roi_mean
                    iou_sum_any_roi_hattnwt = iou_sum_any_roi_hattnwt + iou_any_roi_max_highestattnwt
                print("iou exam any roi iou:", iou_any_roi_max)
                #print("iou exam all roi iou:", iou_all_roi_mean)
                #print("iou exam any roi dsc:", iou_any_roi_max)
                #print("iou exam all roi dsc:", iou_all_roi_mean)

                if dsc_any_roi_max!=[]:
                    df_dsc[exam_name] = [exam_name, dsc_any_roi_max, dsc_all_roi_mean, dsc_any_roi_max_highestattnwt]
                    dsc_sum_any_roi = dsc_sum_any_roi + dsc_any_roi_max
                    dsc_sum_all_roi = dsc_sum_all_roi + dsc_all_roi_mean
                    dsc_sum_any_roi_hattnwt = iou_sum_any_roi_hattnwt + dsc_any_roi_max_highestattnwt
                print("dsc exam any roi:", dsc_any_roi_max)
                #print("dsc exam all roi:", dsc_all_roi_mean)
                #print("dsc exam any roi:", dsc_any_roi_max)
                #print("dsc exam all roi:", dsc_all_roi_mean)
                print(test_labels, test_pred)
                #input('halt')
    df_img_iou = pd.DataFrame.from_dict(df_iou, orient='index', columns=['ImageName', 'iou_any_roi_max', 'iou_all_roi_mean', 'iou_any_roi_max_highestattnwt'])
    df_img_iou.to_csv(os.path.join(config_params['path_to_output'], "iou_score_test_set.csv"), sep=';',na_rep='NULL',index=False)
    iou_avg_any_roi = iou_sum_any_roi/df_img_iou.shape[0]
    iou_avg_all_roi = iou_sum_all_roi/df_img_iou.shape[0]
    iou_avg_any_roi_hattnwt = iou_sum_any_roi_hattnwt/df_img_iou.shape[0]
    print("iou avg any roi:", iou_avg_any_roi)
    print("iou avg all roi:", iou_avg_all_roi)
    print("iou avg any roi hattnwt:", iou_avg_any_roi_hattnwt)

    df_img_dsc = pd.DataFrame.from_dict(df_dsc, orient='index', columns=['ImageName', 'iou_any_roi_max', 'iou_all_roi_mean', 'iou_any_roi_max_highestattnwt'])
    df_img_dsc.to_csv(os.path.join(config_params['path_to_output'], "dsc_score_test_set.csv"), sep=';',na_rep='NULL',index=False)
    dsc_avg_any_roi = dsc_sum_any_roi/df_img_dsc.shape[0]
    dsc_avg_all_roi = dsc_sum_all_roi/df_img_dsc.shape[0]
    dsc_avg_any_roi_hattnwt = dsc_sum_any_roi_hattnwt/df_img_dsc.shape[0]
    print("dsc avg any roi:", dsc_avg_any_roi)
    print("dsc avg all roi:", dsc_avg_all_roi)
    print("dsc avg any roi hattnwt:", dsc_avg_any_roi_hattnwt)

def run_visualization_pipeline(config_params, model, path_to_model, dataloader_test, df_test):
    path_to_trained_model = path_to_model
    model1 = test.load_model_for_testing(model, path_to_trained_model)
    model_output(config_params, model1, dataloader_test, df_test)