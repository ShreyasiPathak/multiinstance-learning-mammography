import os
import cv2
import math
import numpy as np
import pandas as pd
import matplotlib.patches as patches

from utilities import data_augmentation_utils, gmic_utils

def create_mask_file(filepath_roiloc, filepath_newimagesize):
    """
    Function that combines the location coordinates of the cleaned image with the 
    location coordinates of the ROI. This is valid for vindr as vindr dataset has 
    provided a roi location coordinate, unlike, cbis, which has provided a mask image.
    """
    df_roiloc = pd.read_csv(filepath_roiloc).rename(columns={'image_id':'ImageName'}).sort_values(by='ImageName') #20485
    df_newimagesize = pd.read_csv(filepath_newimagesize,sep=';').sort_values(by='ImageName') #19999
    df_merged = df_newimagesize.merge(df_roiloc, on ='ImageName', how = 'left') #20485
    df_nonan = df_merged[['xmin', 'ymin', 'xmax', 'ymax']].dropna()
    df_merged = df_merged.loc[df_nonan.index] #2254
    df_merged.to_csv('/groups/dso/spathak/vindr/'+'vindr_singleinstance_imgpreprocessing_size_withroiloc.csv', sep=';', na_rep='NULL', index=False)

def mask_paths(config_params):
    """
    Function that returns the location of the mask and 
    file path of bounding box location after image cleaning for all datasets 
    """
    if config_params['dataset'] == 'cbis-ddsm':
        #mask_path = '/projects/dso_mammovit/project_kushal/data/roi-images/'
        #image_size_path = '/projects/dso_mammovit/project_kushal/data/cbis-ddsm_singleinstance_imgpreprocessing_size_mod.csv'
        mask_path = '/deepstore/datasets/dmb/medical/breastcancer/mammography/cbis-ddsm/roi-images/'
        image_size_path = '/deepstore/datasets/dmb/medical/breastcancer/mammography/cbis-ddsm/cbis-ddsm_singleinstance_imgpreprocessing_size_mod.csv'
    
    elif config_params['dataset'] == 'vindr':
        mask_path = '/groups/dso/spathak/vindr/physionet.org/files/vindr-mammo/1.0.0/finding_annotations.csv'
        image_size_path = '/groups/dso/spathak/vindr/vindr_singleinstance_imgpreprocessing_size.csv'

    return mask_path, image_size_path

def dice_similarity_score(trueLoc, predLoc):
    """
    Function that calculates IOU given the true ROI location and predicted ROI location
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(trueLoc[0], predLoc[0])
    yA = max(trueLoc[1], predLoc[1])
    xB = min(trueLoc[2], predLoc[2])
    yB = min(trueLoc[3], predLoc[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (trueLoc[2] - trueLoc[0] + 1) * (trueLoc[3] - trueLoc[1] + 1)
    boxBArea = (predLoc[2] - predLoc[0] + 1) * (predLoc[3] - predLoc[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    dsc = 2*interArea / float(boxAArea + boxBArea)
    # return the intersection over union value
    return dsc

def intersection_over_union(trueLoc, predLoc):
    """
    Function that calculates IOU given the true ROI location and predicted ROI location
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(trueLoc[0], predLoc[0])
    yA = max(trueLoc[1], predLoc[1])
    xB = min(trueLoc[2], predLoc[2])
    yB = min(trueLoc[3], predLoc[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (trueLoc[2] - trueLoc[0] + 1) * (trueLoc[3] - trueLoc[1] + 1)
    boxBArea = (predLoc[2] - predLoc[0] + 1) * (predLoc[3] - predLoc[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def bounding_box_true_pred(original_image, true_mask_loc_all, pred_mask_loc_all, view_id, views_names, figure, ax):
    """
    Function that calculates the bounding box around the original ROI mask image
    :true_mask: ROI mask image
    :pred_mask_loc: min_x, min_y, max_x, max_y, where x = width, y = height
    :original_image: pytorch, C, H, W
    """
    original_image = original_image.numpy()[0,:,:]
    ax[view_id].imshow(original_image, aspect='equal', cmap='gray')
    for true_mask_loc in true_mask_loc_all:
        print("True mask loc:", true_mask_loc)
        rect1 = patches.Rectangle(tuple(true_mask_loc[:2]), true_mask_loc[2]-true_mask_loc[0], true_mask_loc[3]-true_mask_loc[1], linewidth=1, edgecolor='g', facecolor='none')
        ax[view_id].add_patch(rect1)
    for pred_mask_loc in pred_mask_loc_all:
        print("Pred mask loc:", pred_mask_loc)
        rect2 = patches.Rectangle(tuple(pred_mask_loc[:2]), pred_mask_loc[2]-pred_mask_loc[0], pred_mask_loc[3]-pred_mask_loc[1], linewidth=1, edgecolor='r', facecolor='none')
        ax[view_id].add_patch(rect2)
    return figure, ax

def extract_patch_position_wrt_image(original_img_pytorch, crop_shape, crop_position, method="center"):
    """
    Function that take a crop on the original image.
    Use PyTorch to do this.
    :param original_img_pytorch: (N,C,H,W) PyTorch Tensor
    :param crop_shape: (h, w) integer tuple
    :param method: supported in ["center", "upper_left"]
    :return: (N, K, h, w) PyTorch Tensor
    (0,0) is at the top left.
    crop_position: h, w
    """
    # retrieve inputs
    _, H, W = original_img_pytorch.shape
    crop_y, crop_x = crop_position 
    y_delta, x_delta = crop_shape

    # locate the four corners
    if method == "center":
        min_y = int(np.round(crop_y - y_delta / 2))
        max_y = int(np.round(crop_y + y_delta / 2))
        min_x = int(np.round(crop_x - x_delta / 2))
        max_x = int(np.round(crop_x + x_delta / 2))
    elif method == "upper_left":
        min_y = int(np.round(crop_y))
        max_y = int(np.round(crop_y + y_delta))
        min_x = int(np.round(crop_x))
        max_x = int(np.round(crop_x + x_delta))

    # make sure that the crops are in range
    min_y = gmic_utils.make_sure_in_range(min_y, 0, H)
    max_y = gmic_utils.make_sure_in_range(max_y, 0, H)
    min_x = gmic_utils.make_sure_in_range(min_x, 0, W)
    max_x = gmic_utils.make_sure_in_range(max_x, 0, W)

    if (max_y - min_y) < y_delta:
        gap = y_delta - (max_y - min_y)
        if (max_y + gap) < H:
            max_y = max_y + gap
        elif (min_y - gap) > 0:
            min_y = min_y - gap
    
    elif (max_x - min_x) < x_delta:
        gap = x_delta - (max_x - min_x)
        if (max_x + gap) < W:
            max_x = max_x + gap
        elif (min_x - gap) > 0:
            min_x = min_x - gap

    patch_position = [min_x, min_y, max_x, max_y] 

    return patch_position

def match_to_mask_images_cbis(config_params, original_image, exam_name, model_patch_attentions, model_patch_locations, seg_eval_metric, view_id, views_names, fig, ax):
    """
    Function to calculate how much does the patch extracted by the model match to the true ROI 
    """
    #Map this exam (image or case) to a name that is a substring of the ROI folder name
    mask_path, image_size_path = mask_paths(config_params)
    if config_params['learningtype'] == 'SIL':
        image_folder_name = "_".join(exam_name.split('_')[:-1])
    else:
        image_folder_name = exam_name 
    print("image_folder_name:", image_folder_name)

    #which ROI folders belong to this exam
    roi_folders = os.listdir(mask_path)
    roi_folder_name = [roi_folder for roi_folder in roi_folders if image_folder_name in roi_folder]
    print("roi folder name:", roi_folder_name)
    
    #Read the bounding box coordinates generated after passing the original image through the cleaning algorithm
    df_img_size = pd.read_csv(image_size_path, sep=';')
    df_img_size = df_img_size[df_img_size['ImageName'].str.split('_').str[:5].str.join('_')==image_folder_name]

    #select the location of the highest attention patch
    #maxattn_patch_location = model_patch_locations[0, np.argmax(model_patch_attentions), :]

    #calculate segmentation evaluation metric over all ROIs belonging to that exam
    iou_all_over_each_roi = []
    iou_max_over_each_roi = []
    iou_highestattnwt_max_over_each_roi = []
    true_mask_loc_all = [] 
    for roi_folder in roi_folder_name:
        #correcting mask size: original to preprocessed (cleaning algo) to resize (2944x1920)
        true_mask_image = cv2.imread(mask_path+'/'+roi_folder+'/'+'1-2.png') 
        true_mask_image = true_mask_image[df_img_size['pro_min_y'].item():df_img_size['pro_max_y'].item(), df_img_size['pro_min_x'].item():df_img_size['pro_max_x'].item()]
        true_mask_image = data_augmentation_utils.myhorizontalflip(true_mask_image, df_img_size['BreastSide'].item())
        true_mask_image = cv2.resize(true_mask_image, dsize=(config_params['resize'][1], config_params['resize'][0]))
        x,y,w,h = cv2.boundingRect(true_mask_image[:,:,0])
        true_mask_loc = [x,y,x+w,y+h]
        true_mask_loc_all.append(true_mask_loc)
        
        iou_over_all_patches = []
        pred_mask_loc_all = []
        for idx in range(model_patch_locations.shape[1]):
            patch_location = model_patch_locations[0, idx, :]
            #extract min_x, min_y, max_x, max_y position from the upper left patch location (extracted by the model)
            pred_mask_loc = extract_patch_position_wrt_image(original_image, config_params['crop_shape'], patch_location)
            if seg_eval_metric=='IOU':
                iou_over_all_patches.append(intersection_over_union(true_mask_loc, pred_mask_loc))
            elif seg_eval_metric=='DSC':
                iou_over_all_patches.append(dice_similarity_score(true_mask_loc, pred_mask_loc))
            pred_mask_loc_all.append(pred_mask_loc)
        
        #max iou over all patches for each roi; [R] where R is number of patches
        iou_max_over_each_roi.append(max(iou_over_all_patches))
        
        #highest attention weighted patch
        iou_highestattnwt_max_over_each_roi.append(iou_over_all_patches[np.argmax(model_patch_attentions)])

        #append iou over all patches for each roi
        iou_all_over_each_roi.append(iou_over_all_patches)
    
    #fig, ax = bounding_box_true_pred(original_image, true_mask_loc_all, pred_mask_loc_all, view_id, views_names, fig, ax)

    #iou_any_roi = max(iou_max_over_each_roi)
    return iou_max_over_each_roi, iou_all_over_each_roi, iou_highestattnwt_max_over_each_roi, fig, ax


def match_to_mask_images_vindr(config_params, original_image, exam_name, model_patch_attentions, model_patch_locations, seg_eval_metric, view_id, views_names, fig, ax):
    #filepath_roiloc, filepath_newimagesize = mask_paths(config_params)
    #create_mask_file(filepath_roiloc, filepath_newimagesize) #20485
    #input('halt')

    #which ROI folders belong to this exam
    print("image_name:", exam_name)
    df_roi = pd.read_csv('/groups/dso/spathak/vindr/vindr_singleinstance_imgpreprocessing_size_withroiloc.csv', sep=';')
    roi_folder_df = df_roi[df_roi['ImageName']==exam_name]#.split('.')[0].split('_')[1]]
    print("roi folder name:", roi_folder_df)
    
    iou_all_over_each_roi = []
    iou_max_over_each_roi = []
    iou_highestattnwt_max_over_each_roi = []
    true_mask_loc_all = [] 
    
    if not roi_folder_df.empty:
        for idx in roi_folder_df.index:
            try:
                roi_row = roi_folder_df.loc[idx]
                #print(roi_row)
                true_mask_image = np.zeros((roi_row['ori_height'], roi_row['ori_width']), dtype=np.uint8)
                mask_white = np.ones((math.ceil(roi_row['ymax']) - math.ceil(roi_row['ymin']), math.ceil(roi_row['xmax']) - math.ceil(roi_row['xmin'])))
                true_mask_image[math.ceil(roi_row['ymin']):math.ceil(roi_row['ymax']), math.ceil(roi_row['xmin']):math.ceil(roi_row['xmax'])] = mask_white
                #plt.imsave('./mask.png', true_mask_image, cmap='gray')
                true_mask_image = true_mask_image * 255
                true_mask_image = true_mask_image[roi_row['pro_min_y'].item():roi_row['pro_max_y'].item(), roi_row['pro_min_x'].item():roi_row['pro_max_x'].item()]
                true_mask_image = data_augmentation_utils.myhorizontalflip(true_mask_image, roi_row['laterality'])
                #plt.imsave('./mask_hf.png', true_mask_image, cmap='gray')
                true_mask_image = cv2.resize(true_mask_image, dsize=(config_params['resize'][1], config_params['resize'][0]))
                #plt.imsave('./mask_rs.png', true_mask_image, cmap='gray')
                #print(true_mask_image.shape)
                #print(true_mask_image.dtype)
                x,y,w,h = cv2.boundingRect(true_mask_image) 
                true_mask_loc = [x,y,x+w,y+h]
                true_mask_loc_all.append(true_mask_loc)
            
                iou_over_all_patches = []
                pred_mask_loc_all = []
                for idx in range(model_patch_locations.shape[1]):
                    patch_location = model_patch_locations[0, idx, :]
                    #extract min_x, min_y, max_x, max_y position from the upper left patch location (extracted by the model)
                    pred_mask_loc = extract_patch_position_wrt_image(original_image, config_params['crop_shape'], patch_location)
                    if seg_eval_metric=='IOU':
                        iou_over_all_patches.append(intersection_over_union(true_mask_loc, pred_mask_loc))
                    elif seg_eval_metric=='DSC':
                        iou_over_all_patches.append(dice_similarity_score(true_mask_loc, pred_mask_loc))
                    pred_mask_loc_all.append(pred_mask_loc)
            
                #max iou over all patches for each roi; [R] where R is number of patches
                iou_max_over_each_roi.append(max(iou_over_all_patches))
                
                #highest attention weighted patch
                iou_highestattnwt_max_over_each_roi.append(iou_over_all_patches[np.argmax(model_patch_attentions)])

                #append iou over all patches for each roi
                iou_all_over_each_roi.append(iou_over_all_patches)
            except:
                pass
    
        #fig, ax = bounding_box_true_pred(original_image, true_mask_loc_all, pred_mask_loc_all, view_id, views_names, fig, ax)
    
    return iou_max_over_each_roi, iou_all_over_each_roi, iou_highestattnwt_max_over_each_roi, fig, ax