import os
import csv
import glob
import shutil
import pandas as pd
import numpy as np

file_creation = 'MIL' #or SIL

#------------------------------------------------------------------MIL csv file creation----------------------------------------------#
def creating_MIL_folder_structure(input_foldername1, output_foldername1):
    imagelist = os.listdir(input_foldername1)
    for i in range(len(imagelist)):
        print(i,'/',len(imagelist))
        case_folder = '_'.join(imagelist[i].split('_')[:3])
        case_folder_path = output_foldername1+'/'+case_folder 
        case_folder_series_path = case_folder_path + '/' + imagelist[i].strip('.png')
        original_path = input_foldername1 + '/' + imagelist[i]
        target_path = case_folder_series_path + '/' + imagelist[i]
        print("original:", original_path)
        print("target:", target_path)
        if not os.path.exists(case_folder_path):
            os.mkdir(case_folder_path)
        if not os.path.exists(case_folder_series_path):
            os.mkdir(case_folder_series_path)
        shutil.copy(original_path, target_path)

#creating mammogram csv file
def create_MIL_csv_file(cbis_ddsm_trainingfile):
    dic_imageinfo={}
    # open the file in the write mode
    f = open(cbis_ddsm_trainingfile, 'w')
    # create the csv writer
    writer = csv.writer(f, delimiter =';')
    header=['FolderName','PatientId','AbnormalityType','Views','ShortPath']
    writer.writerow(header)
    dic_views={'LEFT_CC':'LCC','LEFT_MLO':'LMLO','RIGHT_CC':'RCC','RIGHT_MLO':'RMLO'}
    output_folderlist = os.listdir(output_foldername)
    for i in range(len(output_folderlist)):
        views=''
        print(i,'/',len(output_folderlist))
        tumortype = output_folderlist[i].split('-')[0]
        patientid = '_'.join(output_folderlist[i].split('_')[1:3])
        path = output_folderlist[i]
        views_folder = os.listdir(path)
        for view_folder in views_folder: 
            if views=='':
                views=dic_views['_'.join(view_folder.split('_')[3:5])]
            else:
                views=views+'+'+dic_views['_'.join(view_folder.split('_')[3:5])]
        
        row = [output_folderlist[i],patientid,tumortype,views,path]
        # write a row to the csv file
        writer.writerow(row)
        
    # close the file
    f.close()

def aggregating_patient_case_info(grp):
    #grp_agg=pd.DataFrame(columns=['FolderName', 'PatientId', 'AbnormalityType', 'Views', 'FullPath', 'BreastDensity', 'Assessment', 'Groundtruth'], index=range(1))
    grp_agg=pd.DataFrame(columns=['FolderName', 'PatientId', 'BreastDensity', 'Assessment', 'Groundtruth','AssessmentMax'], index=range(1))
    #if len(np.unique(grp['breast_density']))>1:
    #    print("yes")
    grp_agg['FolderName']=np.unique(grp['foldername'])
    grp_agg['PatientId']=np.unique(grp['patient_id'])
    try:
        grp_agg['BreastDensity']=np.unique(grp['breast_density'])
    except:
        grp_agg['BreastDensity']=np.unique(grp['breast density'])
    #if len(np.unique(grp['assessment']))>1:
    #    print(list(np.unique(grp['assessment'])))
    #    print(grp[['patient_id','assessment','left or right breast','image view', 'abnormality type','subtlety']])
    grp_agg['Assessment']=",".join(str(i) for i in list(np.unique(grp['assessment'])))
    grp_agg['AssessmentMax']=max(list(np.unique(grp['assessment'])))
    grp['pathology']=grp['pathology'].map({'BENIGN':'benign','MALIGNANT':'malignant','BENIGN_WITHOUT_CALLBACK':'benign'})
    #print(grp['pathology'])
    #print(len(np.unique(grp['pathology'])))
    if len(np.unique(grp['pathology']))>1:
        if 'malignant' in list(np.unique(grp['pathology'])):
            grp_agg['Groundtruth']='malignant'
    else:
        grp_agg['Groundtruth']=np.unique(grp['pathology'])
    return grp_agg

#adding groundtruth to MIL mammogram csv
def add_caselevel_groundtruth_MIL_csvfile():
    df_modality = pd.read_csv(cbis_ddsm_trainingfile,sep=';')
    print(df_modality.shape)
    df_original_masstrain = pd.read_csv(cbis_ddsm_originalfile_masstrain,sep=',')
    df_original_masstrain['foldername']=df_original_masstrain['image file path'].str.split('_').apply(lambda x: "_".join(x[:3]))
    #df_original.groupby(by=['foldername']).apply(lambda x: len(np.unique(x['pathology']))>1)
    df_patientfolder_masstrain = df_original_masstrain.groupby(by=['foldername'],as_index=False,group_keys=False).apply(aggregating_patient_case_info)
    df_merged_masstrain=df_modality.merge(df_patientfolder_masstrain,on=['FolderName','PatientId'],how='inner')
    print(df_merged_masstrain.shape)
    print(df_merged_masstrain)
    input('halt')

    df_original_masstest=pd.read_csv(cbis_ddsm_originalfile_masstest,sep=',')
    df_original_masstest['foldername']=df_original_masstest['image file path'].str.split('_').apply(lambda x: "_".join(x[:3]))
    df_patientfolder_masstest=df_original_masstest.groupby(by=['foldername'],as_index=False,group_keys=False).apply(aggregating_patient_case_info)
    df_merged_masstest=df_modality.merge(df_patientfolder_masstest,on=['FolderName','PatientId'],how='inner')
    print(df_merged_masstest.shape)
    print(df_merged_masstest)
    input('halt')

    df_original_calctrain=pd.read_csv(cbis_ddsm_originalfile_calctrain,sep=',')
    df_original_calctrain['foldername']=df_original_calctrain['image file path'].str.split('_').apply(lambda x: "_".join(x[:3]))
    df_patientfolder_calctrain=df_original_calctrain.groupby(by=['foldername'],as_index=False,group_keys=False).apply(aggregating_patient_case_info)
    df_merged_calctrain=df_modality.merge(df_patientfolder_calctrain,on=['FolderName','PatientId'],how='inner')
    print(df_merged_calctrain.shape)
    print(df_merged_calctrain)
    input('halt')

    df_original_calctest=pd.read_csv(cbis_ddsm_originalfile_calctest,sep=',')
    df_original_calctest['foldername']=df_original_calctest['image file path'].str.split('_').apply(lambda x: "_".join(x[:3]))
    df_patientfolder_calctest=df_original_calctest.groupby(by=['foldername'],as_index=False,group_keys=False).apply(aggregating_patient_case_info)
    df_merged_calctest=df_modality.merge(df_patientfolder_calctest,on=['FolderName','PatientId'],how='inner')
    print(df_merged_calctest.shape)
    print(df_merged_calctest)

    df_merged=[df_merged_masstrain,df_merged_masstest,df_merged_calctrain,df_merged_calctest]
    df_merged = pd.concat(df_merged)
    df_merged=df_merged.sort_values(by='PatientId')
    print(df_merged)
    print(df_merged[~df_merged['Groundtruth'].isnull()].shape)

    df_merged.to_csv('/home/MG_training_files_cbis-ddsm_multiinstance_groundtruth.csv',sep=';')

if file_creation == 'MIL':
    input_foldername = "<path to cleaned/preprocessed images>"
    output_foldername = "<path to multi-instance data>"
    #dicom_folder = "C:/Users/PathakS/Shreyasi/CBIS-DDSM/manifest-ZkhPvrLo5216730872708713142/CBIS-DDSM"  

    #ROI input files provided by cbis-ddsm website
    cbis_ddsm_trainingfile = '/home/MG_training_files_cbis-ddsm_multiinstance.csv'
    cbis_ddsm_originalfile_masstrain = '/home/mass_case_description_train_set.csv'
    cbis_ddsm_originalfile_masstest = '/home/mass_case_description_test_set.csv'
    cbis_ddsm_originalfile_calctrain = '/home/calc_case_description_train_set.csv'
    cbis_ddsm_originalfile_calctest = '/home/calc_case_description_test_set.csv'

    creating_MIL_folder_structure(input_foldername, output_foldername)
    create_MIL_csv_file(cbis_ddsm_trainingfile)
    add_caselevel_groundtruth_MIL_csvfile()
#------------------------------------------------------------------MIL csv file creation end----------------------------------------------#









#----------------------------------------------------------------SIL input csv file creation----------------------------------------------#
def conflicting_groundtruth(grp):
    grp_agg=pd.DataFrame(columns=['ImageName', 'Patient_Id', 'BreastDensity','Views', 'AbnormalityType', 'Assessment','Groundtruth','FullPath'], index=range(1))
    grp_agg['ImageName']=np.unique(grp['ImageName'])
    grp_agg['Patient_Id']=np.unique(grp['Patient_Id'])
    try:
        grp_agg['BreastDensity']=np.unique(grp['breast_density'])
    except:
        grp_agg['BreastDensity']=np.unique(grp['breast density'])
    grp_agg['Views']=np.unique(grp['Views'])
    grp_agg['AbnormalityType']=np.unique(grp['abnormality type'])
    grp_agg['Assessment']=",".join(str(i) for i in list(np.unique(grp['assessment'])))
    grp_agg['AssessmentMax']=max(list(np.unique(grp['assessment'])))
    grp_agg['FullPath']=np.unique(grp['FullPath'])
    #grp_agg['Subtlety']=np.unique(grp['subtlety'])
    if len(np.unique(grp['Groundtruth']))>1:
        if 'malignant' in list(np.unique(grp['Groundtruth'])):
            grp_agg['Groundtruth']='malignant'
    else:
        grp_agg['Groundtruth']=np.unique(grp['Groundtruth'])
    return grp_agg

#create single instance csv file
def create_SIL_csvfile():
    df_original_masstrain=pd.read_csv(cbis_ddsm_originalfile_masstrain,sep=',')
    df_original_masstest=pd.read_csv(cbis_ddsm_originalfile_masstest,sep=',')
    df_original_calctrain=pd.read_csv(cbis_ddsm_originalfile_calctrain,sep=',').rename(columns={'breast density':'breast_density'})
    df_original_calctest=pd.read_csv(cbis_ddsm_originalfile_calctest,sep=',').rename(columns={'breast density':'breast_density'})
    print(df_original_masstrain.shape)
    print(df_original_masstest.shape)
    print(df_original_calctrain.shape)
    print(df_original_calctest.shape)

    df_merged=[df_original_masstrain,df_original_masstest,df_original_calctrain,df_original_calctest]
    df_merged = pd.concat(df_merged)
    print(df_merged)
    df_merged['FullPath']=df_merged['image file path'].apply(lambda x: '/projects/dso_mammovit/project_kushal/data/processed/'+x.split('/')[0]+'_1-1.png')
    df_merged['Views'] = df_merged['left or right breast'].map({'LEFT':'L','RIGHT':'R'})+df_merged['image view']
    df_merged['Groundtruth']=df_merged['pathology'].map({'BENIGN':'benign','MALIGNANT':'malignant','BENIGN_WITHOUT_CALLBACK':'benign'})
    df_merged['ImageName']=df_merged['image file path'].apply(lambda x: x.split('/')[0]+'_1.1')
    df_merged = df_merged.rename(columns={'patient_id':'Patient_Id'})

    #df_merged = df_merged.drop(['image file path','cropped image file path','ROI mask file path'], axis=1)
    #print(df_merged.groupby(by=['ImageName']).filter(lambda x: len(np.unique(x['Groundtruth']))>1))

    df_dup = df_merged.groupby(by=['ImageName'],as_index=False,group_keys=False).filter(lambda x: len(np.unique(x['Groundtruth'])) > 1)
    #print(df_dup)
    print(df_dup.shape) # number of images with mutliple abnormalities -> 18 images.

    df_merged = df_merged.groupby(by=['ImageName'],as_index=False,group_keys=False).apply(conflicting_groundtruth)

    df_merged = df_merged.sort_values(by='Patient_Id')
    #df_merged = df_merged[df_merged.duplicated(subset=['ImageName'])]
    df_merged.to_csv('/projects/dso_mammovit/project_kushal/data/MG_training_files_cbis-ddsm_singleinstance_groundtruth.csv',sep=';',na_rep='NULL')

#merge SIL_imagelabel with SIL_caselabel csv such that the final file contain 2 columns - imagelabel and caselabel    
def create_final_SIL_csvfile(imagelabel_file):
    df_imagelabel = pd.read_csv(imagelabel_file, sep=';')
    df_imagelabel['FolderName'] = df_imagelabel.str.split('_').apply(lambda x: "_".join(x[:3]))
    df_imagelabel['CaseLabel'] = df_imagelabel.groupby(by='FolderName')['Groundtruth'].apply(lambda x: 'malignant' if 'malignant' in list(np.unique(x['Groundtruth'])) else 'benign')
    df_imagelabel = df_imagelabel.rename(columns={'Groundtruth':'ImageLabel'})
    print(df_imagelabel.shape)
    df_imagelabel.to_csv('/projects/dso_mammovit/project_kushal/data/cbis-ddsm_singleinstance_groundtruth.csv',sep=';',na_rep='NULL', index=False)

#remove fullpath from the above created sil csv and add the short path, which is the path of the image in multiinstance_data_8bit
def shortpath_addition_SIL_csvfile(sil_csvfile, mil_imgpath):
    df_sil = pd.read_csv(sil_csvfile, sep=';')
    for idx in df_sil.index:
        shortpath = glob.glob(mil_imgpath+'/'+'_'.join(df_sil.loc[idx, 'ImageName'].split('_')[:3])+'/**/'+df_sil.loc[idx, 'ImageName'].replace('.','-')+'.png', recursive=True)[0]
        df_sil.loc[idx,'ShortPath'] = "/".join(shortpath.split('/')[-3:])
        df_sil.loc[idx,'ImageName'] = df_sil.loc[idx,'ImageName'].replace('.','-')
        #print(df_sil.loc[idx,'ShortPath'])
    #df_sil = df_sil.drop(['FullPath_x', 'FullPath_y', 'Unnamed: 0'], axis=1)
    df_sil = df_sil.drop(['Unnamed: 0'], axis=1)
    df_sil.to_csv('/home/cbis-ddsm_singleinstance_groundtruth.csv',sep=';',na_rep='NULL', index=False)

if file_creation == 'SIL':
    create_SIL_csvfile()
    create_final_SIL_csvfile('/home/MG_training_files_cbis-ddsm_singleinstance_groundtruth.csv')
    shortpath_addition_SIL_csvfile('/home/cbis-ddsm_singleinstance_groundtruth.csv', '/home/multiinstance_data')
#----------------------------------------------SIL input csv file creation----------------------------------------------#