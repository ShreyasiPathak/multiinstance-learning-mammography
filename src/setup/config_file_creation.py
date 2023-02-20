# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 21:06:36 2022

@author: PathakS
"""

from configparser import ConfigParser
import os
import hyperparameter_optim

#hyperparam_config = hyperparameter_optim.generate_hyperparameter_configurations()
#print(hyperparam_config)

#hyperparam_config=[{'lr':7.94328234724282E-06, 'wtdecay':0.0000794328234724282}]
hyperparam_config=[{'lr':0.0000630957344480193, 'wtdecay':0.000316227766016838, 'sm_reg_param': 0.000158489319246111}]
#hyperparam_config=[{'lr':0.00002, 'wtdecay':0.00001}]

names=[]
start = 0
end =  1#len(hyperparam_config)
count = 0

for hyperparam in hyperparam_config[start:end]:
    #Get the configparser object
    config_object = ConfigParser()
    #Assume we need 2 sections in the config file, let's call them USERINFO and SERVERCONFIG
    config_object["parametersetting"] = {
            "modelid": 16,
            "run": False,
            "attention": False,  #options = imagewise, breastwise, False
            "dependency": False,
            "selfatt-nonlinear": False,
            "selfatt-gamma": False,
            "milpooling": False, #options=maxpool, average, attention, gatedattention, concat/ ismax, ismean, isatt, isgatt, esmax, esmean, esatt, esgatt
            "activation": 'sigmoid', #options = sigmoid, softmax
            "viewsinclusion": 'all', #option = standard, all -> change this to viewsinclusion: standard, all; in SIL: standard means all views. I put standard to prevent the dynamic training part of the code.
            "classimbalance": 'poswt', #options = wtcostfunc, poswt, oversampling, focalloss,False
            "optimizer": 'Adam', #options = SGD, Adam
            "patienceepochs": 10, #10
            "usevalidation": True,
            "batchsize": 10, #options=10, 20
            "numclasses": 1,
            "maxepochs": 50, #150
            "numworkers": 8,
            "lr": float(hyperparam['lr']), #10**float(hyperparam['lr']), #0.001, 0.00002
            "wtdecay": float(hyperparam['wtdecay']), #10**float(hyperparam['wtdecay']), #0.0005, 0.00001
            "sm_reg_param": float(hyperparam['sm_reg_param']), #10**float(hyperparam['sm_reg_param']), False
            "groundtruthdic": {'benign':0,'malignant':1},
            "classes": [0,1],
            "resize": [2700, 990], #options=1600, zgt: [2944,1920], vindr:[2700, 990], None (for padding to max image size )
            "cam_size": (85, 31), #vindr: (85, 31), zgt:(92, 60)
            "dataaug": 'gmic', #options=small, big, wang, gmic, kim, shu
            "imagecleaning": 'own',
            "datasplit": 'officialtestset', #options: officialtestset, casebasedtestset
            "datascaling": 'scaling', #options=scaling, standardize, standardizeperimage,False
            "flipimage": True,
            "randseedother": 8, #options=8, 24, 80
            "randseeddata": 8, #options=8, 24, 80
            "device": 'cuda:0',
            "trainingmethod": 'fixedlr', #options: multisteplr1, fixedlr, lrdecayshu, lrdecaykim
            "channel": 3, #options: 3 for rgb, 1 for grayscale
            "regionpooling": 't-pool', #options: shu_ggp, shu_rgp, avgpool, maxpool, 1x1conv, t-pool
            "femodel": 'gmic_resnet18', #options: resnet50pretrainedrgbwang, densenet169
            "pretrained": True, #options: True, False
            "topkpatch": 0.02, #options: 0.02, 0.03, 0.05, 0.1
            "ROIpatches": 6, #options: any number, 6 from gmic paper
            "learningtype": 'SIL', #options = SIL, MIL
            "dataset": 'vindr', #options = cbis-ddsm, zgt, vindr
            "bitdepth": 16, #options: 8, 16
            "labeltouse": 'caselabel', #options: imagelabel, caselabel
            "SIL_csvfilepath": "/groups/dso/spathak/vindr/MG_training_files_vindr_singleinstance_groundtruth.csv", #"/homes/spathak/multiview_mammogram/input_data/MG_training_files_studyUID_accessionNum_viewnames_final4_viewsextended_SI.csv", #"/projects/dso_mammovit/project_kushal/data/cbis-ddsm_singleinstance_groundtruth.csv",
            "MIL_csvfilepath": "/groups/dso/spathak/vindr/MG_training_files_vindr_multiinstance_groundtruth.csv", #"/homes/spathak/multiview_mammogram/input_data/MG_training_files_studyUID_accessionNum_viewnames_final4_viewsextended.csv", #"/projects/dso_mammovit/project_kushal/data/cbis-ddsm_multiinstance_groundtruth.csv",
            "preprocessed_imagepath": "/groups/dso/spathak/vindr/processed_png_16bit", #"/groups/dso/spathak", #"/projects/dso_mammovit/project_kushal/data/multiinstance_data_16bit", #"/local/work/spathak/zgt", "/projects/dso_mammovit/project_kushal/data/multiinstance_data_8bit",
            "valloss_resumetrain": False,
            "papertoreproduce": False,
            "extra": False #options: dynamic_training
    }
    count+=1
    filename=''

    for key in config_object["parametersetting"].keys():
        print(key, config_object["parametersetting"][key])
        if key in ['modelid', 'attention', 'dependency', 'milpooling', 'femodel', 'viewsinclusion', 'papertoreproduce', 'learningtype', 'extra']:# 'regionpooling',]:
            #print(key, config_object["parametersetting"][key])
            if config_object["parametersetting"][key]!='False':
                if filename=='':
                    filename=key+config_object["parametersetting"][key]
                else:
                    filename=filename+'_'+key+config_object["parametersetting"][key]

    #filename=filename+'_topkpatch'+str(config_object["parametersetting"]["topkpatch"])+'_SILmodel'
    print(filename)

    config_object["parametersetting"]['filename']=filename
    if config_object["parametersetting"]['dataset'] == 'cbis-ddsm':
        path_to_output="/homes/spathak/multiview_mammogram/models_results/cbis-ddsm/ijcai2023/"+filename+"/"
    elif config_object["parametersetting"]['dataset'] == 'zgt':
        path_to_output="/homes/spathak/multiview_mammogram/models_results/zgt/ijcai23/"+filename+"/"
    elif config_object["parametersetting"]['dataset'] == 'vindr':
        path_to_output="/homes/spathak/multiview_mammogram/models_results/vindr/ijcai23/"+filename+"/"

    #create output_folder path
    if not os.path.exists(path_to_output):
        os.mkdir(path_to_output)

    if str(config_object["parametersetting"]["randseeddata"])!=str(config_object["parametersetting"]["randseedother"]):
        rand_seed = str(config_object["parametersetting"]["randseedother"]) +'_'+ str(config_object["parametersetting"]["randseeddata"])
    else:
        rand_seed = str(config_object["parametersetting"]["randseeddata"])

    #Write the above sections to config.ini file
    if str(config_object["parametersetting"]["run"])!='False':
        #if str(config_object["parametersetting"]["topkpatch"])!='False':
        #    with open(path_to_output+'config_'+rand_seed+'_'+str(config_object["parametersetting"]["run"])+'_'+'topkpatch'+str(config_object["parametersetting"]["topkpatch"])+'.ini', 'w') as conf:
        #        config_object.write(conf)
        #else:
        with open(path_to_output+'config_'+rand_seed+'_'+'run_'+str(config_object["parametersetting"]["run"])+'.ini', 'w') as conf:
            config_object.write(conf)
    else:
        #if str(config_object["parametersetting"]["topkpatch"])!='False':
        #    with open(path_to_output+'config_'+rand_seed+'_'+'topkpatch'+str(config_object["parametersetting"]["topkpatch"])+'.ini', 'w') as conf:
        #        config_object.write(conf)
        #else:
        with open(path_to_output+'config_'+rand_seed+'.ini', 'w') as conf:
            config_object.write(conf)