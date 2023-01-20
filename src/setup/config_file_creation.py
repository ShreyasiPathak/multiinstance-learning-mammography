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

names=[]
start = 0
end =  1#len(hyperparam_config)
count = 0

for hyperparam in hyperparam_config[start:end]:
    #Get the configparser object
    config_object = ConfigParser()
    #Assume we need 2 sections in the config file, let's call them USERINFO and SERVERCONFIG
    config_object["parametersetting"] = {
            "modelid": 105,
            "run": start + count,
            "regionpooling": 't-pool', #options: shu_ggp, shu_rgp, avgpool, maxpool, 1x1conv, t-pool
            "attention": False,  #options = imagewise, breastwise, False
            "milpooling": False, #options=maxpool, average, attention, gatedattention, concat/ ismax, ismean, isatt, isgatt, esmax, esmean, esatt, esgatt
            "activation": 'sigmoid', #options = sigmoid, softmax
            "viewsinclusion": 'standard', #option = standard, all -> change this to viewsinclusion: standard, all
            "classimbalance": 'poswt', #options = wtcostfunc, poswt, oversampling, focalloss,False
            "optimizer": 'Adam', #options = SGD, Adam
            "patienceepochs": False, #10
            "usevalidation": True,
            "batchsize": 10, #options=10, 20
            "numclasses": 1,
            "maxepochs": 50, #150
            "numworkers": 8,
            "lr": float(hyperparam['lr']), #10**float(hyperparam['lr']), #0.001, 0.00002
            "wtdecay": float(hyperparam['wtdecay']), #10**float(hyperparam['wtdecay']), #0.0005, 0.00001
            "sm_reg_param": float(hyperparam['sm_reg_param']), #10**float(hyperparam['sm_reg_param']),
            "groundtruthdic": {'benign':0,'malignant':1},
            "classes": [0,1],
            "resize": [2944,1920], #options=1600, None (for padding to max image size )
            "dataaug": 'gmic', #options=small, big, wang, gmic, kim, shu
            "imagecleaning": 'gmic',
            "datasplit": 'officialtestset', #change this to datasplit
            "datascaling": 'scaling', #options=scaling, standardize, standardizeperimage,False
            "flipimage": True,
            "randseedother": 8, #options=8, 24, 80
            "randseeddata": 8, #options=8, 24, 80
            "device": 'cuda:5',
            "trainingmethod": 'fixedlr', #options: multisteplr1, fixedlr, lrdecayshu, lrdecaykim
            "channel": 3, #options: 3 for rgb, 1 for grayscale
            "femodel": 'gmic_resnet18', #options: resnet50pretrainedrgbwang, densenet169pretrained
            "pretrained": True, #options: True, False
            "topkpatch": 0.02, #options: 0.02, 0.03, 0.05, 0.1
            "ROIpatches": 6, #options: any number
            "learningtype": 'SIL', #options = SIL, MIL
            "dataset": 'cbis-ddsm', #options = cbis-ddsm, zgt, vindr
            "bitdepth": 16, #options: 8, 16
            "labeltouse": 'imagelabel', #options: imagelabel, caselabel
            "SIL_csvfilepath": "/projects/dso_mammovit/project_kushal/data/MG_training_files_cbis-ddsm_singleinstance_groundtruth.csv",
            "MIL_csvfilepath": "/projects/dso_mammovit/project_kushal/data/MG_training_files_cbis-ddsm_multiinstance_groundtruth.csv",
            "preprocessed_imagepath": "/projects/dso_mammovit/project_kushal/data/processed-images-gmic/",
            "papertoreproduce": "shu",
            "extra": False #rgp 
    }
    count+=1
    filename=''

    for key in config_object["parametersetting"].keys():
        print(key, config_object["parametersetting"][key])
        if key in ['modelid','attention','milpooling','femodel','viewsinclusion','learningtype']:
            #print(key, config_object["parametersetting"][key])
            if config_object["parametersetting"][key]!='False':
                if filename=='':
                    filename=key+config_object["parametersetting"][key]
                else:
                    filename=filename+'_'+key+config_object["parametersetting"][key]

    #filename=filename+'_topkpatch'+str(config_object["parametersetting"]["topkpatch"])+'_SILmodel'
    print(filename)

    config_object["parametersetting"]['filename']=filename
    path_to_output="/homes/spathak/multiview_mammogram/models_results/cbis-ddsm/newstoryline1/"+filename+"/"

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