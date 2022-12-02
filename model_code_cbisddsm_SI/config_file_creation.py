# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 21:06:36 2022

@author: PathakS
"""

from configparser import ConfigParser
import os

#Get the configparser object
config_object = ConfigParser()

#Assume we need 2 sections in the config file, let's call them USERINFO and SERVERCONFIG
config_object["parametersetting"] = {
        "modelid": 74,
        "attention": False,  #options = imagewise, breastwise, False
        "milpooling": False, #options=maxpool, average, attention, gatedattention, concat
        "activation": 'sigmoid', #options=sigmoid, softmax
        "featureextractor": 'common', #options=common, viewwise, sharedseparatemix, allseparate
        "data": 'fixed', #option=fixed, variable
        "classimbalance": 'poswt', #options=wtcostfunc, poswt, oversampling, focalloss,False
        "optimizer": 'Adam', #options=SGD, Adam
        "modality": 'MG',
        "patienceepoch": 10,
        "batchsize": 20, #options=10, 20
        "numclasses": 1,
        "maxepochs": 100, #150
        "numworkers": 8,
        "lr": 0.00002, #0.001, 0.00002
        "wtdecay": 0.00001, #0.0005, 0.00001
        "groundtruthdic": {'benign':0,'malignant':1},
        "classes": [0,1],
        "baseline": False,
        "resize": 800, #options=1600, None (for padding to max image size )
        "dataaug": 'big', #options=small, big, wang
        "dataset": 'officialtestset',
        "datasplit":'stratifiedpatient',
        "datascaling": 'scaling', #options=scaling,standardize,standardizeperimage,False
        "flipimage": False,
        "randseedother": 8, #options=8,24,80
        "randseeddata": 8, #options=8,24,80
        "device": 'cuda:3',
        "trainingmethod": False, #multisteplr1
        "featurenorm": 'rgb',
        "femodel": 'densenet169pretrained', #options: resnet50pretrainedrgbwang, densenet169pretrained
        "run": False,
        "topkpatch": 0.7,
        "extra": 'shu_ggp' #rgp
}

filename=''

for key in config_object["parametersetting"].keys():
    print(key, config_object["parametersetting"][key])
    if key in ['modelid','attention','milpooling','data','classimbalance','baseline','flipimage','femodel','extra']:
        #print(key, config_object["parametersetting"][key])
        if config_object["parametersetting"][key]!='False':
            if filename=='':
                filename=key+config_object["parametersetting"][key]
            else:
                filename=filename+'_'+key+config_object["parametersetting"][key]

#filename=filename+'_topkpatch'+str(config_object["parametersetting"]["topkpatch"])+'_SILmodel'

filename=filename+'_SILmodel'
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
    #    with open(path_to_output+'config_'+rand_seed+'_'+str(config_object["parametersetting"]["run"])+'.ini', 'w')+'_'+'topkpatch'+str(config_object["parametersetting"]["topkpatch"]) as conf:
    #        config_object.write(conf)
    #else:
    with open(path_to_output+'config_'+rand_seed+'_'+str(config_object["parametersetting"]["run"])+'.ini', 'w') as conf:
        config_object.write(conf)
else:
    with open(path_to_output+'config_'+rand_seed+'.ini', 'w') as conf:
        config_object.write(conf)