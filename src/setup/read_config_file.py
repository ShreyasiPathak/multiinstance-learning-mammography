import ast
from configparser import ConfigParser
#import mlflow

def read_config_file(config_file):
    '''
    read the configuration file set by the user.
    '''
    config_object = ConfigParser()
    with open(config_file, 'r', encoding='utf-8') as f:
        config_object.read_file(f)
    #global config_params 
    config_params = {}
    
    #parameters from config.ini for training the model
    config_params['randseedother'] = int(config_object["parametersetting"]['randseedother'])
    config_params['randseeddata'] = int(config_object["parametersetting"]['randseeddata'])
    config_params['batchsize'] = int(config_object['parametersetting']['batchsize'])#10
    config_params['numclasses'] = int(config_object["parametersetting"]['numclasses'])
    config_params['maxepochs'] =  int(config_object["parametersetting"]['maxepochs'])
    config_params['numworkers'] = int(config_object["parametersetting"]['numworkers'])
    #config_params['numworkers'] = 8
    config_params['groundtruthdic'] = ast.literal_eval(config_object["parametersetting"]['groundtruthdic'])
    config_params['classes'] = ast.literal_eval(config_object["parametersetting"]['classes'])
    config_params['resize'] = ast.literal_eval(config_object["parametersetting"]['resize'])
    config_params['activation'] = config_object["parametersetting"]['activation']
    config_params['viewsinclusion'] = config_object["parametersetting"]['viewsinclusion'] #-> from data to viewsinclusion
    #config_params['viewsinclusion'] = 'all'
    config_params['dataaug'] = config_object["parametersetting"]['dataaug']
    config_params['dataset'] = config_object["parametersetting"]['dataset']
    config_params['datasplit'] = config_object["parametersetting"]['datasplit']
    config_params['optimizer'] = config_object["parametersetting"]['optimizer']
    config_params['milpooling'] = config_object["parametersetting"]['milpooling']
    config_params['device'] = config_object["parametersetting"]['device']
    config_params['learningtype'] = config_object["parametersetting"]['learningtype']
    config_params['channel'] = int(config_object["parametersetting"]['channel'])
    config_params['labeltouse'] = config_object["parametersetting"]['labeltouse']
    config_params['papertoreproduce'] = config_object["parametersetting"]['papertoreproduce']
    config_params['regionpooling'] = config_object["parametersetting"]['regionpooling']
    
    config_params['patienceepochs'] = config_object["parametersetting"]['patienceepochs']#14
    if config_params['patienceepochs'] == 'False':
        config_params['patienceepochs'] = False
    else:
        config_params['patienceepochs'] = int(config_params['patienceepochs'])

    config_params['classimbalance'] = config_object["parametersetting"]['classimbalance']
    if config_params['classimbalance'] == 'False':
        config_params['classimbalance'] = False
    
    config_params['attention'] = config_object["parametersetting"]['attention']
    if config_params['attention'] == 'False':
        config_params['attention'] = False
    
    config_params['datascaling'] = config_object["parametersetting"]['datascaling']
    if config_params['datascaling'] == 'False':
        config_params['datascaling'] = False
    
    config_params['extra'] = config_object["parametersetting"]['extra']
    #config_params['extra'] = 'False'
    if config_params['extra'] == 'False':
        config_params['extra'] = False
    
    config_params['flipimage'] = config_object["parametersetting"]['flipimage']
    if config_params['flipimage'] == 'False':
        config_params['flipimage'] = False
    else:
        config_params['flipimage'] == True
    
    config_params['femodel'] = config_object["parametersetting"]['femodel']
    if config_params['femodel'] == 'False':
        config_params['femodel'] = False
    
    config_params['trainingmethod'] = config_object["parametersetting"]['trainingmethod']
    if config_params['trainingmethod'] == 'False':
        config_params['trainingmethod'] = False
    
    try:
        config_params['run'] = config_object["parametersetting"]['run']
        if config_params['run'] == 'False':
            config_params['run'] = False
    except:
        config_params['run'] = False
    
    try:
        config_params['lr'] = float(config_object["parametersetting"]['lr'])
    except:
        config_params['lr'] = False
    
    try:
        config_params['wtdecay'] = float(config_object["parametersetting"]['wtdecay'])
    except:
        config_params['wtdecay'] = False
    
    try:
        config_params['topkpatch'] = config_object["parametersetting"]['topkpatch']
        if config_params['topkpatch']=='False':
            config_params['topkpatch']=False
        else:
            config_params['topkpatch']=float(config_params['topkpatch'])
    except:
        config_params['topkpatch']=False   
    
    try:
        config_params['usevalidation'] = config_object["parametersetting"]['usevalidation']
        if config_params['usevalidation'] == 'False':
            config_params['usevalidation'] = False
        else:
            config_params['usevalidation'] = True
    except:
        config_params['usevalidation'] == 'False'
    
    
    config_params['pretrained'] = config_object["parametersetting"]['pretrained']
    if config_params['pretrained'] == 'False':
        config_params['pretrained'] = False
    else:
        config_params['pretrained'] = True

    try:
        config_params['sm_reg_param'] = config_object["parametersetting"]['sm_reg_param']
        if config_params['sm_reg_param']=='False':
            config_params['sm_reg_param']=False
        else:
            config_params['sm_reg_param'] = float(config_params['sm_reg_param'])
    except:
        config_params['sm_reg_param'] = False

    try:
        config_params['imagecleaning'] = config_object["parametersetting"]['imagecleaning']
        if config_params['imagecleaning'] == 'False':
            config_params['imagecleaning'] = False
    except:
        config_params['imagecleaning'] = False

    config_params['ROIpatches'] = config_object["parametersetting"]['ROIpatches']
    if config_params['ROIpatches'] == 'False':
        config_params['ROIpatches'] = False
    else:
        config_params['ROIpatches'] = int(config_params['ROIpatches'])

    try:
        config_params['valloss_resumetrain'] = config_object["parametersetting"]["valloss_resumetrain"]
        if config_params['valloss_resumetrain'] == 'False':
            config_params['valloss_resumetrain'] = None
        else:
            config_params['valloss_resumetrain'] = -float(config_params['valloss_resumetrain'])
    except:
        config_params['valloss_resumetrain'] = None
    
    try:
        config_params['crop_shape'] = ast.literal_eval(config_object["parametersetting"]["crop_shape"])
    except:
        config_params['crop_shape'] = (256, 256)
    
    try:
        config_params['cam_size'] = ast.literal_eval(config_object["parametersetting"]["cam_size"])
    except:
        config_params['cam_size'] = (92, 60)

    try:
        config_params['dependency'] = config_object["parametersetting"]['dependency']
        if config_params['dependency'] == 'False':
            config_params['dependency'] = False
    except:
        config_params['dependency'] = False
    
    try:
        config_params['selfatt-nonlinear'] = config_object["parametersetting"]['selfatt-nonlinear']
        if config_params['selfatt-nonlinear'] == 'False':
            config_params['selfatt-nonlinear'] = False
        elif config_params['selfatt-nonlinear'] == 'True':
            config_params['selfatt-nonlinear'] = True
    except:
        config_params['selfatt-nonlinear'] = False
    
    try:
        config_params['selfatt-gamma'] = config_object["parametersetting"]['selfatt-gamma']
        if config_params['selfatt-gamma'] == 'False':
            config_params['selfatt-gamma'] = False
        elif config_params['selfatt-gamma'] == 'True':
            config_params['selfatt-gamma'] = True
    except:
        config_params['selfatt-gamma'] = False

    config_params['SIL_csvfilepath'] = config_object["parametersetting"]['SIL_csvfilepath']
    #config_params['SIL_csvfilepath'] = "/home/pathaks/PhD/case-level-breast-cancer/multiview_mammogram/input_data/MG_training_files_studyUID_accessionNum_viewnames_final4_viewsextended_SI.csv"
    if config_params['SIL_csvfilepath'] == 'False':
        config_params['SIL_csvfilepath'] = False
    
    config_params['MIL_csvfilepath'] = config_object["parametersetting"]['MIL_csvfilepath'] #"/homes/spathak/multiview_mammogram/input_data/MG_training_files_studyUID_accessionNum_viewnames_biradscombined_breastdensityadded_patientinfoadded_final4.csv"
    #config_params['MIL_csvfilepath'] = "/home/pathaks/PhD/case-level-breast-cancer/multiview_mammogram/input_data/MG_training_files_studyUID_accessionNum_viewnames_final4_viewsextended.csv"
    if config_params['MIL_csvfilepath'] == 'False':
        config_params['MIL_csvfilepath'] = False
    
    config_params['preprocessed_imagepath'] = config_object["parametersetting"]['preprocessed_imagepath']
    #config_params['preprocessed_imagepath'] = "/deepstore/datasets/dmb/medical/breastcancer/mammography/zgt"
    config_params['bitdepth'] = int(config_object["parametersetting"]['bitdepth'])
    
    try:
        config_params['early_stopping_criteria'] = config_object["parametersetting"]['early_stopping_criteria']
        if config_params['early_stopping_criteria'] == 'False':
            config_params['early_stopping_criteria'] = False
    except:
        config_params['early_stopping_criteria'] = 'loss'

    if config_params['femodel']=='gmic_resnet18':
        config_params['gmic_parameters'] = {
            "device_type": 'gpu',
            #"gpu_number": str(config_params['device'].split(':')[1]),
            "gpu_number": str(config_params['device']),
            "max_crop_noise": (100, 100),
            "max_crop_size_noise": 100,
            # model related hyper-parameters
            "cam_size": config_params['cam_size'],
            "K": config_params['ROIpatches'],
            "crop_shape": config_params['crop_shape'],
            "post_processing_dim": 512,
            "num_classes": config_params['numclasses'],
            "use_v1_global":True,
            "percent_t": config_params['topkpatch'],
            'arch':'resnet18',
            'pretrained': config_params['pretrained'],
            'learningtype': config_params['learningtype']
        }

    #config_params['dataset'] = 'zgt'
    #config_params['viewsinclusion'] = 'all'
    #config_params['SIL_csvfilepath'] = "/homes/spathak/multiview_mammogram/input_data/MG_training_files_studyUID_accessionNum_viewnames_final4_viewsextended_SI.csv"
    #config_params['MIL_csvfilepath'] = "/homes/spathak/multiview_mammogram/input_data/MG_training_files_studyUID_accessionNum_viewnames_final4_viewsextended.csv"
    #config_params['preprocessed_imagepath'] = "/groups/dso/spathak"
    #config_params['bitdepth'] = 12
    #config_params['batchsize'] = 4
    #config_params['datasplit'] = 'casebasedtestset'
    #config_params['labeltouse'] = 'caselabel'

    #with mlflow.start_run():
    #mlflow.log_params(config_params)

    return config_params