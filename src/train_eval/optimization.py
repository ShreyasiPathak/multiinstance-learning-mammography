import torch.optim as optim

def cosineannealing(config_params, optimizer, batches_train):
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=batches_train*config_params['maxepochs'], eta_min=config_params['lr']/10.)
    return scheduler

def cosineannealing_pipnet(config_params, optimizer, batches_train):
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0)
    scheduler_backbone = optim.lr_scheduler.CosineAnnealingLR(optimizer[0], T_max=batches_train*config_params['maxepochs'], eta_min=config_params['lr']/10.)
    scheduler_classifier = optim.lr_scheduler.CosineAnnealingLR(optimizer[1], T_max=batches_train*config_params['maxepochs'], eta_min=0.001, verbose=False)
    return scheduler_backbone, scheduler_classifier

def lrdecay_scheduler_kim(optimizer):
    """Sets the learning rate to the initial LR decayed by 0.2 every 10 epochs. Add the paper reference here."""
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)
    return scheduler

def lrdecay_scheduler_shu(optimizer):
    """Sets the learning rate to the initial LR decayed by 0.2 every 10 epochs"""
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.98)
    return scheduler

def multisteplr_routine1(optimizer): #multisteplr1
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5], gamma=0.1) #6 means 7. epochs starts with 0. So, from 7th epoch the model will move to another learning rate
    return scheduler

def select_lr_scheduler(config_params, optimizer, batches_train):
    if config_params['trainingmethod'] == 'multisteplr1':
        scheduler = multisteplr_routine1(optimizer)
    elif config_params['trainingmethod'] == 'lrdecayshu':
        scheduler = lrdecay_scheduler_shu(optimizer)
    elif config_params['trainingmethod'] == 'lrdecaykim':
        scheduler = lrdecay_scheduler_kim(optimizer)
    elif config_params['trainingmethod'] == 'cosineannealing':
        scheduler = cosineannealing(config_params, optimizer, batches_train)
    elif config_params['trainingmethod'] == 'cosineannealing_pipnet':
        scheduler_backbone, scheduler_classifier  = cosineannealing_pipnet(config_params, optimizer, batches_train)
        return scheduler_backbone, scheduler_classifier
    else:
        scheduler = None
    return scheduler

def optimizer_fn(config_params, model):
    if config_params['viewsinclusion']=='all' and config_params['learningtype']=='MIL':
        image_attention_group=[]
        side_attention_group=[]
        rest_group=[]
        param_list=[]
        
        for name,param in model.named_parameters():
            if param.requires_grad:
                if 'img.attention' in name:
                    image_attention_group.append(param)
                elif 'side.attention' in name:
                    side_attention_group.append(param)
                else:
                    rest_group.append(param)
        for item in [image_attention_group, side_attention_group, rest_group]:
            if item:
                if config_params['optimizer']=='Adam':
                    param_list.append({"params":item, "lr":config_params['lr'], "weight_decay": config_params['wtdecay']})
                elif config_params['optimizer']=='SGD':
                    param_list.append({"params":item, "lr":config_params['lr'], "momentum":0.9, "weight_decay": config_params['wtdecay']})
        
        if config_params['optimizer']=='Adam':
            optimizer = optim.Adam(param_list)
        elif config_params['optimizer']=='SGD':
            optimizer = optim.SGD(param_list)
    
    else: #config_params['viewsinclusion'] == 'standard':
        classifier=[]
        rest_group=[]
        if config_params['optimizer']=='Adam':
            if config_params['papertoreproduce'] == 'shu': 
                for name,param in model.named_parameters():
                    if param.requires_grad:
                        if '.fc' in name:
                            classifier.append(param)
                        else:
                            rest_group.append(param)
                optimizer = optim.Adam([{'params':classifier, 'lr':0.0001, "weight_decay":config_params['wtdecay']},{'params':rest_group, 'lr':config_params['lr'], "weight_decay":config_params['wtdecay']}])
            elif config_params['trainingmethod'] == 'cosineannealing_pipnet':
                for name,param in model.named_parameters():
                    if param.requires_grad:
                        if 'classifier' in name:
                            classifier.append(param)
                        else:
                            rest_group.append(param)
                optimizer = optim.Adam([{'params':rest_group, 'lr':config_params['lr'], "weight_decay":config_params['wtdecay']}])
                optimizer_classifier = optim.Adam([{'params':classifier, 'lr':0.05, "weight_decay":config_params['wtdecay']}])
                return optimizer, optimizer_classifier
            else:
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config_params['lr'], weight_decay=config_params['wtdecay'])
            
        elif config_params['optimizer']=='SGD':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config_params['lr'], momentum=0.9, weight_decay=config_params['wtdecay'])
    return optimizer