import torch.optim as optim

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

def select_lr_scheduler(config_params, optimizer):
    if config_params['trainingmethod'] == 'multisteplr1':
        scheduler = multisteplr_routine1(optimizer)
    elif config_params['trainingmethod'] == 'lrdecayshu':
        scheduler = lrdecay_scheduler_shu(optimizer)
    elif config_params['trainingmethod'] == 'lrdecaykim':
        scheduler = lrdecay_scheduler_kim(optimizer)
    else:
        scheduler = None
    return scheduler

def optimizer_fn(config_params, model):
    if config_params['viewsinclusion']=='all':
        both_attention_group=[]
        image_attention_group=[]
        perbreast_attention_group=[]
        rest_group=[]
        param_list=[]
        
        for name,param in model.named_parameters():
            if param.requires_grad:
                if 'both.attention' in name:
                    both_attention_group.append(param)
                elif 'perbreast.attention' in name:
                    perbreast_attention_group.append(param)
                elif 'img.attention' in name:
                    image_attention_group.append(param)
                else:
                    rest_group.append(param)
        for item in [both_attention_group, image_attention_group, perbreast_attention_group, rest_group]:
            if item:
                param_list.append({"params":item, "lr":config_params['lr'], "momentum":0.9, "weight_decay": config_params['wtdecay']})
        if config_params['optimizer']=='Adam':
            optimizer = optim.Adam(param_list)
        elif config_params['optimizer']=='SGD':
            optimizer = optim.SGD(param_list)
    
    elif config_params['viewsinclusion'] == 'standard':
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
            else:
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config_params['lr'], weight_decay=config_params['wtdecay'])
            
        elif config_params['optimizer']=='SGD':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config_params['lr'], momentum=0.9, weight_decay=config_params['wtdecay'])
    return optimizer