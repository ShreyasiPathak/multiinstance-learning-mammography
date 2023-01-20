import torch.optim as optim

def lrdecay_scheduler_kim(optimizer, epoch, init_lr=0.001):
    """Sets the learning rate to the initial LR decayed by 0.2 every 10 epochs"""
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)
    return scheduler

def lrdecay_scheduler_shu(optimizer):#, epoch):
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
    return scheduler

def optimizer_fn(model, config_params):
    if config_params['viewsinclusion']=='all':
        mlo_group=[]
        cc_group=[]
        both_attention_group=[]
        rest_group=[]
        param_list=[]
        
        for name,param in model.named_parameters():
            if param.requires_grad:
                if '.mlo' in name:
                    mlo_group.append(param)
                elif '.cc' in name:
                    cc_group.append(param) 
                elif '_both_b.attention' in name or '_both_m.attention' in name:
                    both_attention_group.append(param)
                else:
                    rest_group.append(param)
        for item in [mlo_group,cc_group,both_attention_group,rest_group]:
            if item:
                param_list.append({"params":item, "lr":config_params['lr'], "momentum":0.9, "weight_decay": config_params['wtdecay']})
        if config_params['optimizer']=='Adam':
            optimizer = optim.Adam(param_list)
        elif config_params['optimizer']=='SGD':
            optimizer = optim.SGD(param_list)
    
    elif config_params['viewsinclusion'] == 'fixed':
        classifier=[]
        rest_group=[]
        if config_params['optimizer']=='Adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config_params['lr'], weight_decay=config_params['wtdecay'])
            '''for name,param in model.named_parameters():
                if param.requires_grad:
                    if '.fc' in name:
                        classifier.append(param)
                    else:
                        rest_group.append(param)
            optimizer = optim.Adam([{'params':classifier, 'lr':0.0001, "weight_decay":wtdecay },{'params':rest_group, 'lr':lrval, "weight_decay":wtdecay }])
            '''
        elif config_params['optimizer']=='SGD':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config_params['lr'], momentum=0.9, weight_decay=config_params['wtdecay'])
    return optimizer