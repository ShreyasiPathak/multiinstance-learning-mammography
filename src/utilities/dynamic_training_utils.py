import numpy as np
from copy import deepcopy

def unfreeze_layers(model, layer_keyword_list):
    for layer_keyword in layer_keyword_list:
        for name, param in model.named_parameters():
            if layer_keyword in name:
                param.requires_grad = True
    return model

def freeze_layers(model, layer_keyword_list):
    for layer_keyword in layer_keyword_list:
        for name, param in model.named_parameters():
            if layer_keyword in name:
                param.requires_grad = False
    return model

def unfreeze_wt_update(optimizer, layer_keyword_list, optimizer_params_dic, previous_lr):
    for layer_keyword in layer_keyword_list:
        index = optimizer_params_dic[layer_keyword]
        optimizer.param_groups[index]['lr'] = previous_lr[layer_keyword]
    return optimizer

def freeze_wt_update(optimizer, layer_keyword_list, optimizer_params_dic):
    old_lr = dict()
    for layer_keyword in layer_keyword_list:
        index = optimizer_params_dic[layer_keyword]
        old_lr[layer_keyword] = optimizer.param_groups[index]['lr']
        optimizer.param_groups[index]['lr'] = 0
    return optimizer, old_lr

def return_optimstate(optimizer, layer_keyword_list, optimizer_params_dic):
    optim_state = {}
    #print("keys optim state dict:", optimizer.state_dict()['state'].keys())
    for layer_keyword in layer_keyword_list:
        count_params_end = 0
        index = optimizer_params_dic[layer_keyword]
        count_param_group = len(optimizer.param_groups[index]['params'])
        for id in range(0, index+1):
            count_params_end+= len(optimizer.param_groups[id]['params'])
        
        for id in range(count_params_end-count_param_group, count_params_end):
            if id in optimizer.state_dict()['state'].keys():
                optim_state[id] = deepcopy(optimizer.state_dict()['state'][id])
    
    #print("optim keys:", optim_state.keys())
    return optim_state

def assign_previous_optimstate(optimizer, previous_state):
    if previous_state:
        state_dict = optimizer.state_dict()
        state_dict['state'].update(previous_state)
        optimizer.load_state_dict(state_dict)
    return optimizer

def dynamic_training(config_params, views_names, model, optimizer, previous_state, previous_lr, before_optimupdate):
    if config_params['attention'] == 'breastwise' and (config_params['milpooling'] == 'esatt' or config_params['milpooling'] == 'esgatt' or config_params['milpooling'] == 'isatt' or config_params['milpooling'] == 'isgatt'):
        optimizer_params_dic = {'img.attention':0, 'side.attention':1}
    elif config_params['attention'] == 'imagewise' and (config_params['milpooling'] == 'esatt' or config_params['milpooling'] == 'esgatt' or config_params['milpooling'] == 'isatt' or config_params['milpooling'] == 'isgatt'):
        optimizer_params_dic = {'img.attention':0}
    
    if config_params['milpooling'] == 'esatt' or config_params['milpooling'] == 'esgatt' or config_params['milpooling'] == 'isatt' or config_params['milpooling'] == 'isgatt':
        view_split = np.array([view[1:] for view in views_names])
        view_split = np.unique(view_split).tolist()
        breast_split = np.array([view[0] for view in views_names])
        breast_split = breast_split.tolist()

        #attention weighing switch off
        if config_params['attention'] == 'breastwise':
            #print("I am switching off side.attention and img.attention")
            if config_params['extra'] == 'dynamic_training_async':
                if len(views_names)==1:
                    #if breast_split.count('L')<2 and breast_split.count('R')<2:
                    if before_optimupdate:
                        model = freeze_layers(model, ['img.attention', 'side.attention'])
                        optimizer, previous_lr = freeze_wt_update(optimizer, ['img.attention', 'side.attention'],  optimizer_params_dic)
                        previous_state = return_optimstate(optimizer, ['img.attention', 'side.attention'],  optimizer_params_dic)
                    else:
                        optimizer = assign_previous_optimstate(optimizer, previous_state)
                        model = unfreeze_layers(model, ['img.attention', 'side.attention'])
                        optimizer = unfreeze_wt_update(optimizer, ['img.attention', 'side.attention'],  optimizer_params_dic, previous_lr)
                
                #print("I am switching off side.attention")       
                elif ((breast_split.count('L')>1) and (breast_split.count('R')==0)) or ((breast_split.count('R')>1) and (breast_split.count('L')==0)):
                    if before_optimupdate:
                        model = freeze_layers(model, ['side.attention'])
                        optimizer, previous_lr = freeze_wt_update(optimizer, ['side.attention'],  optimizer_params_dic)
                        previous_state = return_optimstate(optimizer, ['side.attention'],  optimizer_params_dic)
                    else:
                        optimizer = assign_previous_optimstate(optimizer, previous_state)
                        model = unfreeze_layers(model, ['side.attention'])
                        optimizer = unfreeze_wt_update(optimizer, ['side.attention'],  optimizer_params_dic, previous_lr)

                #print("I am switching off img.attention")     
                elif (breast_split.count('L')==1) and (breast_split.count('R')==1):
                    if before_optimupdate:
                        model = freeze_layers(model, ['img.attention'])
                        optimizer, previous_lr = freeze_wt_update(optimizer, ['img.attention'],  optimizer_params_dic)
                        previous_state = return_optimstate(optimizer, ['img.attention'],  optimizer_params_dic)
                    else:
                        optimizer = assign_previous_optimstate(optimizer, previous_state)
                        model = unfreeze_layers(model, ['img.attention'])
                        optimizer = unfreeze_wt_update(optimizer, ['img.attention'],  optimizer_params_dic, previous_lr)
            
            elif config_params['extra'] == 'dynamic_training_momentumupdate':
                if len(views_names)==1:
                    #if breast_split.count('L')<2 and breast_split.count('R')<2:
                    if before_optimupdate:
                        model = freeze_layers(model, ['img.attention', 'side.attention'])
                        optimizer, previous_lr = freeze_wt_update(optimizer, ['img.attention', 'side.attention'],  optimizer_params_dic)
                        #previous_state = return_optimstate(optimizer, ['img.attention', 'side.attention'],  optimizer_params_dic)
                    else:
                        #optimizer = assign_previous_optimstate(optimizer, previous_state)
                        model = unfreeze_layers(model, ['img.attention', 'side.attention'])
                        optimizer = unfreeze_wt_update(optimizer, ['img.attention', 'side.attention'],  optimizer_params_dic, previous_lr)
                
                #print("I am switching off side.attention")       
                elif ((breast_split.count('L')>1) and (breast_split.count('R')==0)) or ((breast_split.count('R')>1) and (breast_split.count('L')==0)):
                    if before_optimupdate:
                        model = freeze_layers(model, ['side.attention'])
                        optimizer, previous_lr = freeze_wt_update(optimizer, ['side.attention'],  optimizer_params_dic)
                        #previous_state = return_optimstate(optimizer, ['side.attention'],  optimizer_params_dic)
                    else:
                        #optimizer = assign_previous_optimstate(optimizer, previous_state)
                        model = unfreeze_layers(model, ['side.attention'])
                        optimizer = unfreeze_wt_update(optimizer, ['side.attention'],  optimizer_params_dic, previous_lr)

                #print("I am switching off img.attention")     
                elif (breast_split.count('L')==1) and (breast_split.count('R')==1):
                    if before_optimupdate:
                        model = freeze_layers(model, ['img.attention'])
                        optimizer, previous_lr = freeze_wt_update(optimizer, ['img.attention'],  optimizer_params_dic)
                        #previous_state = return_optimstate(optimizer, ['img.attention'],  optimizer_params_dic)
                    else:
                        #optimizer = assign_previous_optimstate(optimizer, previous_state)
                        model = unfreeze_layers(model, ['img.attention'])
                        optimizer = unfreeze_wt_update(optimizer, ['img.attention'],  optimizer_params_dic, previous_lr)

            elif config_params['extra'] == 'dynamic_training_couplemomentumupdate':
                if len(views_names)==1:
                    #if breast_split.count('L')<2 and breast_split.count('R')<2:
                    if before_optimupdate:
                        model = freeze_layers(model, ['img.attention', 'side.attention'])
                        optimizer, previous_lr = freeze_wt_update(optimizer, ['img.attention', 'side.attention'],  optimizer_params_dic)
                        previous_state = return_optimstate(optimizer, ['img.attention', 'side.attention'],  optimizer_params_dic)
                    else:
                        optimizer = assign_previous_optimstate(optimizer, previous_state)
                        model = unfreeze_layers(model, ['img.attention', 'side.attention'])
                        optimizer = unfreeze_wt_update(optimizer, ['img.attention', 'side.attention'],  optimizer_params_dic, previous_lr)
                
                #print("I am switching off side.attention")       
                elif ((breast_split.count('L')>1) and (breast_split.count('R')==0)) or ((breast_split.count('R')>1) and (breast_split.count('L')==0)):
                    if before_optimupdate:
                        model = freeze_layers(model, ['side.attention'])
                        optimizer, previous_lr = freeze_wt_update(optimizer, ['side.attention'],  optimizer_params_dic)
                        #previous_state = return_optimstate(optimizer, ['side.attention'],  optimizer_params_dic)
                    else:
                        #optimizer = assign_previous_optimstate(optimizer, previous_state)
                        model = unfreeze_layers(model, ['side.attention'])
                        optimizer = unfreeze_wt_update(optimizer, ['side.attention'],  optimizer_params_dic, previous_lr)

                #print("I am switching off img.attention")     
                elif (breast_split.count('L')==1) and (breast_split.count('R')==1):
                    if before_optimupdate:
                        model = freeze_layers(model, ['img.attention'])
                        optimizer, previous_lr = freeze_wt_update(optimizer, ['img.attention'],  optimizer_params_dic)
                        #previous_state = return_optimstate(optimizer, ['img.attention'],  optimizer_params_dic)
                    else:
                        #optimizer = assign_previous_optimstate(optimizer, previous_state)
                        model = unfreeze_layers(model, ['img.attention'])
                        optimizer = unfreeze_wt_update(optimizer, ['img.attention'],  optimizer_params_dic, previous_lr)

            elif config_params['extra'] == 'dynamic_training_sync':
                if len(views_names)==1:
                    #if breast_split.count('L')<2 and breast_split.count('R')<2:
                    if before_optimupdate:
                        model = freeze_layers(model, ['img.attention', 'side.attention'])
                        optimizer, previous_lr = freeze_wt_update(optimizer, ['img.attention', 'side.attention'],  optimizer_params_dic)
                        previous_state = return_optimstate(optimizer, ['img.attention', 'side.attention'],  optimizer_params_dic)
                    else:
                        optimizer = assign_previous_optimstate(optimizer, previous_state)
                        model = unfreeze_layers(model, ['img.attention', 'side.attention'])
                        optimizer = unfreeze_wt_update(optimizer, ['img.attention', 'side.attention'],  optimizer_params_dic, previous_lr)    
            

        elif config_params['attention'] == 'imagewise':
            if len(views_names)==1:
                if before_optimupdate:
                    model = freeze_layers(model, ['img.attention'])
                    optimizer, previous_lr = freeze_wt_update(optimizer, ['img.attention'],  optimizer_params_dic)
                    previous_state = return_optimstate(optimizer, ['img.attention'],  optimizer_params_dic)
                else:
                    optimizer = assign_previous_optimstate(optimizer, previous_state)
                    model = unfreeze_layers(model, ['img.attention'])
                    optimizer = unfreeze_wt_update(optimizer, ['img.attention'],  optimizer_params_dic, previous_lr)     
        
    if before_optimupdate:
        return model, optimizer, previous_state, previous_lr
    else:
        return model, optimizer