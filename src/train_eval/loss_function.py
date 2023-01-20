import torch
import torch.nn as nn
from torch import linalg as LA
import torch.nn.functional as F

from utilities import utils

class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

def loss_fn_crossentropy(config_params, df, test_bool):
    if test_bool:
        criterion = nn.CrossEntropyLoss()
    else:
        if config_params['classimbalance'] == 'poswt':
            class_weights = utils.class_distribution_poswt(df)
            class_weights = torch.tensor([1,class_weights[0]]).float().to(config_params['device'])
            if config_params['extra'] == 'labelsmoothing':
                criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
            else:
                criterion = nn.CrossEntropyLoss(class_weights)
        elif config_params['classimbalance'] == 'wtcostfunc':
            class_weights = utils.class_distribution_weightedloss(df).to(config_params['device'])
            criterion = nn.CrossEntropyLoss(class_weights)
        else:
            if config_params['extra']=='labelsmoothing':
                criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            else:
                criterion = nn.CrossEntropyLoss()
    return criterion

def loss_fn_bce(config_params, df, test_bool):
    if test_bool or not config_params['classimbalance']:
        criterion = nn.BCEWithLogitsLoss()

    elif config_params['classimbalance'] == 'poswt':
        class_weights = utils.class_distribution_poswt(df).to(config_params['device'])
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[0])
    
    return criterion

def loss_fn_gmic_initialize(config_params, df, test_bool):
    if test_bool or not config_params['classimbalance']:
        bcelogitloss = nn.BCEWithLogitsLoss()
        bceloss = nn.BCELoss()
    
    elif config_params['classimbalance'] == 'poswt':
        class_weights = utils.class_distribution_poswt(df).to(config_params['device'])
        bcelogitloss = nn.BCEWithLogitsLoss(pos_weight=class_weights[0])
        bceloss = nn.BCELoss()
    
    return bcelogitloss, bceloss

def loss_fn_gmic(config_params, bcelogitsloss, bceloss, y_local, y_global, y_fusion, saliency_map, y_true, df, test_bool):
    if config_params['classimbalance'] == 'poswt' and not test_bool:
        class_weights = utils.class_distribution_poswt(df).to(config_params['device'])
        weight_batch = torch.tensor([1,class_weights[0]]).float().to(config_params['device'])[y_true]
        bceloss.weight = weight_batch
    local_network_loss = bcelogitsloss(y_local, y_true)
    global_network_loss = bceloss(y_global, y_true)
    fusion_network_loss = bcelogitsloss(y_fusion, y_true)
    saliency_map_regularizer = torch.mean(LA.norm(saliency_map.view(saliency_map.shape[0],saliency_map.shape[1],-1), ord=1, dim=2))
    total_loss = local_network_loss + global_network_loss + fusion_network_loss + config_params['sm_reg_param']*saliency_map_regularizer
    return total_loss