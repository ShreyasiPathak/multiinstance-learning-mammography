import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision import models
import resnetkim
import resnet
import densenet
import gmic
import math
import torchvision
from torchvision.models.resnet import BasicBlock
import warnings

views_allowed=['LCC','LMLO','RCC','RMLO']

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

class Attention(nn.Module):
    def __init__(self, L, D, K):
        super(Attention, self).__init__()
        self.L = L
        self.D = D
        self.K = K

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
    
    def forward(self, x):
        #x = x.squeeze(0)
        
        if len(x.shape)==5:
            H = x.view(-1, x.shape[1], x.shape[2] * x.shape[3] * x.shape[4])
        elif len(x.shape)==4:
            H = x.view(-1, x.shape[1], x.shape[2] * x.shape[3])
        elif len(x.shape)==3:
            H = x.view(-1, x.shape[1], x.shape[2])

        A = self.attention(H)  # NxK
        return A, H

class GatedAttention(nn.Module):
    def __init__(self, L, D, K):
        super(GatedAttention, self).__init__()
        self.L = L#500
        self.D = D#128
        self.K = K#1
        
        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x):
        #x = x.squeeze(0)

        if len(x.shape)==5:
            H = x.view(-1, x.shape[1], x.shape[2] * x.shape[3] * x.shape[4])
        if len(x.shape)==4:
            H = x.view(-1, x.shape[1], x.shape[2] * x.shape[3])
        elif len(x.shape)==3:
            H = x.view(-1, x.shape[1], x.shape[2])
        
        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK

        return A, H

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class SILmodel(nn.Module):
    def __init__(self, activation, featureextractormodel, extra, topkpatch, gmic_parameters):
        super(SILmodel, self).__init__()
        self.activation = activation
        self.featureextractormodel = featureextractormodel
        self.extra = extra
        self.topkpatch = topkpatch
        
        if featureextractormodel:
            print(featureextractormodel)
            if 'resnet18pretrained' in featureextractormodel:
                self.feature_extractor = resnet.resnet18(pretrained=True, inchans=3, activation=activation, topkpatch = self.topkpatch)
            elif 'resnet34pretrained' in featureextractormodel:
                self.feature_extractor = resnet.resnet34(pretrained=True, inchans=3, activation=activation, topkpatch = self.topkpatch)
            elif 'resnet50pretrained' in featureextractormodel:
                self.feature_extractor = resnet.resnet50(pretrained=True, inchans=3, activation=activation, topkpatch = self.topkpatch)
            elif 'resnet18scratch' in featureextractormodel:
                self.feature_extractor = resnet.resnet18(pretrained=False, inchans=1, activation=activation, topkpatch = self.topkpatch)
            elif 'resnet50scratch' in featureextractormodel:
                self.feature_extractor = resnet.resnet50(pretrained=False, inchans=1, activation=activation, topkpatch = self.topkpatch)
            elif 'densenet121' in featureextractormodel:
                self.feature_extractor = densenet.densenet121(pretrained=True, activation=activation, extra=self.extra, topkpatch = self.topkpatch)
            elif 'densenet169' in featureextractormodel:
                self.feature_extractor = densenet.densenet169(pretrained=True, activation=activation, extra=self.extra, topkpatch = self.topkpatch)
            elif 'kim' in featureextractormodel:
                self.feature_extractor = resnetkim.resnet18_features(activation=self.activation, extra = self.extra)
            elif 'convnext-T' in featureextractormodel:
                self.feature_extractor = torchvision.models.convnext_tiny(weights=torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
                self.feature_extractor.classifier = Identity()
            elif 'gmic_resnet18_pretrained' in featureextractormodel:
                self.feature_extractor = gmic._gmic(gmic_parameters)
        
        if 'kimgap' in featureextractormodel:
            self.adaptiveavgpool = nn.AdaptiveAvgPool2d((1, 1)) #(1,1) is the output dimension of the result
    
    def forward(self, x):
        if self.featureextractormodel=='gmic_resnet18_pretrained':
            y_local, y_global, y_fusion, saliency_map = self.feature_extractor(x)
            return y_local, y_global, y_fusion, saliency_map
        else:
            M = self.feature_extractor(x)
            if 'kimgap' in self.featureextractormodel:
                M = self.adaptiveavgpool(M)
            M = M.view(M.shape[0],-1)
            print(M.shape)
            return M