import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision import models


import torchvision
from torchvision.models.resnet import BasicBlock
import warnings

from models import resnetkim, resnet, densenet, gmic

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
    def __init__(self, config_params):
        super(SILmodel, self).__init__()
        self.activation = config_params['activation']
        self.featureextractormodel = config_params['femodel']
        self.extra = config_params['extra']
        self.topkpatch = config_params['topkpatch']
        self.pretrained = config_params['pretrained']
        self.channel = config_params['channel']
        self.regionpooling = config_params['regionpooling']
        
        if self.featureextractormodel:
            print(self.featureextractormodel)
            if 'resnet18' in self.featureextractormodel:
                self.feature_extractor = resnet.resnet18(pretrained = self.pretrained, inchans = self.channel, activation = self.activation, topkpatch = self.topkpatch, regionpooling = self.regionpooling)
            elif 'resnet34' in self.featureextractormodel:
                self.feature_extractor = resnet.resnet34(pretrained = self.pretrained, inchans = self.channel, activation = self.activation, topkpatch = self.topkpatch, regionpooling = self.regionpooling)
            elif 'resnet50' in self.featureextractormodel:
                self.feature_extractor = resnet.resnet50(pretrained = self.pretrained, inchans = self.channel, activation = self.activation, topkpatch = self.topkpatch, regionpooling = self.regionpooling)
            elif 'densenet121' in self.featureextractormodel:
                self.feature_extractor = densenet.densenet121(pretrained = self.pretrained, activation = self.activation, topkpatch = self.topkpatch, regionpooling = self.regionpooling)
            elif 'densenet169' in self.featureextractormodel:
                self.feature_extractor = densenet.densenet169(pretrained = self.pretrained, activation = self.activation, topkpatch = self.topkpatch, regionpooling = self.regionpooling)
            elif 'kim' in self.featureextractormodel:
                self.feature_extractor = resnetkim.resnet18_features(activation = self.activation, extra = self.extra)
            elif 'convnext-T' in self.featureextractormodel:
                self.feature_extractor = torchvision.models.convnext_tiny(weights=torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
                self.feature_extractor.classifier[2] = nn.Linear(768, 1)
            elif 'gmic_resnet18' in self.featureextractormodel:
                self.feature_extractor = gmic._gmic(config_params['gmic_parameters'])
        
        if 'kimgap' in self.featureextractormodel:
            self.adaptiveavgpool = nn.AdaptiveAvgPool2d((1, 1)) #(1,1) is the output dimension of the result
    
    def forward(self, x):
        if self.featureextractormodel=='gmic_resnet18':
            y_local, y_global, y_fusion, saliency_map = self.feature_extractor(x)
            return y_local, y_global, y_fusion, saliency_map
        else:
            M = self.feature_extractor(x)
            if 'kimgap' in self.featureextractormodel:
                M = self.adaptiveavgpool(M)
            M = M.view(M.shape[0],-1)
            print(M.shape)
            return M

class MILmodel(nn.Module):
    '''Breast wise separate pipeline
    '''
    def __init__(self, config_params):
        super(MILmodel, self).__init__()
        
        self.milpooling = config_params['milpooling']
        self.activation = config_params['activation']
        self.attention = config_params['attention']
        self.device = config_params['device']
        self.featureextractormodel = config_params['femodel']
        self.numclasses = config_params['numclasses']

        self.D = 128
        self.K = 1
        if 'resnet18' in self.featureextractormodel:
            self.L = 512#2500
        elif 'resnet50' in self.featureextractormodel:
            self.L = 2048
        elif 'convnext-T' in self.featureextractormodel:
            self.L = 768
        
        if self.milpooling=='isatt' or self.milpooling=='esatt': 
            if self.attention=='imagewise':
                self.model_attention = Attention(self.L, self.D, self.K)
            elif self.attention=='breastwise':
                self.model_attention_perbreast = Attention(self.L, self.D, self.K)
                self.model_attention_both = Attention(self.L, self.D, self.K)
            self.classifier = nn.Linear(self.L*self.K, self.numclasses)
        
        elif self.milpooling=='isgatt' or self.milpooling=='esgatt':
            if self.attention=='imagewise':
                self.model_attention = GatedAttention(self.L, self.D, self.K)
            elif self.attention=='breastwise':
                self.model_attention_perbreast = GatedAttention(self.L, self.D, self.K)
                self.model_attention_both = GatedAttention(self.L, self.D, self.K)
            self.classifier = nn.Linear(self.L*self.K, self.numclasses)
        
        elif self.milpooling=='ismean' or self.milpooling=='esmean' or self.milpooling=='ismax' or self.milpooling=='esmax':
            self.classifier = nn.Linear(self.L, self.numclasses)

        self.four_view_resnet = FourViewResNet(config_params)

    def breast_wise_attention(self, views_names, h_view):
        if 'LMLO' in views_names and 'LCC' in views_names:
            h_view['LCC'] = torch.unsqueeze(h_view['LCC'],1) #shape=Nx1x256 #shape=Nx1x2x25x25
            h_view['LMLO'] = torch.unsqueeze(h_view['LMLO'],1) #shape=Nx1x2x25x25
            h_left = torch.cat((h_view['LCC'],h_view['LMLO']),dim=1) #shape=Nx2(views)x2(b/m)x25x25
            A_left, h_left= self.attention_weights(h_left) #Nx256, #Nx2xL
            h_left = self.MILPooling_attention(A_left, h_left)

        elif 'LCC' in views_names:
            h_left = h_view['LCC'] #shape=Nx256
            h_left = self.reshape(h_left) #shape=Nx2xL
            
        elif 'LMLO' in views_names:
            h_left = h_view['LMLO']
            h_left = self.reshape(h_left) #shape=Nx2xL
            
        else:
            h_left = torch.zeros(size=(0,1),device=self.device)
        
        if 'RMLO' in views_names and 'RCC' in views_names:
            h_view['RCC'] = torch.unsqueeze(h_view['RCC'],1) #shape=Nx1x2x25x25
            h_view['RMLO'] = torch.unsqueeze(h_view['RMLO'],1) #shape=Nx1x2x25x25
            h_right = torch.cat((h_view['RCC'],h_view['RMLO']),dim=1) #shape=Nx2x2x25x25
            A_right, h_right = self.attention_weights(h_right) #shape=Nx2xL
            h_right = self.MILPooling_attention(A_right, h_right)
            
        elif 'RCC' in views_names:
            h_right = h_view['RCC']
            h_right = self.reshape(h_right) #shape=Nx2xL
            
        elif 'RMLO' in views_names:
            h_right = h_view['RMLO']
            h_right = self.reshape(h_right) #shape=Nx2xL
            
        else:
            h_right = torch.zeros(size=(0,1),device=self.device)
        
        if len(h_left) and len(h_right):
            h_left = torch.unsqueeze(h_left,1) #shape=Nx1x256 #shape=Nx1x2xL
            h_right = torch.unsqueeze(h_right,1) #shape=Nx1x256 #shape=Nx1x2xL
            h_both = torch.cat((h_left,h_right),dim=1) #shape=Nx2x2xL
            A_final, h_final = self.attention_weights(h_both, both_breast=True) #shape=Nx2xL
            h_final = self.MILPooling_attention(A_final, h_final)
        
        elif len(h_left):
            h_final = h_left
        
        elif len(h_right):
            h_final = h_right
        
        return h_final

    def MILPooling_attention(self, A, H):
        ''' Attention Pooling
        '''
        #print(A)
        A = torch.transpose(A, 2, 1)  # KxN 10,2,1->10,1,2 #Nx4x1->Nx1x4
        #print(A)
        A = F.softmax(A, dim=2)  # softmax over 4
        #print(A)
        M = torch.bmm(A, H)  # KxL 10,1,1250 #Nx1x4 x Nx4x625 -> Nx1x625
        #print(M)
        M = M.squeeze(1) # Nx625
        return M
    
    def MILPooling_maxpool(self, x, views_names, activation):
        '''Max pooling based on bag size
        '''
        x = x.view(x.shape[0],x.shape[1],x.shape[2]) #Nx4x2
        if activation == 'softmax':
            #print(np.unique(np.array(bag_size_list))[0])
            x = torch.cat((x[:,:,0].view(-1,1,len(views_names)),x[:,:,1].view(-1,1,len(views_names))),dim=1) #Nx2x4
            #print(x)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fxn()
                max_val, max_id = F.max_pool2d(x, kernel_size=(x.shape[1],x.shape[2]), return_indices=True) #Nx2x1
            max_id = torch.remainder(max_id, len(views_names))
            max_id = max_id.repeat(1,2,1)
            x = torch.gather(x,2,max_id)
        elif activation == 'sigmoid':
            x = x.view(x.shape[0],x.shape[2],x.shape[1])
            x = F.max_pool1d(x, kernel_size = len(views_names))
        return x
    
    def MILPooling_average(self, x, views_names, activation):
        '''Average pooling based on bag size
        '''
        x = x.view(x.shape[0],x.shape[1],x.shape[2]) #Nx4x2
        if activation == 'softmax':
            x = torch.cat((x[:,:,0].view(-1,1,len(views_names)),x[:,:,1].view(-1,1,len(views_names))),dim=1) #Nx2x4
        elif activation == 'sigmoid':
            x = x.view(x.shape[0],x.shape[2],x.shape[1])
        x = torch.mean(x, dim=2)
        #x = F.avg_pool1d(x,kernel_size=np.unique(np.array(bag_size_list))[0]) #Nx2x1
        return x
    
    def attention_weights(self, h, both_breast=False):
        if self.attention=='breastwise':
            if both_breast:
                A, H = self.model_attention_both(h)
            else:
                A, H = self.model_attention_perbreast(h)
        elif self.attention=='imagewise':
            A, H = self.model_attention(h) #Nx4xL
        return A, H
    
    def classifier_score(self, M):
        if self.activation=='softmax':
            M = self.classifier(M)
        elif self.activation=='sigmoid':
            if self.extra == 'fclayerreduction':
                if len(M.shape)==5:
                    M = self.reshape(M)
                    M = self.classifier(M)
            else:
                M = self.adaptiveavgpool(M)#Nx625->Nx1
        return M
    
    def reshape(self, x):
        if len(x.shape)==5:
            x = x.view(-1, x.shape[1], x.shape[2] * x.shape[3] * x.shape[4])
        elif len(x.shape)==4:
            x = x.view(-1, x.shape[1], x.shape[2] * x.shape[3])
        #elif len(x.shape)==3:
        #    x = x.view(-1, x.shape[1], x.shape[2])
        return x
    
    def forward(self, x, views_names):
        h = self.four_view_resnet(x, views_names) #feature extractor, h['LCC'].shape=Nx2x25x25
       
        if self.attention=='imagewise':
            for counter, view in enumerate(views_names):
                if counter==0:
                    h_all=h[view].unsqueeze(1)
                else:
                    h_all=torch.cat((h_all, h[view].unsqueeze(1)), dim=1)
            
            if self.milpooling=='isatt' or self.milpooling=='isgatt':
                M = self.classifier_score(h_all)
                if len(views_names)>1:
                    A, _ = self.attention_weights(h_all) #Nx2xL
                    M = self.MILPooling_attention(A, M) #NxL
                M = M.view(M.shape[0],-1)
            
            elif self.milpooling=='ismax':
                M = self.classifier_score(h_all)
                M = self.MILPooling_maxpool(M, views_names, self.activation) #Nx2 or Nx1
                M = M.view(M.shape[0],-1) #Nx2
        
            elif self.milpooling=='ismean':
                M = self.classifier_score(h_all)
                M = self.MILPooling_average(M, views_names, self.activation) #shape=Nx2 or Nx1
                M = M.view(M.shape[0],-1) #Nx2
            
            elif self.milpooling=='esatt' or self.milpooling=='esgatt':
                if len(views_names)>1:
                    A, M = self.attention_weights(h_all) #Nx2xL
                    M = self.MILPooling_attention(A, M) #NxL
                else:
                    h_all = h_all.squeeze(1)
                    M = self.reshape(h_all) #Nx2xL
                M = self.classifier_score(M) #Nx2x1
                M = M.view(M.shape[0],-1)
            
            elif self.milpooling=='esmean':
                for counter, view in enumerate(views_names):
                    if counter==0:
                        h_all=h[view]
                    else:
                        h_all=h_all.add(h[view])
                h_all = h_all/len(views_names)
                M = self.classifier_score(h_all) 
                M = M.view(M.shape[0],-1)
        
            elif self.milpooling=='esmax':
                h_all, _ = torch.max(h_all, dim=1)
                M = self.classifier_score(h_all) 
                M = M.view(M.shape[0],-1)
            
            elif self.milpooling=='essum':
                for counter, view in enumerate(views_names):
                    if counter==0:
                        h_all=h[view]
                    else:
                        h_all=h_all.add(h[view])
                M = self.classifier_score(h_all) 
                M = M.view(M.shape[0],-1)
            
        elif self.attention=='breastwise':
            if self.milpooling=='esatt' or self.milpooling=='esgatt':
                M = self.breast_wise_attention(views_names, h) #shape=Nx2xL
                M = self.classifier_score(M) #shape=Nx2x1
                M = M.view(M.shape[0],-1)
        
            elif self.milpooling=='essum':
                if 'LCC' in views_names and 'LMLO' in views_names:
                    #h['LCC'] = h['LCC']*0.5
                    #h['LMLO'] = h['LMLO']*0.5
                    h_left = h['LCC'].add(h['LMLO']) #Nx2x25x25
                elif 'LCC' in views_names:
                    h_left = h['LCC']
                elif 'LMLO' in views_names:
                    h_left = h['LMLO']
                else:
                    h_left = torch.zeros(size=(0,1),device=self.device)
                
                if 'RCC' in views_names and 'RMLO' in views_names:
                    #h['RCC'] = h['RCC']*0.5
                    #h['RMLO'] = h['RMLO']*0.5
                    h_right = h['RCC'].add(h['RMLO']) #Nx2x25x25
                elif 'RCC' in views_names:
                    h_right = h['RCC']
                elif 'RMLO' in views_names:
                    h_right = h['RMLO']
                else:
                    h_right = torch.zeros(size=(0,1),device=self.device)
                
                if len(h_left) and len(h_right):
                    h_left_score = self.classifier(h_left) #Nx625
                    h_right_score = self.classifier(h_right) #Nx2
                    h_score = torch.cat((h_left_score.unsqueeze(1),h_right_score.unsqueeze(1)),dim=1) #Nx2x2
                    M = torch.mean(h_score,dim=1) #Nx2
                elif len(h_left):
                    M = self.classifier(h_left) #Nx2
                elif len(h_right):
                    M = self.classifier(h_right) #Nx2
        print(M.shape)
        return M

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class FourViewResNet(nn.Module):
    def __init__(self, config_params):
        super(FourViewResNet, self).__init__()

        self.activation = config_params['activation']
        self.featureextractormodel = config_params['femodel']
        self.extra = config_params['extra']
        self.topkpatch = config_params['topkpatch']
        self.pretrained = config_params['pretrained']
        self.channel = config_params['channel']
        self.regionpooling = config_params['regionpooling']

        if 'resnet18' in self.featureextractormodel:
            self.feature_extractor = resnet.resnet18(pretrained = self.pretrained, inchans = self.channel, activation = self.activation, topkpatch = self.topkpatch, regionpooling = self.regionpooling)
        elif 'resnet34' in self.featureextractormodel:
            self.feature_extractor = resnet.resnet34(pretrained = self.pretrained, inchans = self.channel, activation = self.activation, topkpatch = self.topkpatch, regionpooling = self.regionpooling)
        elif 'resnet50' in self.featureextractormodel:
            self.feature_extractor = resnet.resnet50(pretrained = self.pretrained, inchans = self.channel, activation = self.activation, topkpatch = self.topkpatch, regionpooling = self.regionpooling)
        elif 'densenet121' in self.featureextractormodel:
            self.feature_extractor = densenet.densenet121(pretrained = self.pretrained, activation = self.activation, topkpatch = self.topkpatch, regionpooling = self.regionpooling)
        elif 'densenet169' in self.featureextractormodel:
            self.feature_extractor = densenet.densenet169(pretrained = self.pretrained, activation = self.activation, topkpatch = self.topkpatch, regionpooling = self.regionpooling)
        elif 'kim' in self.featureextractormodel:
            self.feature_extractor = resnetkim.resnet18_features(activation = self.activation, extra = self.extra)
        elif 'convnext-T' in self.featureextractormodel:
            self.feature_extractor = torchvision.models.convnext_tiny(weights = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            self.feature_extractor.classifier = Identity()
            #self.feature_extractor.classifier[2] = nn.Linear(768, 1)
        elif 'gmic_resnet18' in self.featureextractormodel:
            self.feature_extractor = gmic._gmic(config_params['gmic_parameters'])

    def forward(self, x, views_names):
        if len(x.shape) == 5:
            h_dict = {
                view: self.single_forward(x[:,views_names.index(view),:,:,:], view)
                for view in views_names
            }
        elif len(x.shape) == 4:
            h_dict = {
                view: self.single_forward(x[:,views_names.index(view),:,:].unsqueeze(1), view)
                for view in views_names
            }
        return h_dict

    def single_forward(self, single_view):
        single_view_feature = self.feature_extractor(single_view)
        single_view_feature = single_view_feature.view(single_view_feature.shape[0], single_view_feature.shape[1])
        #print(single_view_feature.shape)
        return single_view_feature