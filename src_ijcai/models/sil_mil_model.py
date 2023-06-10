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
        self.learningtype = config_params['learningtype']
        
        if self.featureextractormodel:
            print(self.featureextractormodel)
            if self.featureextractormodel == 'resnet18':
                self.feature_extractor = resnet.resnet18(pretrained = self.pretrained, inchans = self.channel, activation = self.activation, topkpatch = self.topkpatch, regionpooling = self.regionpooling)
            elif self.featureextractormodel == 'resnet34':
                self.feature_extractor = resnet.resnet34(pretrained = self.pretrained, inchans = self.channel, activation = self.activation, topkpatch = self.topkpatch, regionpooling = self.regionpooling)
            elif self.featureextractormodel == 'resnet50':
                self.feature_extractor = resnet.resnet50(pretrained = self.pretrained, inchans = self.channel, activation = self.activation, topkpatch = self.topkpatch, regionpooling = self.regionpooling)
            elif self.featureextractormodel == 'densenet121':
                self.feature_extractor = densenet.densenet121(pretrained = self.pretrained, activation = self.activation, topkpatch = self.topkpatch, regionpooling = self.regionpooling)
            elif self.featureextractormodel == 'densenet169':
                self.feature_extractor = densenet.densenet169(pretrained = self.pretrained, activation = self.activation, topkpatch = self.topkpatch, regionpooling = self.regionpooling)
            elif self.featureextractormodel == 'kim':
                self.feature_extractor = resnetkim.resnet18_features(activation = self.activation, learningtype = self.learningtype)
            elif self.featureextractormodel == 'convnext-T':
                self.feature_extractor = torchvision.models.convnext_tiny(weights=torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
                self.feature_extractor.classifier[2] = nn.Linear(768, 1)
            elif self.featureextractormodel == 'gmic_resnet18':
                self.feature_extractor = gmic._gmic(config_params['gmic_parameters'])
    
    def forward(self, x):
        if self.featureextractormodel=='gmic_resnet18':
            y_local, y_global, y_fusion, saliency_map = self.feature_extractor(x)
            return y_local, y_global, y_fusion, saliency_map
        else:
            M = self.feature_extractor(x)
            M = M.view(M.shape[0],-1)
            print(M.shape)
            return M

class SelfAttention(nn.Module):
    def __init__(self, L, D, config_params):
        super(SelfAttention, self).__init__()
        self.L = L
        self.D = D
        self.sqrt_dim = np.sqrt(self.D)
        self.gamma = config_params['selfatt-gamma']

        if config_params['selfatt-nonlinear']:
            self.attention_self_query = nn.Sequential(
                nn.Linear(self.L, self.D),
                nn.Tanh()
            )
            self.attention_self_key = nn.Sequential(
                nn.Linear(self.L, self.D),
                nn.Tanh()
            )
        else:
            self.attention_self_query = nn.Linear(self.L, self.D)
            self.attention_self_key = nn.Linear(self.L, self.D)
        
        self.attention_self_value = nn.Linear(self.L, self.L)

        if self.gamma:
            self.attention_self_gamma = nn.Parameter((torch.zeros(1)).to(config_params['device']))

    def forward(self, x):
        #print('x:', x.shape)
        q = self.attention_self_query(x)
        #print('q:', q.shape)
        k = self.attention_self_key(x)
        #print('k:', k.shape)
        v = self.attention_self_value(x)
        #print('v:', v.shape)
        #print(q)
        #print(k)
        #print(v)

        score = torch.bmm(q, k.transpose(1, 2)) / self.sqrt_dim #NxVxD x NxDxV -> NxVXV
        #print(score)
        #print('score:', score.shape)
        A = F.softmax(score, -1) #NxVxV
        context = torch.bmm(A, v) #NxVxV x NxVxL -> NxVxL
        if self.gamma:
            out = self.attention_self_gamma * context + x
        else:
            out = context
        #print(context)
        #print('context:', context.shape)
        return out

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
    
class MILpooling(nn.Module):
    def __init__(self, config_params):
        super(MILpooling, self).__init__()
        self.milpooling = config_params['milpooling']
        self.attention = config_params['attention']
        self.activation = config_params['activation']
        self.numclasses = config_params['numclasses']
        self.device = config_params['device']
        self.featureextractormodel = config_params['femodel']
        self.dependency = config_params['dependency']
        print("in mil pooling 2:", self.featureextractormodel)
        self.D = 128
        self.K = 1
        if self.featureextractormodel == 'resnet18':
            self.L = 512#2500
        elif self.featureextractormodel == 'resnet50':
            self.L = 2048
        elif self.featureextractormodel == 'densenet169':
            self.L = 1664
        elif self.featureextractormodel == 'convnext-T':
            self.L = 768
        elif self.featureextractormodel == 'gmic_resnet18':
            self.L_local = 512
            self.L_global = int(round(config_params['gmic_parameters']['cam_size'][0]*config_params['gmic_parameters']['cam_size'][1]*config_params['gmic_parameters']['percent_t']))
            self.L_fusion = config_params['gmic_parameters']['post_processing_dim']+512
        
        if self.featureextractormodel == 'gmic_resnet18':
            if self.dependency=='selfatt':
                self.model_selfattention_local_img = SelfAttention(self.L_local, int(self.L_local/4), config_params)
                self.model_selfattention_global_img = SelfAttention(self.L_global, self.L_global, config_params)
                self.model_selfattention_fusion_img = SelfAttention(self.L_fusion, int(self.L_fusion/4), config_params)

            if self.milpooling=='isatt' or self.milpooling=='esatt': 
                if self.attention=='imagewise':
                    self.model_attention_local_img = Attention(self.L_local, self.D, self.K)
                    self.model_attention_global_img = Attention(self.L_global, 50, self.K)
                    self.model_attention_fusion_img = Attention(self.L_fusion, self.D, self.K)
                elif self.attention=='breastwise':
                    self.model_attention_local_img = Attention(self.L_local, self.D, self.K)
                    self.model_attention_global_img = Attention(self.L_global, 50, self.K)
                    self.model_attention_fusion_img = Attention(self.L_fusion, self.D, self.K)
                    self.model_attention_local_side = Attention(self.L_local, self.D, self.K)
                    self.model_attention_global_side = Attention(self.L_global, 50, self.K)
                    self.model_attention_fusion_side = Attention(self.L_fusion, self.D, self.K)
            
            elif self.milpooling=='isgatt' or self.milpooling=='esgatt':
                if self.attention=='imagewise':
                    self.model_attention_local_img = GatedAttention(self.L_local, self.D, self.K)
                    self.model_attention_global_img = GatedAttention(self.L_global, 50, self.K)
                    self.model_attention_fusion_img = GatedAttention(self.L_fusion, self.D, self.K)
                elif self.attention=='breastwise':
                    self.model_attention_local_img = GatedAttention(self.L_local, self.D, self.K)
                    self.model_attention_global_img = GatedAttention(self.L_global, 50, self.K)
                    self.model_attention_fusion_img = GatedAttention(self.L_fusion, self.D, self.K)
                    self.model_attention_local_side = GatedAttention(self.L_local, self.D, self.K)
                    self.model_attention_global_side = GatedAttention(self.L_global, 50, self.K)
                    self.model_attention_fusion_side = GatedAttention(self.L_fusion, self.D, self.K)
            
            self.classifier_local = nn.Linear(self.L_local, config_params['gmic_parameters']["num_classes"], bias=False)
            self.classifier_fusion = nn.Linear(self.L_fusion, config_params['gmic_parameters']["num_classes"])

        else:
            print("in mil pooling 1:", self.featureextractormodel)
            if self.milpooling=='isatt' or self.milpooling=='esatt': 
                if self.attention=='imagewise':
                    self.model_attention_img = Attention(self.L, self.D, self.K)
                elif self.attention=='breastwise':
                    self.model_attention_img = Attention(self.L, self.D, self.K)
                    self.model_attention_side = Attention(self.L, self.D, self.K)
                self.classifier = nn.Linear(self.L*self.K, self.numclasses)
            
            elif self.milpooling=='isgatt' or self.milpooling=='esgatt':
                if self.attention=='imagewise':
                    self.model_attention_img = GatedAttention(self.L, self.D, self.K)
                elif self.attention=='breastwise':
                    self.model_attention_img = GatedAttention(self.L, self.D, self.K)
                    self.model_attention_side = GatedAttention(self.L, self.D, self.K)
                self.classifier = nn.Linear(self.L*self.K, self.numclasses)
            
            elif self.milpooling=='ismean' or self.milpooling=='esmean' or self.milpooling=='ismax' or self.milpooling=='esmax':
                print("in mil pooling", self.featureextractormodel)
                if self.featureextractormodel == 'kim':
                    self.classifier = nn.AdaptiveAvgPool2d((1, 1))
                else:
                    self.classifier = nn.Linear(self.L, self.numclasses)  
    
    def reshape(self, x):
        if len(x.shape)==5:
            x = x.view(-1, x.shape[1], x.shape[2] * x.shape[3] * x.shape[4])
        elif len(x.shape)==4:
            x = x.view(-1, x.shape[1], x.shape[2] * x.shape[3])
        #elif len(x.shape)==3:
        #    if x.shape[1] == 1:
        #        x = x.view(-1, x.shape[1] * x.shape[2])
        return x

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
        #M = M.squeeze(1) # Nx625
        return M
    
    def MILPooling_ISMax(self, x, views_names, activation):
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
    
    def MILPooling_ISMean(self, x, views_names, activation):
        '''Average pooling based on bag size
        '''
        #print("start:", x.shape)
        x = x.view(x.shape[0],x.shape[1],x.shape[2]) #Nx4x2/ Nx4x1
        #print('x.shape:', x.shape)
        if activation == 'softmax':
            x = torch.cat((x[:,:,0].view(-1,1,len(views_names)),x[:,:,1].view(-1,1,len(views_names))),dim=1) #Nx2x4
        elif activation == 'sigmoid':
            x = x.view(x.shape[0],x.shape[2],x.shape[1])
        #print('again x.shape:', x.shape)
        x = torch.mean(x, dim=2)
        #x = F.avg_pool1d(x,kernel_size=np.unique(np.array(bag_size_list))[0]) #Nx2x1
        return x

    def attention_weights(self, featuretype, h, both_breast=False):
        if self.featureextractormodel == 'gmic_resnet18':
            if self.attention=='breastwise':
                if both_breast:
                    if featuretype == 'local':
                        A, H = self.model_attention_local_side(h)
                    elif featuretype == 'global':
                        A, H = self.model_attention_global_side(h)
                    elif featuretype == 'fusion':
                        A, H = self.model_attention_fusion_side(h)
                else:
                    if featuretype == 'local':
                        A, H = self.model_attention_local_img(h)
                    elif featuretype == 'global':
                        A, H = self.model_attention_global_img(h)
                    elif featuretype == 'fusion':
                        A, H = self.model_attention_fusion_img(h)
                    
            elif self.attention=='imagewise':
                if featuretype == 'local':
                    A, H = self.model_attention_local_img(h) #Nx4xL
                elif featuretype == 'global':
                    A, H = self.model_attention_global_img(h) #Nx4xL
                elif featuretype == 'fusion':
                    A, H = self.model_attention_fusion_img(h) #Nx4xL
        else:
            if self.attention=='breastwise':
                if both_breast:
                    A, H = self.model_attention_side(h)
                else:
                    A, H = self.model_attention_img(h)
            elif self.attention=='imagewise':
                A, H = self.model_attention_img(h) #Nx4xL
        return A, H
    
    def classifier_score(self, featuretype, M):
        if self.featureextractormodel == 'gmic_resnet18':
            if featuretype == 'local':
                M = self.classifier_local(M)  
            elif featuretype == 'global':
                M = M.mean(dim=2)
                M = M.unsqueeze(2)
            elif featuretype == 'fusion':
                M = self.classifier_fusion(M)
        else:
            M = self.classifier(M)
        #print("M.shape:",M.shape)
        return M

    def ISMean(self, featuretype, h_all, views_names):
        #print("h_all:",h_all.shape)
        #print(featuretype)
        #print(h_all.shape)
        M = self.classifier_score(featuretype, h_all)
        #print("1:",M.shape)
        M = self.MILPooling_ISMean(M, views_names, self.activation) #shape=Nx2 or Nx1
        #print("2:",M.shape)
        M = M.view(M.shape[0],-1) #Nx2
        #print(M.shape)
        return M
    
    def ISMax(self, featuretype, h_all, views_names):
        #print(featuretype)
        #print(h_all.shape)
        M = self.classifier_score(featuretype, h_all)
        #print(M.shape)
        M = self.MILPooling_ISMax(M, views_names, self.activation) #Nx2 or Nx1
        #print(M.shape)
        M = M.view(M.shape[0],-1) #Nx2
        #print(M.shape)
        return M
    
    def ISAtt(self, featuretype, h_all, views_names): #same for ISGatt
        M = self.classifier_score(featuretype, h_all)
        if len(views_names)>1:
            A, _ = self.attention_weights(featuretype, h_all) #Nx2xL
            M = self.MILPooling_attention(A, M) #NxL
        M = M.view(M.shape[0],-1)
        return M
    
    def ESMean(self, featuretype, h_all, views_names):
        h_all = torch.sum(h_all, dim = 1)/len(views_names)
        h_all = h_all.unsqueeze(1)
        M = self.classifier_score(featuretype, h_all) 
        M = M.view(M.shape[0],-1)
        return M
    
    def ESMax(self, featuretype, h_all):
        h_all, _ = torch.max(h_all, dim=1)
        h_all = h_all.unsqueeze(1)
        M = self.classifier_score(featuretype, h_all) 
        M = M.view(M.shape[0],-1)
        return M
    
    def ESAtt(self, featuretype, h_all, views_names):
        if len(views_names)>1:
            #print('h_all:', h_all.shape)
            A, M = self.attention_weights(featuretype, h_all) #Nx2xL
            M = self.MILPooling_attention(A, M) #NxL
        else:
            #h_all = h_all.squeeze(1)
            M = self.reshape(h_all) #Nx2xL
        #print('M:', M.shape)
        M = self.classifier_score(featuretype, M) #Nx2x1
        M = M.view(M.shape[0],-1)
        return M
    
    def ESSum(self, featuretype, h_all):
        h_all = torch.sum(h_all, dim = 1)
        M = self.classifier_score(featuretype, h_all) 
        M = M.view(M.shape[0],-1)
        return M

    def ESAtt_breastwise(self, featuretype, h_view, views_names):
        if featuretype == 'local':
            idx_h = 0
        elif featuretype == 'global':
            idx_h = 1
        elif featuretype == 'fusion':
            idx_h = 2

        views_names_left=np.array([view for view in views_names if view[0]=='L'])
        views_names_left=views_names_left.tolist()
        
        views_names_right=np.array([view for view in views_names if view[0]=='R'])
        views_names_right=views_names_right.tolist()

        if len(views_names_left)>1:
            for counter, view in enumerate(views_names_left):
                if counter == 0:
                    h_left = h_view[view][idx_h].unsqueeze(1)
                else:
                    h_left = torch.cat((h_left, h_view[view][idx_h].unsqueeze(1)), dim=1)   
            
            if self.dependency=='selfatt':
                if featuretype == 'local':
                    h_left = self.model_selfattention_local_img(h_left)
                elif featuretype == 'global':
                    h_left = self.model_selfattention_global_img(h_left)
                    h_left = torch.sigmoid(h_left)
                elif featuretype == 'fusion':
                    h_left = self.model_selfattention_fusion_img(h_left)
            
            A_left, h_left= self.attention_weights(featuretype, h_left) #Nx2xL
            h_left = self.MILPooling_attention(A_left, h_left) #Nx1xL

        elif len(views_names_left)==1:
            h_left = h_view[views_names_left[0]][idx_h].unsqueeze(1) #Nx1xL
            h_left = self.reshape(h_left)
            
        else:
            h_left = torch.zeros(size=(0,1),device=self.device)
        
        if len(views_names_right)>1:
            for counter1, view in enumerate(views_names_right):
                if counter1 == 0:
                    h_right = h_view[view][idx_h].unsqueeze(1)
                else:
                    h_right = torch.cat((h_right, h_view[view][idx_h].unsqueeze(1)), dim=1)   
            
            if self.dependency == 'selfatt':
                if featuretype == 'local':
                    h_right = self.model_selfattention_local_img(h_right)
                elif featuretype == 'global':
                    h_right = self.model_selfattention_global_img(h_right)
                    h_right = torch.sigmoid(h_right)
                elif featuretype == 'fusion':
                    h_right = self.model_selfattention_fusion_img(h_right)
            
            A_right, h_right = self.attention_weights(featuretype, h_right) #Nx2xL
            h_right = self.MILPooling_attention(A_right, h_right) #Nx1xL

        elif len(views_names_right)==1:
            h_right = h_view[views_names_right[0]][idx_h].unsqueeze(1) #Nx1xL
            h_right = self.reshape(h_right) #shape=Nx2xL
            
        else:
            h_right = torch.zeros(size=(0,1),device=self.device)
        
        if len(h_left) and len(h_right):
            h_both = torch.cat((h_left, h_right),dim=1) #shape=Nx2xL
            A_final, h_final = self.attention_weights(featuretype, h_both, both_breast=True) #shape=Nx2xL
            h_final = self.MILPooling_attention(A_final, h_final)
        
        elif len(h_left):
            h_final = h_left
        
        elif len(h_right):
            h_final = h_right
        
        M = self.classifier_score(featuretype, h_final)
        
        return M
    
    def ESSum_breastwise(self, h, views_names):
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
        return M

class MILmodel(nn.Module):
    '''Breast wise separate pipeline
    '''
    def __init__(self, config_params):
        super(MILmodel, self).__init__()
        
        self.milpooling = config_params['milpooling']
        self.attention = config_params['attention']
        self.featureextractormodel = config_params['femodel']
        self.dependency = config_params['dependency']
        print("in mil pooling 3:", self.featureextractormodel)
        self.four_view_resnet = FourViewResNet(config_params)
        self.milpooling_block = MILpooling(config_params)
    
    def capture_views(self, h, views_names):
        for counter, view in enumerate(views_names):
            if self.featureextractormodel == 'gmic_resnet18':
                if counter==0:
                    h_all_local = h[view][0].unsqueeze(1) #Nx1x512
                    h_all_global = h[view][1].unsqueeze(1) #Nx1x110
                    #y_all_global = h[view][2].unsqueeze(1) #Nx1x1024
                    h_all_fusion = h[view][2].unsqueeze(1) #Nx1x1x92x60
                    all_saliency_map = h[view][3].unsqueeze(1)
                else:
                    h_all_local = torch.cat((h_all_local, h[view][0].unsqueeze(1)), dim=1)
                    h_all_global = torch.cat((h_all_global, h[view][1].unsqueeze(1)), dim=1)
                    #y_all_global = torch.cat((y_all_global, h[view][2].unsqueeze(1)), dim=1)
                    h_all_fusion = torch.cat((h_all_fusion, h[view][2].unsqueeze(1)), dim=1)
                    all_saliency_map = torch.cat((all_saliency_map, h[view][3].unsqueeze(1)), dim=1)
            else:
                if counter == 0:
                    h_all = h[view].unsqueeze(1)
                else:
                    h_all = torch.cat((h_all, h[view].unsqueeze(1)), dim=1)
        
        if self.featureextractormodel == 'gmic_resnet18':
            h_all = [h_all_local, h_all_global, h_all_fusion, all_saliency_map]
        
        return h_all
    
    def capture_saliency_map(self, h, views_names):
        for counter, view in enumerate(views_names):
            if self.featureextractormodel == 'gmic_resnet18':
                if counter==0:
                    all_saliency_map = h[view][3].unsqueeze(1)
                else:
                    all_saliency_map = torch.cat((all_saliency_map, h[view][3].unsqueeze(1)), dim=1)
        return all_saliency_map

    def forward(self, x, views_names):
        print(views_names)
        h = self.four_view_resnet(x, views_names) #feature extractor, h['LCC'].shape=Nx2x25x25

        if self.attention=='imagewise':
            h_all = self.capture_views(h, views_names)
            if self.featureextractormodel == 'gmic_resnet18':
                all_saliency_map = h_all[3]

            if self.dependency == 'selfatt':
                if len(views_names)>1:
                    #print('before:',h_all[0].shape)
                    h_all[0] = self.milpooling_block.model_selfattention_local_img(h_all[0])
                    #print('after:', h_all[0].shape)
                    h_all[1] = self.milpooling_block.model_selfattention_global_img(h_all[1])
                    h_all[1] = torch.sigmoid(h_all[1])
                    h_all[2] = self.milpooling_block.model_selfattention_fusion_img(h_all[2])


            if self.milpooling=='isatt' or self.milpooling=='isgatt':
                if self.featureextractormodel == 'gmic_resnet18':
                    y_local = self.milpooling_block.ISAtt('local', h_all[0], views_names)
                    y_global = self.milpooling_block.ISAtt('global', h_all[1], views_names)
                    y_fusion = self.milpooling_block.ISAtt('fusion', h_all[2], views_names)
                else:
                    y_pred = self.milpooling_block.ISAtt(None, h_all, views_names)
            
            elif self.milpooling=='ismax':
                if self.featureextractormodel == 'gmic_resnet18':
                    y_local = self.milpooling_block.ISMax('local', h_all[0], views_names)
                    y_global = self.milpooling_block.ISMax('global', h_all[1], views_names)
                    y_fusion = self.milpooling_block.ISMax('fusion', h_all[2], views_names)
                else:
                    y_pred = self.milpooling_block.ISMax(None, h_all, views_names)
        
            elif self.milpooling=='ismean':
                if self.featureextractormodel == 'gmic_resnet18':
                    y_local = self.milpooling_block.ISMean('local', h_all[0], views_names)
                    y_global = self.milpooling_block.ISMean('global', h_all[1], views_names)
                    y_fusion = self.milpooling_block.ISMean('fusion', h_all[2], views_names)
                else:
                    y_pred = self.milpooling_block.ISMean(None, h_all, views_names)
            
            elif self.milpooling=='esatt' or self.milpooling=='esgatt':
                if self.featureextractormodel == 'gmic_resnet18':
                    #print('h_all in esatt:', h_all[0].shape)
                    y_local = self.milpooling_block.ESAtt('local', h_all[0], views_names)
                    y_global = self.milpooling_block.ESAtt('global', h_all[1], views_names)
                    y_fusion = self.milpooling_block.ESAtt('fusion', h_all[2], views_names)
                else:
                    y_pred = self.milpooling_block.ESAtt(None, h_all, views_names)
            
            elif self.milpooling=='esmean':
                if self.featureextractormodel == 'gmic_resnet18':
                    y_local = self.milpooling_block.ESMean('local', h_all[0], views_names)
                    y_global = self.milpooling_block.ESMean('global', h_all[1], views_names)
                    y_fusion = self.milpooling_block.ESMean('fusion', h_all[2], views_names)
                else:
                    y_pred = self.milpooling_block.ESMean(None, h_all, views_names)
        
            elif self.milpooling=='esmax':
                if self.featureextractormodel == 'gmic_resnet18':
                    y_local = self.milpooling_block.ESMax('local', h_all[0])
                    y_global = self.milpooling_block.ESMax('global', h_all[1])
                    y_fusion = self.milpooling_block.ESMax('fusion', h_all[2])
                else:
                    y_pred = self.milpooling_block.ESMax(None, h_all)
            
        elif self.attention=='breastwise':
            if self.milpooling=='esatt' or self.milpooling=='esgatt':
                if self.featureextractormodel == 'gmic_resnet18':
                    y_local = self.milpooling_block.ESAtt_breastwise('local', h, views_names)
                    y_global = self.milpooling_block.ESAtt_breastwise('global', h, views_names)
                    y_fusion = self.milpooling_block.ESAtt_breastwise('fusion', h, views_names)
                    all_saliency_map = self.capture_saliency_map(h, views_names)
                else:
                    y_pred = self.milpooling_block.ESAtt_breastwise(None, h, views_names)
        
        if self.featureextractormodel == 'gmic_resnet18':
            return y_local, y_global, y_fusion, all_saliency_map
        else:
            return y_pred

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
        self.learningtype = config_params['learningtype']

        if self.featureextractormodel == 'resnet18':
            self.feature_extractor = resnet.resnet18(pretrained = self.pretrained, inchans = self.channel, activation = self.activation, topkpatch = self.topkpatch, regionpooling = self.regionpooling, learningtype = self.learningtype)
        elif self.featureextractormodel == 'resnet34':
            self.feature_extractor = resnet.resnet34(pretrained = self.pretrained, inchans = self.channel, activation = self.activation, topkpatch = self.topkpatch, regionpooling = self.regionpooling, learningtype = self.learningtype)
        elif self.featureextractormodel == 'resnet50':
            self.feature_extractor = resnet.resnet50(pretrained = self.pretrained, inchans = self.channel, activation = self.activation, topkpatch = self.topkpatch, regionpooling = self.regionpooling, learningtype = self.learningtype)
        elif self.featureextractormodel == 'densenet121':
            self.feature_extractor = densenet.densenet121(pretrained = self.pretrained, activation = self.activation, topkpatch = self.topkpatch, regionpooling = self.regionpooling, learningtype = self.learningtype)
        elif self.featureextractormodel == 'densenet169':
            self.feature_extractor = densenet.densenet169(pretrained = self.pretrained, activation = self.activation, topkpatch = self.topkpatch, regionpooling = self.regionpooling, learningtype = self.learningtype)
        elif self.featureextractormodel == 'kim':
            self.feature_extractor = resnetkim.resnet18_features(activation = self.activation, learningtype = self.learningtype)
        elif self.featureextractormodel == 'convnext-T':
            self.feature_extractor = torchvision.models.convnext_tiny(weights = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            self.feature_extractor.classifier = Identity()
            #self.feature_extractor.classifier[2] = nn.Linear(768, 1)
        elif self.featureextractormodel == 'gmic_resnet18':
            self.feature_extractor = gmic._gmic(config_params['gmic_parameters'])

    def forward(self, x, views_names):
        if len(x.shape) == 5:
            h_dict = {
                view: self.single_forward(x[:,views_names.index(view),:,:,:])
                for view in views_names
            }
        elif len(x.shape) == 4:
            h_dict = {
                view: self.single_forward(x[:,views_names.index(view),:,:].unsqueeze(1))
                for view in views_names
            }
        return h_dict

    def single_forward(self, single_view):
        if self.featureextractormodel == 'gmic_resnet18':
            local_feature, topt_feature_global, y_global, fusion_feature, saliency_map = self.feature_extractor(single_view)
            single_view_feature = [local_feature, topt_feature_global.squeeze(1), fusion_feature, saliency_map]
        else:
            single_view_feature = self.feature_extractor(single_view)
            #print(single_view_feature.shape)
            #single_view_feature = single_view_feature.view(single_view_feature.shape[0], single_view_feature.shape[1])
        #print(single_view_feature.shape)
        return single_view_feature