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
                self.feature_extractor = resnet.resnet18(pretrained=True, inchans=3, activation=activation)
            elif 'resnet34pretrained' in featureextractormodel:
                self.feature_extractor = resnet.resnet34(pretrained=True, inchans=3, activation=activation)
            elif 'resnet50pretrained' in featureextractormodel:
                self.feature_extractor = resnet.resnet50(pretrained=True, inchans=3, activation=activation)
            elif 'resnet18scratch' in featureextractormodel:
                self.feature_extractor = resnet.resnet18(pretrained=False, inchans=1, activation=activation)
            elif 'resnet50scratch' in featureextractormodel:
                self.feature_extractor = resnet.resnet50(pretrained=False, inchans=1, activation=activation)
            elif 'densenet121' in featureextractormodel:
                self.feature_extractor = densenet.densenet121(pretrained=True, activation=activation)
            elif 'densenet169' in featureextractormodel:
                self.feature_extractor = densenet.densenet169(pretrained=True, activation=activation, extra=self.extra, topkpatch = self.topkpatch)
            elif 'kim' in featureextractormodel:
                self.feature_extractor = resnetkim.resnet18_features(activation=self.activation, extra = self.extra)
            elif 'convnext-T' in featureextractormodel:
                self.feature_extractor = torchvision.models.convnext_tiny(weights=torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
                self.feature_extractor.classifier[2] = nn.Linear(768, 1)

            elif 'gmic_resnet18_pretrained' in featureextractormodel:
                self.feature_extractor = gmic._gmic(gmic_parameters)
        else:
            self.feature_extractor = resnetkim.resnet18_features(activation=self.activation, extra = self.extra)
        
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
    

class MILmodel(nn.Module):
    def __init__(self, milpooling, activation, device, extra):
        super(MILmodel, self).__init__()
        
        self.L = 1250
        self.L1 = 625
        self.D = 128
        self.K = 1
        self.bag_size=4
        self.milpooling = milpooling
        self.activation = activation
        self.device = device
        
        
        if extra:
            if 'resnet18_pretrained' in extra:
                self.feature_extractor = resnet.resnet18(pretrained=True, inchans=1, wt_combine=extra)
            elif 'wu_resnet22' in extra:
                self.feature_extractor = wu_resnet.resnet22(input_channels=1)
            else:
                self.feature_extractor = resnet_features.resnet18_features()
        else:
            self.feature_extractor = resnet_features.resnet18_features()
        
        if (self.milpooling=='attention' and self.activation=='sigmoid') or (self.milpooling=='gatedattention' and self.activation=='sigmoid'):
            if self.milpooling=='attention':
                self.model_attention = Attention(self.L, self.D, self.K)
            elif self.milpooling=='gated_attention':
                self.model_attention = GatedAttention()
            
            self.classifier = nn.Sequential(
                nn.Linear(self.L*self.K, 1),#next experiment: replace linear with adaptive average pool layer
                nn.Sigmoid())
        
        if (self.milpooling=='attention' and self.activation=='softmax') or (self.milpooling=='gatedattention' and self.activation=='softmax'):
            if self.milpooling=='attention':
                self.model_attention = Attention(self.L1, self.D, self.K)
                #self.model_attention_m = Attention(self.L1, self.D, self.K)
            elif self.milpooling=='gated_attention':
                self.model_attention = GatedAttention()
            
            self.adaptiveavgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        elif self.milpooling=='average' or self.milpooling=='maxpool':
            self.adaptiveavgpool = nn.AdaptiveAvgPool2d((1, 1)) #(1,1) is the output dimension of the result
            #self.avgpool = nn.AvgPool1d(4)
    
    def _create_bag(self, x):
        if x.shape==2:
            x = x.view(-1,self.bag_size,x.shape[1])
        elif x.shape==4:
            x = x.view(-1,4,x.shape[1],x.shape[2],x.shape[3])
        return x
    
    def _flatten_bag(self, x):
        x = x[0]
        print(x)
        x = x.view(x.shape[0]*x.shape[1],x.shape[2],x.shape[3],x.shape[4])
        return x
    
    def MILPooling_attention(self, A, H, bag_size_list):
        ''' Attention Pooling
        '''
        start=0
        for bn in range(len(bag_size_list)):
            end = start+bag_size_list[bn]
            A1 = A[start:end]
            A1 = torch.transpose(A1, 1, 0)  # KxN
            A1 = F.softmax(A1, dim=1)  # softmax over N
            
            H1 = H[start:end] #NxL
            M1 = torch.mm(A1, H1)  # KxL
            M1 = M1.unsqueeze(0)
            if bn==0:
                M=M1
            else:
                M=torch.cat((M,M1),dim=0)
            #M = torch.bmm(A1,H)
            start=end
        print("M shape:",M.shape)
        return M
    
    def MILPooling_average(self, x, bag_size_list):
        '''Average pooling based on bag size
        '''
        start=0
        for bn in range(len(bag_size_list)):
            end = start+bag_size_list[bn]
            x1 = x[start:end]
            #print(x1)
            x1 = torch.cat((x1[:,0,:,:].view(-1,1,bag_size_list[bn]),x1[:,1,:,:].view(-1,1,bag_size_list[bn])),dim=1)
            #print(x1)
            x1 = F.avg_pool1d(x1,kernel_size=bag_size_list[bn])
            #print(x1)
            if bn==0:
                M = x1
            else:
                M = torch.cat((M,x1),dim=0)
            #print(M)
            #input('halt')
            start = end
        M = M.view(M.shape[0],-1)
        return M
    
    def MILPooling_maxpool(self, x, bag_size_list):
        start=0
        for bn in range(len(bag_size_list)):
            end = start+bag_size_list[bn]
            x1 = x[start:end]
            x1 = torch.cat((x1[:,0,:,:].view(-1,1,bag_size_list[bn]),x1[:,1,:,:].view(-1,1,bag_size_list[bn])),dim=1)
            x1 = F.max_pool1d(x1,kernel_size=bag_size_list[bn])
            if bn==0:
                M = x1
            else:
                M = torch.cat((M,x1),dim=0)
            start = end
        M = M.view(M.shape[0],-1)
        return M
    
    def forward(self, x, bag_size_list, views_names):
        #x = self._flatten_bag(x)
        H = self.feature_extractor(x)
        if self.milpooling=='attention':
            if self.activation=='sigmoid':
                A, H = self.model_attention(H)
                M = self.MILPooling_attention(A, H, bag_size_list)
                M = self.classifier(M)
                M = M.squeeze(1)
                M = M.view(-1)
            elif self.activation=='softmax':
                H_b = H[:,0,:,:].unsqueeze(1)
                H_m = H[:,1,:,:].unsqueeze(1)
                A_b, H_b = self.model_attention(H_b)
                A_m, H_m = self.model_attention(H_m)
                print(A_m.shape,H_m.shape)
                M_b = self.MILPooling_attention(A_b, H_b, bag_size_list)
                M_m = self.MILPooling_attention(A_m, H_m, bag_size_list)
                print(M_b.shape,M_m.shape)
                M_b = self.adaptiveavgpool(M_b)
                M_m = self.adaptiveavgpool(M_m)
                print(M_b.shape,M_m.shape)
                M = torch.cat((M_b,M_m),dim=1)
                print(M.shape)
                M = M.view(M.shape[0],-1)
                print(M.shape)
        elif self.milpooling=='average':
            M = self.adaptiveavgpool(H)
            M = self.MILPooling_average(M, bag_size_list)
        elif self.milpooling=='maxpool':
            M = self.adaptiveavgpool(H)
            M = self.MILPooling_maxpool(M, bag_size_list)
        print("final shape:",M.shape)
        return M

class feature_extractor(nn.Module):
    def __init__(self, L1, L2):
        super(feature_extractor, self).__init__()
        self.L1 = L1
        self.L2 = L2
        
        self.linear = nn.Sequential(
            nn.Linear(self.L1, self.L2),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.linear(x)
        return x

class SeparatePipelineMIL(nn.Module):
    '''Breast wise separate pipeline
    '''
    def __init__(self, milpooling, activation, device, extra, attention, feature_extractor):
        super(SeparatePipelineMIL, self).__init__()
        
        self.D = 128
        self.K = 1
        self.bag_size=4
        self.milpooling = milpooling
        self.activation = activation
        self.attention = attention
        self.device = device
        self.extra = extra
        if self.activation == 'softmax':
            self.L = 625
        elif self.activation == 'sigmoid':
            self.L = 625
        
        self.four_view_resnet = FourViewResNet(feature_extractor, extra, activation)
        
        if self.milpooling=='attention': 
            if self.activation=='softmax':
                if self.attention=='imagewise':
                    self.model_attention_b = Attention(self.L, self.D, self.K)
                    self.model_attention_m = Attention(self.L, self.D, self.K)
                elif self.attention=='breastwise':
                    self.model_attention_perbreast_b = Attention(self.L, self.D, self.K)
                    self.model_attention_perbreast_m = Attention(self.L, self.D, self.K)
                    self.model_attention_both_b = Attention(self.L, self.D, self.K)
                    self.model_attention_both_m = Attention(self.L, self.D, self.K)
                self.adaptiveavgpool = nn.AdaptiveAvgPool1d(1)
            
            elif self.activation=='sigmoid':
                if self.attention=='imagewise':
                    self.model_attention = Attention(self.L, self.D, self.K)
                elif self.attention=='breastwise':
                    self.model_attention_perbreast = Attention(self.L, self.D, self.K)
                    self.model_attention_both = Attention(self.L, self.D, self.K)
                if self.extra == 'fclayerreduction':
                    self.classifier = nn.Linear(self.L*self.K, 1) #2*25*25 = 1250 -> 1
                else:
                    self.adaptiveavgpool = nn.AdaptiveAvgPool1d(1)
        
        elif self.milpooling=='gatedattention':
            if self.activation=='softmax':
                if self.attention=='imagewise':
                    self.model_attention_b = GatedAttention(self.L, self.D, self.K)
                    self.model_attention_m = GatedAttention(self.L, self.D, self.K)
                elif self.attention=='breastwise':
                    self.model_attention_perbreast_b = GatedAttention(self.L, self.D, self.K)
                    self.model_attention_perbreast_m = GatedAttention(self.L, self.D, self.K)
                    self.model_attention_both_b = GatedAttention(self.L, self.D, self.K)
                    self.model_attention_both_m = GatedAttention(self.L, self.D, self.K)
                self.adaptiveavgpool = nn.AdaptiveAvgPool1d(1)
            
            elif self.activation=='sigmoid':
                if self.attention=='imagewise':
                    self.model_attention = GatedAttention(self.L, self.D, self.K)
                elif self.attention=='breastwise':
                    self.model_attention_perbreast = GatedAttention(self.L, self.D, self.K)
                    self.model_attention_both = GatedAttention(self.L, self.D, self.K)
                if self.extra == 'fclayerreduction':
                    self.classifier = nn.Linear(self.L*self.K, 1) #2*25*25 = 1250 -> 1
                else:
                    self.adaptiveavgpool = nn.AdaptiveAvgPool1d(1)
        
        elif self.milpooling=='average' or self.milpooling=='maxpool':
            if self.activation=='softmax':
                self.adaptiveavgpool = nn.AdaptiveAvgPool2d((1, 1)) #(1,1) is the output dimension of the result
            elif self.activation=='sigmoid':
                if self.extra == 'fclayerreduction':
                    self.classifier = nn.Linear(self.L*self.K, 1) #2*25*25 = 1250 -> 1
                else:
                    self.adaptiveavgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        elif self.milpooling=='concat':
            self.L1 = self.L*2
            self.classifier = nn.Linear(self.L1, 2) #2*25*25 = 1250 -> 1
            #self.classifier2 = nn.Linear(self.L, 2, bias=False)
        
        #for m in self.modules():
        #    if isinstance(m, nn.Linear):
        #        print('in init')
        #        nn.init.kaiming_normal_(m.weight)
        #        #nn.init.constant_(m.bias, 0)

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
    
    def MILPooling_maxpool(self, x, bag_size_list, activation):
        '''Max pooling based on bag size
        '''
        x = x.view(x.shape[0],x.shape[1],x.shape[2]) #Nx4x2
        if activation == 'softmax':
            #print(np.unique(np.array(bag_size_list))[0])
            x = torch.cat((x[:,:,0].view(-1,1,np.unique(np.array(bag_size_list))[0]),x[:,:,1].view(-1,1,np.unique(np.array(bag_size_list))[0])),dim=1) #Nx2x4
            #print(x)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fxn()
                max_val, max_id = F.max_pool2d(x, kernel_size=(x.shape[1],x.shape[2]), return_indices=True) #Nx2x1
            #print(max_val)
            #print(max_id)
            max_id = torch.remainder(max_id, np.unique(np.array(bag_size_list))[0])
            #print(max_id)
            max_id = max_id.repeat(1,2,1)
            #print(max_id)
            #print(x.shape)
            x = torch.gather(x,2,max_id)
            #print(x)
        elif activation == 'sigmoid':
            x = x.view(x.shape[0],x.shape[2],x.shape[1])
            x = F.max_pool1d(x,kernel_size=np.unique(np.array(bag_size_list))[0])
        x = x.view(x.shape[0],-1) #Nx2
        #print(x)
        #print(x.shape)
        return x
    
    def MILPooling_average(self, x, bag_size_list, activation):
        '''Average pooling based on bag size
        '''
        x = x.view(x.shape[0],x.shape[1],x.shape[2]) #Nx4x2
        if activation == 'softmax':
            #print(x.shape)
            #print(bag_size_list)
            #print(np.unique(np.array(bag_size_list))[0])
            x = torch.cat((x[:,:,0].view(-1,1,np.unique(np.array(bag_size_list))[0]),x[:,:,1].view(-1,1,np.unique(np.array(bag_size_list))[0])),dim=1) #Nx2x4
            #print(x.shape)
        elif activation == 'sigmoid':
            x = x.view(x.shape[0],x.shape[2],x.shape[1])
        x = F.avg_pool1d(x,kernel_size=np.unique(np.array(bag_size_list))[0]) #Nx2x1
        x = x.view(x.shape[0],-1) #Nx2
        return x
    
    def singlevsclassspecific_featuremap(self, h, bag_size_list, both_breast=False):
        if self.activation=='softmax':
            if both_breast:
                H_b = h[:,:,0,:].unsqueeze(2)
                H_m = h[:,:,1,:].unsqueeze(2)
            else:
                H_b = h[:,:,0,:,:].unsqueeze(2)
                H_m = h[:,:,1,:,:].unsqueeze(2)
            if self.attention == 'breastwise':
                if both_breast:
                    A_b, H_b = self.model_attention_both_b(H_b) #Nx2x625 -> benign map
                    A_m, H_m = self.model_attention_both_m(H_m) #Nx2x625 -> malignant map
                else:
                    A_b, H_b = self.model_attention_perbreast_b(H_b) #Nx2(views)x625 -> benign map
                    A_m, H_m = self.model_attention_perbreast_m(H_m) #Nx2x625
                #print(A_b.shape,A_m.shape)
            elif self.attention=='imagewise':
                #print(H_b)
                #print(H_m)
                A_b, H_b = self.model_attention_b(H_b) #Nx4x625
                A_m, H_m = self.model_attention_m(H_m) #Nx4x625
                #print(H_b)
                #print(A_b)
                #print(H_m)
                #print(A_m)
                #print(A_b.shape,A_m.shape)
            
            M_b = self.MILPooling_attention(A_b, H_b) #NxL -> aggregated benign maps over views
            M_m = self.MILPooling_attention(A_m, H_m) #NxL -> aggregated malignant maps over views
            #print(M_b)
            #print(M_m)
            M_b = M_b.unsqueeze(1) #Nx1xL
            M_m = M_m.unsqueeze(1) #Nx1xL
            M = torch.cat((M_b,M_m),dim=1) #Nx2xL -> aggregated benign & malignant maps over views
        
        elif self.activation=='sigmoid':
            if self.attention=='breastwise':
                if both_breast:
                    A, H = self.model_attention_both(h)
                else:
                    A, H = self.model_attention_perbreast(h)
            elif self.attention=='imagewise':
                A, H = self.model_attention(h) #Nx4xL
            M = self.MILPooling_attention(A, H) #NxL
        return M
    
    def singlevsmultipleclassscore(self, M):
        if self.activation=='softmax':
            #print(M.shape)
            M = self.adaptiveavgpool(M) #Nx2x625 -> Nx2x1
            if M.shape[2]==1:
                M = M.squeeze(2) #Nx2
        elif self.activation=='sigmoid':
            if self.extra == 'fclayerreduction':
                if len(M.shape)==5:
                    M = self.reshape(M)
                    M = self.classifier(M)
            else:
                M = self.adaptiveavgpool(M)#Nx625->Nx1
        return M
    
    def breast_wise_attention(self, views_names, h_view, bag_size_list):
        if 'LMLO' in views_names and 'LCC' in views_names:
            h_view['LCC'] = torch.unsqueeze(h_view['LCC'],1) #shape=Nx1x2x25x25
            h_view['LMLO'] = torch.unsqueeze(h_view['LMLO'],1) #shape=Nx1x2x25x25
            h_left = torch.cat((h_view['LCC'],h_view['LMLO']),dim=1) #shape=Nx2(views)x2(b/m)x25x25
            h_left = self.singlevsclassspecific_featuremap(h_left, bag_size_list) #Nx2xL
            #print(h_left.shape)
        elif 'LCC' in views_names:
            h_left = h_view['LCC']
            h_left = self.reshape(h_left) #shape=Nx2xL
            #print(h_left.shape)
        elif 'LMLO' in views_names:
            h_left = h_view['LMLO']
            h_left = self.reshape(h_left) #shape=Nx2xL
            #print(h_left.shape)
        else:
            h_left = torch.zeros(size=(0,1),device=self.device)
        
        if 'RMLO' in views_names and 'RCC' in views_names:
            h_view['RCC'] = torch.unsqueeze(h_view['RCC'],1) #shape=Nx1x2x25x25
            h_view['RMLO'] = torch.unsqueeze(h_view['RMLO'],1) #shape=Nx1x2x25x25
            h_right = torch.cat((h_view['RCC'],h_view['RMLO']),dim=1) #shape=Nx2x2x25x25
            h_right = self.singlevsclassspecific_featuremap(h_right, bag_size_list) #shape=Nx2xL
        elif 'RCC' in views_names:
            h_right = h_view['RCC']
            h_right = self.reshape(h_right) #shape=Nx2xL
            #print("h_right RCC:", h_right.shape)
        elif 'RMLO' in views_names:
            h_right = h_view['RMLO']
            h_right = self.reshape(h_right) #shape=Nx2xL
            #print("h_right RMLO:",h_right.shape)
        else:
            h_right = torch.zeros(size=(0,1),device=self.device)
        
        if len(h_left) and len(h_right):
            h_left = torch.unsqueeze(h_left,1) #shape=Nx1x2xL
            h_right = torch.unsqueeze(h_right,1) #shape=Nx1x2xL
            h_both = torch.cat((h_left,h_right),dim=1) #shape=Nx2x2xL
            h_final = self.singlevsclassspecific_featuremap(h_both, bag_size_list, both_breast=True) #shape=Nx2xL
        elif len(h_left):
            h_final = h_left
        elif len(h_right):
            h_final = h_right
        #print(h_final.shape)
        return h_final
    
    def reshape(self, x):
        if len(x.shape)==5:
            x = x.view(-1, x.shape[1], x.shape[2] * x.shape[3] * x.shape[4])
        elif len(x.shape)==4:
            x = x.view(-1, x.shape[1], x.shape[2] * x.shape[3])
        #elif len(x.shape)==3:
        #    x = x.view(-1, x.shape[1], x.shape[2])
        return x
    
    def forward(self, x, bag_size_list, views_names):
        h = self.four_view_resnet(x, views_names) #feature extractor, h['LCC'].shape=Nx2x25x25
        
        if self.attention=='imagewise':
            if 'LCC' in views_names:
                h['LCC']=torch.unsqueeze(h['LCC'],1) #shape=Nx1x2x25x25
                #print('LCC')
                #print(h['LCC'])
            if 'LMLO' in views_names:
                h['LMLO']=torch.unsqueeze(h['LMLO'],1) #shape=Nx1x2x25x25
                #print('LMLO')
                #print(h['LMLO'])
            if 'RCC' in views_names:
                h['RCC']=torch.unsqueeze(h['RCC'],1) #shape=Nx1x2x25x25
                #print('RCC')
                #print(h['RCC'])
            if 'RMLO' in views_names:
                h['RMLO']=torch.unsqueeze(h['RMLO'],1) #shape=Nx1x2x25x25
                #print('RMLO')
                #print(h['RMLO'])
            for counter, view in enumerate(views_names):
                if counter==0:
                    h_all=h[view]
                else:
                    h_all=torch.cat((h_all,h[view]),dim=1)
            #h_all = torch.cat((h['LCC'],h['LMLO'],h['RCC'],h['RMLO']),dim=1) #shape=Nx4x2x25x25
            #print(h_all)
            M = self.singlevsclassspecific_featuremap(h_all, bag_size_list) #Nx2xL
            #print(M.shape)
            M = self.singlevsmultipleclassscore(M) #Nx2x1
            #print(M.shape)
            #input('halt')
            
        elif self.attention=='breastwise':
            M = self.breast_wise_attention(views_names, h, bag_size_list) #shape=Nx2xL
            M = self.singlevsmultipleclassscore(M) #shape=Nx2x1
        
        elif self.milpooling=='maxpool':
            if 'LCC' in views_names:
                h['LCC']=torch.unsqueeze(h['LCC'],1) #shape=Nx1x2x25x25
            if 'LMLO' in views_names:
                h['LMLO']=torch.unsqueeze(h['LMLO'],1) #shape=Nx1x2x25x25
            if 'RCC' in views_names:
                h['RCC']=torch.unsqueeze(h['RCC'],1) #shape=Nx1x2x25x25
            if 'RMLO' in views_names:
                h['RMLO']=torch.unsqueeze(h['RMLO'],1) #shape=Nx1x2x25x25
            for counter, view in enumerate(views_names):
                if counter==0:
                    h_all=h[view]
                else:
                    h_all=torch.cat((h_all,h[view]),dim=1)
            #h_all = torch.cat((h['LCC'],h['LMLO'],h['RCC'],h['RMLO']),dim=1) #shape=Nx4x2x25x25
            #print(h_all)
            M = self.singlevsmultipleclassscore(h_all)
            #print(M.shape)
            #M = self.adaptiveavgpool(h_all) #shape=Nx4x2x1
            M = self.MILPooling_maxpool(M, bag_size_list, self.activation) #Nx2 or Nx1
            #print(M.shape)
        
        elif self.milpooling=='average':
            if 'LCC' in views_names:
                h['LCC']=torch.unsqueeze(h['LCC'],1) #shape=Nx1x2x25x25
            if 'LMLO' in views_names:
                h['LMLO']=torch.unsqueeze(h['LMLO'],1) #shape=Nx1x2x25x25
            if 'RCC' in views_names:
                h['RCC']=torch.unsqueeze(h['RCC'],1) #shape=Nx1x2x25x25
            if 'RMLO' in views_names:
                h['RMLO']=torch.unsqueeze(h['RMLO'],1) #shape=Nx1x2x25x25
            for counter, view in enumerate(views_names):
                if counter==0:
                    h_all=h[view]
                else:
                    h_all=torch.cat((h_all,h[view]),dim=1)
            #h_all = torch.cat((h['LCC'],h['LMLO'],h['RCC'],h['RMLO']),dim=1) #shape=Nx4x2x25x25
            #print(h_all.shape)
            M = self.singlevsmultipleclassscore(h_all)
            #print(M.shape)
            #M = self.adaptiveavgpool(h_all) #shape=Nx4x2x1
            M = self.MILPooling_average(M, bag_size_list, self.activation) #shape=Nx2 or Nx1
        
        elif self.milpooling=='concat':
            #h['LCC']=torch.unsqueeze(h['LCC'],1) #shape=Nx1x2x25x25
            #h['LMLO']=torch.unsqueeze(h['LMLO'],1) #shape=Nx1x2x25x25
            #h['RCC']=torch.unsqueeze(h['RCC'],1) #shape=Nx1x2x25x25
            #h['RMLO']=torch.unsqueeze(h['RMLO'],1) #shape=Nx1x2x25x25
            #print('h_lcc, h_lmlo:',h['LCC'].shape,h['LMLO'].shape)
            h_left = h['LCC'].add(h['LMLO']) #Nx2x25x25
            #print('h_left:',h_left.shape)
            h_right = h['RCC'].add(h['RMLO']) #Nx2x25x25
            #print('h_right:',h_right.shape)
            h_left = torch.flatten(h_left,start_dim=1) #Nx1250
            h_right = torch.flatten(h_right,start_dim=1) #Nx1250
            #print('flatten:',h_left.shape,h_right.shape)
            #h_both = torch.cat((h_left,h_right),dim=1) #Nx2x1250
            #h_both = h_both.view(-1,h_both.shape[2])
            h_left_score = self.classifier(h_left) #Nx625
            #h_left_score = self.classifier2(h_left1) #Nx2
            #print('h_left_score:',h_left_score.shape)
            h_right_score = self.classifier(h_right) #Nx2
            #h_right_score = self.classifier2(h_right2) #Nx2
            #print('h_right_score:',h_right_score.shape)
            h_score = torch.cat((h_left_score.unsqueeze(1),h_right_score.unsqueeze(1)),dim=1) #Nx2x2
            #print('h_score:',h_score.shape)
            #print(h_score)
            M = torch.mean(h_score,dim=1) #Nx2
            #print('M:',M.shape)
            #M = M.squeeze(1)
            #print('M:',M.shape)
            #print(M)
            #input('halt')
        
        print(M.shape)
        return M

class FourViewResNet(nn.Module):
    def __init__(self, feature_extractor, extra, activation):
        super(FourViewResNet, self).__init__()
        
        '''self.L1 = 1250
        self.L2 = 500'''
        
        self.model_dict = {}
        self.feature_extractor = feature_extractor
        self.extra = extra
        
        if self.feature_extractor=='viewwise':
            if self.extra == 'lastlayershared' or self.extra == 'last2layershared':
                self.shared_layers = resnet_features.MyResNet_shared(BasicBlock, [2, 2, 2, 2], self.extra)
                self.cc = resnet_features.MyResNet_separate(BasicBlock, [2, 2, 2, 2], self.extra) #this is for view-wise where first layer is shared
                self.mlo = resnet_features.MyResNet_separate(BasicBlock, [2, 2, 2, 2], self.extra) #this is for view-wise where first layer is shared
            elif self.extra == 'firstlayerlastlayershared':
                self.shared_firstlayer = resnet_features.MyResNet_shared(BasicBlock, [2, 2, 2, 2], 'firstlayershared')
                self.cc = resnet_features.MyResNet_separate(BasicBlock, [2, 2, 2, 2], self.extra) #this is for view-wise where first layer is shared
                self.mlo = resnet_features.MyResNet_separate(BasicBlock, [2, 2, 2, 2], self.extra) #this is for view-wise where first layer is shared
                self.shared_lastlayer = resnet_features.MyResNet_shared(BasicBlock, [2, 2, 2, 2], 'lastlayershared')
            else:
                self.cc = resnet_features.resnet18_features(activation=activation, extra = extra)
                self.mlo = resnet_features.resnet18_features(activation=activation, extra = extra)
            self.model_dict['LCC'] = self.cc
            self.model_dict['LMLO'] = self.mlo 
            self.model_dict['RCC'] = self.cc 
            self.model_dict['RMLO'] = self.mlo 
        
        elif self.feature_extractor=='common':
            if extra:
                if 'resnet18pretrained' in extra:
                    if 'resnet18pretrainedavg' in extra or 'resnet18pretrainedsum' in extra or 'resnet18pretrainedrchan' in extra:
                        self.view = resnet.resnet18(pretrained=True, inchans=1, wt_combine=extra)
                    elif 'resnet18pretrainedrgb' in extra:
                        print('i am in resnet18pretrainedrgb')
                        self.view = resnet.resnet18(pretrained=True, inchans=3)
                elif 'resnet18scratch' in extra:
                    self.view = resnet.resnet18(pretrained=False, inchans=1)
                elif 'wuresnet22' in extra:
                    self.view = wu_resnet.resnet22(input_channels=1)
                else:
                    self.view = resnet_features.resnet18_features(activation=self.activation, extra = self.extra)
            else:
                self.view = resnet_features.resnet18_features(activation=self.activation, extra = self.extra)
            self.model_dict['LCC'] = self.view 
            self.model_dict['LMLO'] = self.view 
            self.model_dict['RCC'] = self.view 
            self.model_dict['RMLO'] = self.view 
        
        elif self.feature_extractor=='sharedseparatemix': #view-wise where the first layer is shared is this block.
            self.shared_layers = resnet_features.MyResNet_shared(BasicBlock, [2, 2, 2, 2], self.extra)
            #self.cc = resnet_features.MyResNet_separate(BasicBlock, [2, 2, 2, 2], self.extra) #this is for view-wise where first layer is shared
            #self.mlo = resnet_features.MyResNet_separate(BasicBlock, [2, 2, 2, 2], self.extra) #this is for view-wise where first layer is shared
            self.lcc = resnet_features.MyResNet_separate(BasicBlock, [2, 2, 2, 2], self.extra) #this is for view-wise where first layer is shared
            self.lmlo = resnet_features.MyResNet_separate(BasicBlock, [2, 2, 2, 2], self.extra) #this is for view-wise where first layer is shared
            self.rcc = resnet_features.MyResNet_separate(BasicBlock, [2, 2, 2, 2], self.extra)
            self.rmlo = resnet_features.MyResNet_separate(BasicBlock, [2, 2, 2, 2], self.extra)
            self.model_dict['LCC'] = self.lcc
            self.model_dict['LMLO'] = self.lmlo
            self.model_dict['RCC'] = self.rcc
            self.model_dict['RMLO'] = self.rmlo
        
        elif self.feature_extractor=='allseparate':
            self.lcc = resnet_features.resnet18_features(activation=activation, extra = extra)
            self.lmlo = resnet_features.resnet18_features(activation=activation, extra = extra)
            self.rcc = resnet_features.resnet18_features(activation=activation, extra = extra)
            self.rmlo = resnet_features.resnet18_features(activation=activation, extra = extra)
            self.model_dict['LCC'] = self.lcc
            self.model_dict['LMLO'] = self.lmlo 
            self.model_dict['RCC'] = self.rcc 
            self.model_dict['RMLO'] = self.rmlo 
        
        '''self.cc_linear_feature = feature_extractor(self.L1, self.L2)
        self.mlo_linear_feature = feature_extractor(self.L1, self.L2)
        self.linear_feature={}
        self.linear_feature['LCC'] = self.cc_linear_feature
        self.linear_feature['LMLO'] = self.mlo_linear_feature
        self.linear_feature['RCC'] = self.cc_linear_feature
        self.linear_feature['RMLO'] = self.mlo_linear_feature'''

    def forward(self, x, views_names):
        #print("views names:",views_names)
        #print("input shape:",x.shape)
        h_dict = {
            #view: self.single_forward(x[views_names.index(view),:,:,:], view)
            view: self.single_forward(x[:,views_allowed.index(view),:,:], view)
            for view in views_names
        }
        return h_dict

    def single_forward(self, single_view, view):
        #print('single view:',single_view.shape)
        single_view = single_view.unsqueeze(1)
        #print('single view 1:',torch.count_nonzero(single_view,dim=(3,2)))
        #print('single view 1:',single_view)
        if self.feature_extractor=='sharedseparatemix':
            if self.extra == 'lastlayershared' or self.extra == 'last2layershared':
                single_view_feature = self.model_dict[view](single_view)
                single_view_feature = self.shared_layers(single_view_feature)
                #print("single forward:",single_view_feature.shape)
            else:
                single_view_feature = self.shared_layers(single_view)
                single_view_feature = self.model_dict[view](single_view_feature)
        
        elif self.feature_extractor=='viewwise':
            if self.extra == 'firstlayerlastlayershared':
                single_view_feature = self.shared_firstlayer(single_view)
                single_view_feature = self.model_dict[view](single_view_feature)
                single_view_feature = self.shared_lastlayer(single_view_feature)
            else:
                if self.extra == 'firstlayershared':
                    single_view_feature = self.shared_layers(single_view)
                    single_view_feature = self.model_dict[view](single_view_feature)
                
                elif self.extra == 'lastlayershared' or self.extra == 'last2layershared':
                    single_view_feature = self.model_dict[view](single_view)
                    single_view_feature = self.shared_layers(single_view_feature)
                
                else:
                    single_view_feature = self.model_dict[view](single_view)
        
        else:
            single_view_feature = self.model_dict[view](single_view)
            
        #single_view_feature = reshape(single_view_feature)
        #single_view_feature = self.linear_feature[view](single_view_feature)
        #print('single view 1:',torch.count_nonzero(single_view_feature,dim=(0,1,2)))
        #print('single view 1:',single_view_feature.shape)
        #print(single_view_feature.shape)
        return single_view_feature


