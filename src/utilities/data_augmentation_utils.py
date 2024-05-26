import torch
import math
import random
import numpy as np
from sklearn import utils
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF

class ROIRotateTransform:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = np.random.choice(self.angles).item()
        return TF.rotate(x, angle)

class MyCrop:
    """Randomly crop the sides."""

    def __init__(self, left=100,right=100,top=100,bottom=100):
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom

    def __call__(self, x):
        width, height=x.size
        size_left = random.randint(0,self.left)
        size_right = random.randint(width-self.right,width)
        size_top = random.randint(0,self.top)
        size_bottom = random.randint(height-self.bottom,height)
        x = TF.crop(x,size_top,size_left,size_bottom,size_right)
        return x
    
class MyGammaCorrection:
    def __init__(self, factor=0.2):
        self.lb = 1-factor
        self.ub = 1+factor

    def __call__(self, x):
        gamma = random.uniform(self.lb,self.ub)
        return TF.adjust_gamma(x,gamma)

def myhorizontalflip(image,breast_side):
    if breast_side=='R':
        image = np.fliplr(image).copy() #R->L (following GMIC)
    return image

class MyHorizontalFlip:
    """Flip horizontally."""

    def __init__(self):
        pass

    def __call__(self, x, breast_side):
        #if breast_side=='L':
        #    return TF.hflip(x) #L->R
        if breast_side=='R':
            return TF.hflip(x) #R->L (following GMIC)
        else:
            return x

class MyPadding:
    def __init__(self, breast_side, max_height, max_width, height, width):
        self.breast_side = breast_side
        self.max_height=max_height
        self.max_width=max_width
        self.height=height
        self.width=width
          
    def __call__(self,img):
        print(img.shape)
        print(self.max_height-self.height)
        if self.breast_side=='L':
            image_padded=F.pad(img,(0,self.max_width-self.width,0,self.max_height-self.height,0,0),'constant',0)
        elif self.breast_side=='R':
            image_padded=F.pad(img,(self.max_width-self.width,0,0,self.max_height-self.height,0,0),'constant',0)
        print(image_padded.shape)
        return image_padded

class MyPaddingLongerSide:
    def __init__(self, resize):
        self.max_height=resize[0]
        self.max_width=resize[1]
        
    def __call__(self, img, breast_side):
        width=img.size[0]
        height=img.size[1]
        if height<self.max_height:
            diff=self.max_height-height
            img=TF.pad(img,(0,math.floor(diff/2),0,math.ceil(diff/2)),0,'constant')
        if width<self.max_width:
            diff=self.max_width-width
            if breast_side == 'L':
                img=TF.pad(img,(0,0,diff,0),0,'constant')
            elif breast_side == 'R':
                img=TF.pad(img,(diff,0,0,0),0,'constant')
        return img
    
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def data_augmentation_train_kim(config_params, mean, std_dev):
    preprocess_train = transforms.Compose([
        transforms.ColorJitter(brightness=0.10, contrast=0.10),
        transforms.Resize((config_params['resize'][0], config_params['resize'][1])),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std_dev)
    ])
    return preprocess_train

def data_augmentation_train_shu(config_params, mean, std_dev):
    preprocess_train_list=[]
    if config_params['resize']:
        preprocess_train_list.append(transforms.Resize((config_params['resize'][0],config_params['resize'][1])))

    preprocess_train_list.append(transforms.ToTensor())

    preprocess_train_list=preprocess_train_list+[
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(contrast=0.20, saturation=0.20),
        transforms.RandomRotation(30),
        AddGaussianNoise(mean=0, std=0.005)
        ]
    
    if config_params['datascaling']:
        if config_params['datascaling']!='standardizeperimage':
            preprocess_train_list.append(transforms.Normalize(mean=mean, std=std_dev))
    
    preprocess_train = transforms.Compose(preprocess_train_list)
    return preprocess_train

def data_augmentation_train_shen_gmic(config_params, mean, std_dev):
    preprocess_train_list=[]

    if config_params['imagecleaning']!='gmic':
        if config_params['resize']:
            preprocess_train_list.append(transforms.Resize((config_params['resize'][0], config_params['resize'][1])))
    
    if config_params['papertoreproduce'] == 'wu20':
        noise_std = 0.01
    else:
        noise_std = 0.005

    preprocess_train_list=preprocess_train_list+[
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=(15),translate=(0.1,0.1),scale=(0.8,1.6),shear=(25)),
        AddGaussianNoise(mean=0, std=noise_std)
        ]
    
    if config_params['datascaling']:
        if config_params['datascaling']!='standardizeperimage':
            preprocess_train_list.append(transforms.Normalize(mean=mean, std=std_dev))
    
    preprocess_train = transforms.Compose(preprocess_train_list)
    return preprocess_train

def data_augmentation_test_shen_gmic(config_params, mean, std_dev):
    preprocess_test_list=[]

    if config_params['imagecleaning']!='gmic':
        if config_params['resize']:
            preprocess_test_list.append(transforms.Resize((config_params['resize'][0],config_params['resize'][1])))
    
    if config_params['datascaling']:  
        if config_params['datascaling']!='standardizeperimage':
            preprocess_test_list.append(transforms.Normalize(mean=mean, std=std_dev))
    
    preprocess_test = transforms.Compose(preprocess_test_list)
    return preprocess_test

class TrivialAugmentWideNoColor(transforms.TrivialAugmentWide):
    def _augmentation_space(self, num_bins: int):
        return {
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.5, num_bins), True), 
            "ShearY": (torch.linspace(0.0, 0.5, num_bins), True), 
            "TranslateX": (torch.linspace(0.0, 16.0, num_bins), True), 
            "TranslateY": (torch.linspace(0.0, 16.0, num_bins), True), 
            "Rotate": (torch.linspace(0.0, 60.0, num_bins), True), 
        }

def data_augmentation_train_pipnet(config_params, mean, std_dev):
    transform = transforms.Compose([
        transforms.Resize(size=(config_params['resize'][0], config_params['resize'][1])),
        TrivialAugmentWideNoColor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop(size=(config_params['resize'][0], config_params['resize'][1]), scale=(0.95, 1.)),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std_dev)
    ])
    return transform

def data_augmentation_test_pipnet(config_params, mean, std_dev):
    transform = transforms.Compose([
            transforms.Resize(size=(config_params['resize'][0], config_params['resize'][1])),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std_dev)
        ])
    return transform

def data_augmentation(config_params):
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            utils.MyCrop(),
            transforms.Pad(100),
            transforms.RandomRotation(3),
            transforms.ColorJitter(brightness=0.20, contrast=0.20),
            transforms.RandomAdjustSharpness(sharpness_factor=0.20),
            utils.MyGammaCorrection(0.20),
            utils.MyPaddingLongerSide(config_params['resize']),
            transforms.Resize((config_params['resize'][0],config_params['resize'][1])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((config_params['resize'][0],config_params['resize'][1])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    return data_transforms

def data_augmentation_train(config_params, mean, std_dev):
    preprocess_train_list=[]

    preprocess_train_list=preprocess_train_list+[
        MyCrop(),
        #transforms.Pad(100),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(3),
        transforms.ColorJitter(brightness=0.20, contrast=0.20),
        transforms.RandomAdjustSharpness(sharpness_factor=0.20),
        MyGammaCorrection(0.20),
        MyPaddingLongerSide(config_params['resize'])]
    
    if config_params['resize']:
        preprocess_train_list.append(transforms.Resize((config_params['resize'][0],config_params['resize'][1])))
    
    preprocess_train_list.append(transforms.ToTensor())
    
    if config_params['datascaling']:
        if config_params['datascaling']!='standardizeperimage':
            preprocess_train_list.append(transforms.Normalize(mean=mean, std=std_dev))
    
    preprocess_train = transforms.Compose(preprocess_train_list)
    return preprocess_train

def data_augmentation_test(config_params, mean, std_dev):
    preprocess_test_list=[]

    if config_params['resize']:
        preprocess_test_list.append(transforms.Resize((config_params['resize'][0],config_params['resize'][1])))
    
    preprocess_test_list.append(transforms.ToTensor())
    
    if config_params['datascaling']:  
        if config_params['datascaling']!='standardizeperimage':
            preprocess_test_list.append(transforms.Normalize(mean=mean, std=std_dev))
    
    preprocess_test = transforms.Compose(preprocess_test_list)
    return preprocess_test