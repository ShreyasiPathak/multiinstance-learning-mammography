# Copyright (C) 2020 Yiqiu Shen, Nan Wu, Jason Phang, Jungkyu Park, Kangning Liu,
# Sudarshini Tyagi, Laura Heacock, S. Gene Kim, Linda Moy, Kyunghyun Cho, Krzysztof J. Geras
#
# This file is part of GMIC.
#
# GMIC is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# GMIC is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with GMIC.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================

"""
Module that define the core logic of GMIC
"""

import torch
import torch.nn as nn
import numpy as np
from torchvision._internally_replaced_utils import load_state_dict_from_url

from utilities import data_augmentation_utils, gmic_utils
from models import gmic_modules_multigpu as m


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


class GMIC(nn.Module):
    def __init__(self, parameters):
        super(GMIC, self).__init__()

        # save parameters
        self.experiment_parameters = parameters
        self.cam_size = parameters["cam_size"]

        # construct networks
        # global network
        self.global_network = m.GlobalNetwork(self.experiment_parameters)
        #self.global_network.add_layers()

        # aggregation function
        self.aggregation_function = m.TopTPercentAggregationFunction(self.experiment_parameters)

        # detection module
        self.retrieve_roi_crops = m.RetrieveROIModule(self.experiment_parameters)

        # detection network
        #self.roitransform = data_augmentation_utils.ROIRotateTransform([0, 90, 180, 270])
        self.local_network = m.LocalNetwork()
        #self.local_network.add_layers()

        # MIL module
        self.attention_module = m.AttentionModule(self.experiment_parameters)
        #self.attention_module.add_layers()

        # fusion branch
        if self.experiment_parameters['learningtype'] == 'SIL':
            self.fusion_dnn = nn.Linear(parameters["post_processing_dim"]+512, parameters["num_classes"])

    def _convert_crop_position(self, crops_x_small, cam_size, x_original):
        """
        Function that converts the crop locations from cam_size to x_original
        :param crops_x_small: N, k*c, 2 numpy matrix
        :param cam_size: (h,w)
        :param x_original: N, C, H, W pytorch variable
        :return: N, k*c, 2 numpy matrix
        """
        # retrieve the dimension of both the original image and the small version
        h, w = cam_size
        _, _, H, W = x_original.size()

        # interpolate the 2d index in h_small to index in x_original
        top_k_prop_x = crops_x_small[:, :, 0] / h
        top_k_prop_y = crops_x_small[:, :, 1] / w
        # sanity check
        assert np.max(top_k_prop_x) <= 1.0, "top_k_prop_x >= 1.0"
        assert np.min(top_k_prop_x) >= 0.0, "top_k_prop_x <= 0.0"
        assert np.max(top_k_prop_y) <= 1.0, "top_k_prop_y >= 1.0"
        assert np.min(top_k_prop_y) >= 0.0, "top_k_prop_y <= 0.0"
        # interpolate the crop position from cam_size to x_original
        top_k_interpolate_x = np.expand_dims(np.around(top_k_prop_x * H), -1)
        top_k_interpolate_y = np.expand_dims(np.around(top_k_prop_y * W), -1)
        top_k_interpolate_2d = np.concatenate([top_k_interpolate_x, top_k_interpolate_y], axis=-1)
        return top_k_interpolate_2d

    def _retrieve_crop(self, x_original_pytorch, crop_positions, crop_method):
        """
        Function that takes in the original image and cropping position and returns the crops
        :param x_original_pytorch: PyTorch Tensor array (N,C,H,W)
        :param crop_positions:
        :return:
        """
        batch_size, num_crops, _ = crop_positions.shape
        crop_h, crop_w = self.experiment_parameters["crop_shape"]

        #output = torch.ones((batch_size, num_crops, x_original_pytorch.shape[1], crop_h, crop_w))
        output = torch.ones((batch_size, num_crops, crop_h, crop_w))
        if self.experiment_parameters["device_type"] == "gpu":
            #device = torch.device("cuda:{}".format(self.experiment_parameters["gpu_number"]))
            device = torch.device(self.experiment_parameters["gpu_number"])
            output = output.cuda().to(device)
        for i in range(batch_size):
            for j in range(num_crops):
                gmic_utils.crop_pytorch(x_original_pytorch[i, 0, :, :], 
                                   self.experiment_parameters["crop_shape"], 
                                   crop_positions[i,j,:], 
                                   output[i,j,:,:],
                                   method=crop_method)
        #print("output:", output)
        return output

    def forward(self, x_original, eval_mode):
        """
        :param x_original: N,H,W,C numpy matrix
        """
        #print("gmic:", x_original.get_device(), x_original.shape)
        # global network: x_small -> class activation map
        h_g, self.saliency_map, sal_map_before_sigmoid = self.global_network(x_original)
        #print("h_g shape:", h_g.shape, flush=True)
        #print("saliency map shape:", self.saliency_map.shape, flush=True)
        #print("h_g:", h_g.shape) #3, 512, 92, 60
        # calculate y_global
        # note that y_global is not directly used in inference
        topt_feature, self.y_global = self.aggregation_function(self.saliency_map, sal_map_before_sigmoid)
        #print("topt features shape:", topt_feature.shape, flush=True)
        #print("y_global shape:", self.y_global.shape, flush=True)

        # gmic region proposal network
        small_x_locations = self.retrieve_roi_crops(x_original, self.cam_size, self.saliency_map)
        #print("small x location shape:", small_x_locations.shape, flush=True)

        # convert crop locations that is on self.cam_size to x_original
        self.patch_locations = self._convert_crop_position(small_x_locations, self.cam_size, x_original)

        # patch retriever
        crops_variable = self._retrieve_crop(x_original, self.patch_locations, self.retrieve_roi_crops.crop_method)
        self.patches = crops_variable.data.cpu().numpy()
        
        # detection network
        #batch_size, num_crops, Ch, I, J = crops_variable.size()
        batch_size, num_crops, I, J = crops_variable.size()
        #crops_variable = crops_variable.view(batch_size * num_crops, Ch, I, J) #.unsqueeze(1) #60x1x256x256
        crops_variable = crops_variable.view(batch_size * num_crops, I, J).unsqueeze(1)
        #print("crops_variable:", crops_variable.shape)
        #print(np.moveaxis(x_original[0, :, :, :].cpu().numpy(), 0, -1).shape)
        #plt.imsave(str(0)+'_image.png', x_original[0, 0, :, :].cpu().numpy(), cmap='gray')
        #print("patch location:", self.patch_locations.shape) # 10, 6, 2
        #print(self.patch_locations[0,:,:])
        #if not eval_mode:
        #    for i in range(0, crops_variable.shape[0]):
        #        crops_variable[i, :, :, :] = self.roitransform(crops_variable[i, :, :, :])
        #        #plt.imsave(str(i)+'_roi.png', crops_variable[i, 0, :, :].cpu().numpy(), cmap='gray', vmin=-2.117, vmax=2.248)
        
        h_crops = self.local_network(crops_variable).view(batch_size, num_crops, -1)
        #input('halt')
        # MIL module
        # y_local is not directly used during inference
        if self.experiment_parameters['learningtype'] == 'SIL':
            z, self.patch_attns, self.y_local = self.attention_module(h_crops)
            #print("z shape", z.shape, flush=True)
            #print("y_local shape", self.y_local.shape, flush=True)
        elif self.experiment_parameters['learningtype'] == 'MIL':
            z, self.patch_attns = self.attention_module(h_crops)

        # fusion branch
        # use max pooling to collapse the feature map
        g1, _ = torch.max(h_g, dim=2)
        #print('g1.shape', g1.shape) #3, 512, 60
        global_vec, _ = torch.max(g1, dim=2) 
        #print('global_vec shape:', global_vec.shape) #3, 512
        '''if self.parameters['num_classes'] > 1:
            fusion_dim = global_vec.shape[-1] + z.shape[-1]
            concat_vec = torch.empty((batch_size, self.parameters['num_classes'], fusion_dim))
            for k in range(self.parameters['num_classes']):
                concat_vec[:, k, :] = torch.cat([global_vec, z[:, k, :]], dim=1)
        else:'''
        concat_vec = torch.cat([global_vec, z], dim=1)
        #self.y_fusion = torch.sigmoid(self.fusion_dnn(concat_vec))
        if self.experiment_parameters['learningtype'] == 'SIL':
            self.y_fusion = self.fusion_dnn(concat_vec)
            #print("fusion:", self.y_fusion.shape, flush=True)

        if self.experiment_parameters['learningtype'] == 'SIL':
            return self.y_local, self.y_global, self.y_fusion, self.saliency_map, self.patch_locations, self.patches, self.patch_attns, h_crops
        elif self.experiment_parameters['learningtype'] == 'MIL':
            #return z, topt_feature, self.y_global, concat_vec, self.saliency_map
            return z, topt_feature, self.y_global, concat_vec, self.saliency_map, self.patch_locations, self.patches, self.patch_attns, h_crops, global_vec#, topt_feature_before_sig


def _gmic(gmic_parameters):
    gmic_model = GMIC(gmic_parameters)

    #print("gmic module", flush=True)
    #for name, param in gmic_model.named_parameters():
    #    if param.requires_grad:
    #        print(f"{name} -> {param.device}", flush=True)      

    if gmic_parameters['pretrained']:
        resnet_pretrained_dict = load_state_dict_from_url(model_urls[gmic_parameters['arch']],progress=True)
        
        #load pretrained ImageNet weights for the global network
        gmic_model_dict = gmic_model.state_dict()
        global_network_dict = {k:v for (k,v) in gmic_model_dict.items() if 'ds_net' in k}
        # 1. filter out unnecessary keys
        global_network_pretrained_dict = {'ds_net.'+k: v for k, v in resnet_pretrained_dict.items() if ('ds_net.'+k in global_network_dict) and (k!='fc.weight') and (k!='fc.bias')}
        # 2. overwrite entries in the existing state dict
        gmic_model_dict.update(global_network_pretrained_dict) 
        
        #load pretrained ImageNet weights for the local network
        local_network_dict = {k:v for (k,v) in gmic_model_dict.items() if 'dn_resnet' in k}
        # 1. filter out unnecessary keys
        local_network_pretrained_dict = {'dn_resnet.'+k: v for k, v in resnet_pretrained_dict.items() if ('dn_resnet.'+k in local_network_dict) and (k!='fc.weight') and (k!='fc.bias')}
        # 2. overwrite entries in the existing state dict
        gmic_model_dict.update(local_network_pretrained_dict) 
        
        gmic_model.load_state_dict(gmic_model_dict)
        
    return gmic_model