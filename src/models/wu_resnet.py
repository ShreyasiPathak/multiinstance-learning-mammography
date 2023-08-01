# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 15:17:37 2022

@author: PathakS

Functions in this script taken from Wu et al. repository
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import conv3x3

class SplitBreastModel(nn.Module):
    def __init__(self, config_params):
        super(SplitBreastModel, self).__init__()

        self.channel = config_params['channel']

        self.four_view_resnet = FourViewResNet(self.channel)

        self.fc1_cc = nn.Linear(256 * 2, 256 * 2)
        self.fc1_mlo = nn.Linear(256 * 2, 256 * 2)
        self.output_layer_cc = OutputLayer(256 * 2, 1)
        self.output_layer_mlo = OutputLayer(256 * 2, 1)

        self.all_views_avg_pool = AllViewsAvgPool()
        #self.all_views_gaussian_noise_layer = AllViewsGaussianNoise(0.01)

    def forward(self, x, views_names, eval_mode):
        #h = self.all_views_gaussian_noise_layer(x)
        result = self.four_view_resnet(x, views_names)
        h = self.all_views_avg_pool(result)

        # Pool, flatten, and fully connected layers
        h_cc = torch.cat([h['LCC'], h['RCC']], dim=1)
        h_mlo = torch.cat([h['LMLO'], h['RMLO']], dim=1)

        h_cc = F.relu(self.fc1_cc(h_cc))
        h_mlo = F.relu(self.fc1_mlo(h_mlo))

        h_cc = self.output_layer_cc(h_cc)
        h_mlo = self.output_layer_mlo(h_mlo)
        #print(h_cc, h_mlo)
        out = torch.cat((h_cc, h_mlo), dim = 1)
        #print(out)
        out = torch.mean(out, dim = 1)
        #print(out)
        return out

class FourViewResNet(nn.Module):
    def __init__(self, input_channels):
        super(FourViewResNet, self).__init__()

        self.cc = resnet22(input_channels)
        self.mlo = resnet22(input_channels)
        self.model_dict = {}
        self.model_dict['LCC'] = self.l_cc = self.cc
        self.model_dict['LMLO'] = self.l_mlo = self.mlo
        self.model_dict['RCC'] = self.r_cc = self.cc
        self.model_dict['RMLO'] = self.r_mlo = self.mlo

    def forward(self, x, views_names):
        h_dict = {
            view: self.single_forward(x[:,views_names.index(view),:,:].unsqueeze(1), view)
            for view in views_names
        }
        return h_dict

    def single_forward(self, single_x, view):
        return self.model_dict[view](single_x)

class OutputLayer(nn.Module):
    def __init__(self, in_features, output_shape):
        super(OutputLayer, self).__init__()
        #if not isinstance(output_shape, (list, tuple)):
        #    output_shape = [output_shape]
        self.output_shape = output_shape
        #self.flattened_output_shape = int(np.prod(output_shape))
        self.fc_layer = nn.Linear(in_features, self.output_shape) #self.flattened_output_shape)

    def forward(self, x):
        h = self.fc_layer(x)
        #if len(self.output_shape) > 1:
        h = h.view(h.shape[0], self.output_shape)
        #h = F.log_softmax(h, dim=-1)
        return h

class AllViewsGaussianNoise(nn.Module):
    """Add gaussian noise across all 4 views"""

    def __init__(self, gaussian_noise_std):
        super(AllViewsGaussianNoise, self).__init__()
        self.gaussian_noise_std = gaussian_noise_std

    def forward(self, x):
        return {
            'LCC': self.single_add_gaussian_noise(x['LCC']),
            'LMLO': self.single_add_gaussian_noise(x['LMLO']),
            'RCC': self.single_add_gaussian_noise(x['RCC']),
            'RMLO': self.single_add_gaussian_noise(x['RMLO']),
        }

    def single_add_gaussian_noise(self, single_view):
        if not self.gaussian_noise_std or not self.training:
            return single_view
        return single_view + single_view.new(single_view.shape).normal_(std=self.gaussian_noise_std)


class AllViewsAvgPool(nn.Module):
    """Average-pool across all 4 views"""

    def __init__(self):
        super(AllViewsAvgPool, self).__init__()

    def forward(self, x):
        return {
            view_name: self.single_avg_pool(view_tensor)
            for view_name, view_tensor in x.items()
        }

    @staticmethod
    def single_avg_pool(single_view):
        n, c, _, _ = single_view.size()
        return single_view.view(n, c, -1).mean(-1)

class BasicBlockV2(nn.Module):
    """
    Adapted fom torchvision ResNet, converted to v2
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockV2, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=1)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        # Phase 1
        out = self.bn1(x)
        out = self.relu(out)
        if self.downsample is not None:
            residual = self.downsample(out)
        out = self.conv1(out)

        # Phase 2
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual

        return out

class ViewResNetV2(nn.Module):
    """
    Adapted fom torchvision ResNet, converted to v2
    """
    def __init__(self,
                 input_channels, num_filters,
                 first_layer_kernel_size, first_layer_conv_stride,
                 blocks_per_layer_list, block_strides_list, block_fn,
                 first_layer_padding=0,
                 first_pool_size=None, first_pool_stride=None, first_pool_padding=0,
                 growth_factor=2):
        super(ViewResNetV2, self).__init__()
        self.first_conv = nn.Conv2d(
            in_channels=input_channels, out_channels=num_filters,
            kernel_size=first_layer_kernel_size,
            stride=first_layer_conv_stride,
            padding=first_layer_padding,
            bias=False,
        )
        self.first_pool = nn.MaxPool2d(
            kernel_size=first_pool_size,
            stride=first_pool_stride,
            padding=first_pool_padding,
        )

        self.layer_list = nn.ModuleList()
        current_num_filters = num_filters
        self.inplanes = num_filters
        for i, (num_blocks, stride) in enumerate(zip(
                blocks_per_layer_list, block_strides_list)):
            self.layer_list.append(self._make_layer(
                block=block_fn,
                planes=current_num_filters,
                blocks=num_blocks,
                stride=stride,
            ))
            current_num_filters *= growth_factor
        self.final_bn = nn.BatchNorm2d(
            current_num_filters // growth_factor * block_fn.expansion
        )
        self.relu = nn.ReLU()

        # Expose attributes for downstream dimension computation
        self.num_filters = num_filters
        self.growth_factor = growth_factor

    def forward(self, x):
        h = self.first_conv(x)
        h = self.first_pool(h)
        for i, layer in enumerate(self.layer_list):
            h = layer(h)
        h = self.final_bn(h)
        h = self.relu(h)
        return h

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
        )

        layers_ = [
            block(self.inplanes, planes, stride, downsample)
        ]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers_.append(block(self.inplanes, planes))

        return nn.Sequential(*layers_)


def resnet22(input_channels):
    return ViewResNetV2(
        input_channels=input_channels,
        num_filters=16,
        first_layer_kernel_size=7,
        first_layer_conv_stride=2,
        blocks_per_layer_list=[2, 2, 2, 2, 2],
        block_strides_list=[1, 2, 2, 2, 2],
        block_fn=BasicBlockV2,
        first_layer_padding=0,
        first_pool_size=3,
        first_pool_stride=2,
        first_pool_padding=0,
        growth_factor=2
    )