"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.nn.utils.spectral_norm as spectral_norm
from models.networks.normalization import SPADE, SPADELight, SPADE3D


# ResNet block that uses SPADE.
# It differs from the ResNet block of pix2pixHD in that
# it takes in the segmentation map as input, learns the skip connection if necessary,
# and applies normalization first and then convolution.
# This architecture seemed like a standard architecture for unconditional or
# class-conditional GAN architecture using residual block.
# The code was inspired from https://github.com/LMescheder/GAN_stability.
class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        # create conv layers
        add_channels = 1 if (opt.norm_mode == 'clade' and not opt.no_instance) else 0
        if (opt.norm_mode == 'spade3d'): 
            self.conv_0 = nn.Conv3d(fin, fmiddle, kernel_size=3, padding=1)
            self.conv_1 = nn.Conv3d(fmiddle, fout, kernel_size=3, padding=1)
            ##Why needed?
            if self.learned_shortcut:
                self.conv_s = nn.Conv3d(fin+add_channels, fout, kernel_size=1, bias=False)

        elif (opt.norm_mode == 'spade'):
            self.conv_0 = nn.Conv2d(fin+add_channels, fmiddle, kernel_size=3, padding=1)
            self.conv_1 = nn.Conv2d(fmiddle+add_channels, fout, kernel_size=3, padding=1)
            if self.learned_shortcut:
                self.conv_s = nn.Conv2d(fin+add_channels, fout, kernel_size=1, bias=False)
            


        # apply spectral norm if specified
        if 'spectral' in opt.norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = opt.norm_G.replace('spectral', '')
        if opt.norm_mode == 'spade':
            self.norm_0 = SPADE(spade_config_str, fin, opt.semantic_nc)
            self.norm_1 = SPADE(spade_config_str, fmiddle, opt.semantic_nc)
            if self.learned_shortcut:
                self.norm_s = SPADE(spade_config_str, fin, opt.semantic_nc)
        elif opt.norm_mode == 'clade':
            input_nc = opt.label_nc + (1 if opt.contain_dontcare_label else 0)
            self.norm_0 = SPADELight(spade_config_str, fin, input_nc, opt.no_instance, opt.add_dist)
            self.norm_1 = SPADELight(spade_config_str, fmiddle, input_nc, opt.no_instance, opt.add_dist)
            if self.learned_shortcut:
                self.norm_s = SPADELight(spade_config_str, fin, input_nc, opt.no_instance, opt.add_dist)
        elif opt.norm_mode == 'spade3d':
            self.norm_0 = SPADE3D(spade_config_str, fin, opt.semantic_nc)
            self.norm_1 = SPADE3D(spade_config_str, fmiddle, opt.semantic_nc)
            if self.learned_shortcut:
                self.norm_s = SPADE3D(spade_config_str, fin, opt.semantic_nc)
        else:
            raise ValueError('%s is not a defined normalization method' % opt.norm_mode)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg, input_dist=None):
        x_s = self.shortcut(x, seg, input_dist)
        #print(f'x: {x.shape}, seg: {seg.shape}, input_dist: {input_dist.shape}')
        #print(f'x_s: {x_s.shape}')

        #get info on the parameters of the network
        dx = self.conv_0(self.actvn(self.norm_0(x, seg, input_dist)))
        #print(f'dx after conv_0: {dx.shape}')
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg, input_dist)))
        #print(f'dx after_conv1: {dx.shape}')

        out = x_s + dx

        return out

    def shortcut(self, x, seg, input_dist=None):
        #print(f'shortcut is calledx: {x.shape}, seg: {seg.shape}')
        
        if self.learned_shortcut:
            #print(f'learned_shortcut is called')
            x_s = self.conv_s(self.norm_s(x, seg, input_dist))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


# ResNet block used in pix2pixHD
# We keep the same architecture as pix2pixHD.
class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, activation=nn.ReLU(False), kernel_size=3):
        super().__init__()

        pw = (kernel_size - 1) // 2
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size)),
            activation,
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size))
        )

    def forward(self, x):
        y = self.conv_block(x)
        out = x + y
        return out
    
class ResnetBlock3D(nn.Module):
    def __init__(self, dim, norm_layer,kernel_size,activation=nn.ReLU(False), ):
        super().__init__()

        pad = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2, (kernel_size[2] - 1) // 2)
        self.conv_block = nn.Sequential(
            nn.ConstantPad3d((pad[1], pad[1], pad[2], pad[2], pad[0], pad[0]), 0),
            norm_layer(nn.Conv3d(dim, dim, kernel_size=kernel_size)),
            activation,
            nn.ConstantPad3d((pad[1], pad[1], pad[2], pad[2], pad[0], pad[0]), 0),
            norm_layer(nn.Conv3d(dim, dim, kernel_size=kernel_size))
        )

    def forward(self, x):

        #print(f'x: {x.shape}')
        y= self.conv_block(x)
        #print(f'y: {y.shape}')
        return x + y





# VGG architecter, used for the perceptual loss using a pretrained VGG network
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()





        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        #print(f'X: {X.shape}')
        depth = X.shape[2]
        if (len(X.shape)== 4):
            X = X.unsqueeze(1)
        h_relu_sums = [0, 0, 0, 0, 0]  # to store sum of losses for each h_relu

        for z in range(X.shape[2]):  # loop over the z-axis (depth dimension)
            X_slice = X[:, :, z, :, :]
            if X_slice.shape[1] == 1:  # if number of channels is less than 3:
                X_slice = X_slice.repeat(1, 3, 1, 1)


                #TODO: Check if this vgg makes more senes!
            #X_slice = (X_slice + 1) / 2.0  # Adjust to [0, 1] range
            # Apply ImageNet normalization
            # mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(X_slice.device)
            # std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(X_slice.device)
            # X_slice = (X_slice - mean) / std

            h_relu1 = self.slice1(X_slice)
            h_relu2 = self.slice2(h_relu1)
            h_relu3 = self.slice3(h_relu2)
            h_relu4 = self.slice4(h_relu3)
            h_relu5 = self.slice5(h_relu4)
            h_relu_sums[0] += h_relu1.sum()
            h_relu_sums[1] += h_relu2.sum()
            h_relu_sums[2] += h_relu3.sum()
            h_relu_sums[3] += h_relu4.sum()
            h_relu_sums[4] += h_relu5.sum()

        h_relu_tensors = [torch.tensor(h_relu_sum) for h_relu_sum in h_relu_sums]

        ##:TODO: check if this is correct
        #h_relu_tensors = [torch.tensor(h_relu_sum)/depth for h_relu_sum in h_relu_sums]
        
        return h_relu_tensors   # return the output for each slice