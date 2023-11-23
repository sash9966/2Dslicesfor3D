"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.architecture import VGG19
from models.networks.architecture import Modified3DUNet
import os
import numpy as np
from scipy import linalg
#from pytorch_fid.inception import InceptionV3
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image



# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        currTensor = self.Tensor if 'HalfTensor' not in input.type() else torch.cuda.HalfTensor
        if self.zero_tensor is None:
            self.zero_tensor = currTensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)


# Perceptual loss that uses a pretrained VGG network
class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        if(torch.cuda.is_available()):
            self.vgg = VGG19().cuda()
        else:
            self.vgg = VGG19().cpu()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


# KL Divergence loss used in VAE with an image encoder
class KLDLoss(nn.Module):
    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
        self.criterion = nn.L1Loss()
    def forward(self, fake, real):
        #look at shake of fake and real
        
        #have to make sure the shapes are the same

        #BUG: Why is [0,:,:,:] needed?
        # print(f'fake shape: {fake.shape}')
        # print(f'real shape: {real.shape}')
        # print(f' fake[0,:,:,:] shape {fake[0,:,:,:].shape}')
        # print(f' real[0,:,:,:] shape {real[0,:,:,:].shape}')
        return self.criterion(fake[0,:,:,:], real[0,:,:,:])
    

class Modified3DUNetLoss(nn.Module):
    def __init__(self, in_channels, n_classes, base_n_filter, gpu_ids, pretrained_model_path=None):
        super(Modified3DUNetLoss, self).__init__()
        self.unet = Modified3DUNet(in_channels, n_classes, base_n_filter)

        # Load pre-trained weights if a path is provided
        if pretrained_model_path is not None:
            #self.unet.load_state_dict(torch.load(pretrained_model_path))

            checkpoint_folder = os.path.join(pretrained_model_path,'net_150.pt')
            self.unet.load_state_dict(torch.load(checkpoint_folder, map_location=torch.device('cpu'))['state_dict'])
            

        # Freeze the Modified3DUNet model
        for param in self.unet.parameters():
            param.requires_grad = False

        # Move the model to GPU if available
        if torch.cuda.is_available():
            self.unet = self.unet.cuda()
        else:
            self.unet = self.unet.cpu()

        self.criterion = nn.MSELoss()  # Using Mean Squared Error Loss

    def forward(self, fake,real):
        real = real.unsqueeze(0)
        real_unet = self.unet(real)
        fake_unet = self.unet(fake)
        loss = self.criterion(real_unet, fake_unet.detach())
        return loss


class FIDLoss(nn.Module):
    #adapted version of https://github.com/mseitzer/pytorch-fid 
    def __init__(self, num_slices=50, batch_size=50, dims=2048, device='cpu'):
        super(FIDLoss, self).__init__()
        #self.inception_model = InceptionV3([3]).to(device)
        #self.inception_model.eval()
        self.num_slices = num_slices
        self.batch_size = batch_size
        self.dims = dims
        self.device = device
    def preprocess_slices(self, volume, real):

        #Check for real vs synthetic, as synthetic have different shapes.
        #Also make it compatible regardless of batchsize, always take the first two voxels and compare
        if real:
            volume = volume[0,:,:,:]
        else:
            volume = volume[0,0,:,:,:]
        depth = volume.shape[0]
        start = (depth - self.num_slices) // 2
        end = start + self.num_slices
        slices = volume[start:end, :, :]

        processed_slices = []
        for slice in slices:
            # Convert to PIL image
            slice_image = Image.fromarray(slice)

            # Convert to tensor and repeat the grayscale channel
            slice_tensor = ToTensor()(slice_image).repeat(3, 1, 1)

            # Resize and Normalize
            slice_tensor = Resize((299, 299))(slice_tensor)
            slice_tensor = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(slice_tensor)

            processed_slices.append(slice_tensor)

        return torch.stack(processed_slices)


    def get_activations(self, slices):
        dataloader = torch.utils.data.DataLoader(slices, batch_size=self.batch_size)
        pred_arr = np.empty((0, self.dims))

        for batch in dataloader:
            batch = batch.to(self.device)
            with torch.no_grad():
                pred = self.inception_model(batch)[0]

            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = nn.functional.adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            pred_arr = np.append(pred_arr, pred, axis=0)

        return pred_arr

    def calculate_fid(self, real_features, gen_features):
        mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = gen_features.mean(axis=0), np.cov(gen_features, rowvar=False)
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * 1e-6
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * np.trace(covmean))
        return fid

    def forward(self, real_volume, synthetic_volume ):
        print(f'shape of real_volume: {real_volume.shape}, synthetic volume: {synthetic_volume.shape}')
        real_slices = self.preprocess_slices(real_volume, real = True)
        synthetic_slices = self.preprocess_slices(synthetic_volume, real = False)
        real_features = self.get_activations(real_slices)
        gen_features = self.get_activations(synthetic_slices)
        fid_score = self.calculate_fid(real_features, gen_features)
        return fid_score
