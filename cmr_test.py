import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html
import torch
import numpy as np
import nibabel as nib
import re
import SimpleITK as sitk
"/home/users/sastocke/2Dslicesfor3D/util/util.py"

#ref_img = sitk.ReadImage('/home/users/sastocke/data/alltrainingdata/data/images/ct_1001_image.nii.gz')


opt = TestOptions().parse()
opt.use_vae = False

#voxel size
vs = opt.voxel_size
depth= 221

dataloader = data.create_dataloader(opt)

model = Pix2PixModel(opt)
model.eval()

visualizer = Visualizer(opt)

# create a webpage that summarizes the all results
web_dir = os.path.join(opt.results_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))


webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))

# test
for i, data_i in enumerate(dataloader):
    if(i%vs != 0):
        continue

    generated = model(data_i, mode='inference')
    generated = generated.permute(0,1,4,3,2)
    
    img_path = data_i['path']

    
    
    if(i==0):
        print(f'inital')
        path = data_i['gtname'][0]
        #Expected 3D
        image3D_epoch = torch.empty(512,512,221)
        image3D_epoch[:,:,0:vs] = generated[0,0,:,:,:]

        




    elif(path != data_i['gtname'][0]):
        #save old 3D stacked, should be 221 images stacked together
        #Override for the new 3D stacked image
        # affine = np.array([[1, 0, 0, 0],
        #            [0, 1, 0, 0],
        #            [0, 0, 1, 0],
        #            [0, 0, 0, 1]])
        #SimpleITK -> find the call to get the transformation, is in the load function, read image function
        # affine = np.eye(4)
        image3D_epoch_np = image3D_epoch.detach().numpy()
        # img = nib.Nifti1Image(image3D_epoch_np, affine)
        img = sitk.GetImageFromArray(image3D_epoch_np.transpose(2, 1, 0))
        img.CopyInformation(ref_img)


        #get image nr. from path file name
        path = data_i['gtname'][0]


        imgNr= int(re.search(r"\d{4}", path).group())


        filename = f"3DImage{imgNr}Synthetic{i}.nii.gz"

        sitk.WriteImage(img, os.path.join(opt.checkpoints_dir, opt.name,'web','images', filename))
        print(f'saved image: {filename}')
        #nib.save(img,filename= '/home/sastocke/2Dslicesfor3D/'+filename)

        # start new stacking for the next 3D image
        image3D_epoch = torch.empty(512,512,221)

        image3D_epoch[:,:,0:vs] = generated[0,0,:,:,:]
    elif(True):
        #Add to the stack of 3D

        #Start and end index of the current slice
        start_idx = i%depth
        end_idx = start_idx+vs

        #Edge case for the last slice when 221%voxel_size != 0
        if(end_idx > depth):
            end_idx = depth

        print(f'end_idx: {end_idx}, start_idx: {start_idx}')
        print(f'generated shape: {generated.shape}')
        print(f'i: {i}')
        print(f'len(dataloader): {len(dataloader)}')
        stack_slice = generated[0,0,:,:,:end_idx-start_idx]
        image3D_epoch[:,:,start_idx:end_idx] = stack_slice
        print(f'path: {path}')

#Save last image
sitk.WriteImage(img, os.path.join(opt.checkpoints_dir, opt.name,'web','images', filename))

print(f'done')
    





webpage.save()
