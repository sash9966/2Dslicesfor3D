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
import gc
from util.util import save_as_resized_pickle
import pickle

 
ref_img = sitk.ReadImage('/scratch/users/fwkong/SharedData/Synthesized_correction/ct_1001_image_pred_r0.nii.gz')




opt = TestOptions().parse()

#Generate image for these masks
#opt.label_dir = '/scratch/users/fwkong/SharedData/Synthesized'
#Background image for generation!
opt.image_dir = '/scratch/users/sastocke/data/data/images/ct_1001_image.nii.gz'
#Output path to save the generated images
output_path = opt.results_dir
#target_path = '/scratch/users/fwkong/SharedData/imageCHDcleaned_all/whole_heart_processed/pytorch/ct_1001_image_0.pkl'
name = opt.name

#For generation, batchSize must be 1, create one slice at a time
opt.batchSize = 1


dataloader = data.create_dataloader(opt)

model = Pix2PixModel(opt)
model.eval()

visualizer = Visualizer(opt)


# test
for i, data_i in enumerate(dataloader):


    generated = model(data_i, mode='inference')
    
    
    if(i==0):
        print(f'inital')
        path = data_i['gtname'][0]
        #Expected 3D
        image3D_epoch = torch.empty(512,512,221)
        image3D_epoch[:,:,0] = generated[0,0,:,:]




    elif(path != data_i['gtname'][0] or (len(dataloader)-1)== i):

        #save old 3D stacked, should be 221 images stacked together
        #Override for the new 3D stacked image
        print(f'new image')
        if((len(dataloader)-1 )== i):
            print(f'last image')
            print(f'path: {path}')
            image3D_epoch[:,:,i%221] = generated[0,0,:,:]
        # affine = np.array([[1, 0, 0, 0],
        #            [0, 1, 0, 0],
        #            [0, 0, 1, 0],
        #            [0, 0, 0, 1]])
        #SimpleITK -> find the call to get the transformation, is in the load function, read image function
        image3D_epoch_np = image3D_epoch.detach().numpy()
        img = sitk.GetImageFromArray(image3D_epoch_np.transpose(2, 1, 0))
        img.CopyInformation(ref_img)


        #get image nr. from path file name



        imgNr= int(re.search(r"\d{4}", path).group())


        filename = f"3DImage{name}{imgNr}.nii.gz"

        sitk.WriteImage(img, os.path.join(output_path, filename))

        
        #nib.save(img,filename= '/home/sastocke/2Dslicesfor3D/'+filename)

        # start new stacking for the next 3D image
        path = data_i['gtname'][0]
        image3D_epoch = torch.empty(512,512,221)
        generated = model(data_i, mode='inference')
        image3D_epoch[:,:,0] = generated[0,0,:,:]
    elif(True):
        print(f'adding')
        #Add to the stack of 3D
        image3D_epoch[:,:,i%221] = generated[0,0,:,:]
    

        
    


    





