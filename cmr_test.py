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

ref_img = sitk.ReadImage('/home/sastocke/data/testimages/ct_1129_image.nii.gz')



opt = TestOptions().parse()
opt.label_dir = '/home/sastocke/data/SynthesizedTest'
opt.image_dir = '/home/sastocke/data/testimages'
name = opt.name

#For generation, batchSize must be 1, create one slice at a time
opt.batchSize = 1


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


    #print(f'i: {i}')

    if i * opt.batchSize >= opt.how_many:
        break

    generated = model(data_i, mode='inference')
    
    
    if(i==0):
        print(f'inital')
        path = data_i['gtname'][0]
        #Expected 3D
        image3D_epoch = torch.empty(512,512,221)




    elif(path != data_i['gtname'][0] or (len(dataloader)-1 )== i):
        #save old 3D stacked, should be 221 images stacked together
        #Override for the new 3D stacked image
        print(f'new image')
        if((len(dataloader)-1 )== i):
            print(f'last image')
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



        imgNr= int(re.search(r"\d{4}", path).group())


        filename = f"3DImage{name}{imgNr}{i}.nii.gz"

        sitk.WriteImage(img, os.path.join(opt.checkpoints_dir, opt.name,'web','images', filename))
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
    
    print(f'path: {path}')
    print(f'path of the data_i: {data_i["gtname"][0]}')





webpage.save()
