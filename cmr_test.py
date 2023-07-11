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

opt = TestOptions().parse()

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
    
    img_path = data_i['path']
    
    
    if(i==0):
        print(f'inital')
        path = data_i['path'][0]
        #Expected 3D
        image3D_epoch = torch.empty(221,512,512)




    elif(path != data_i['path'][0] or (len(dataloader)-1 )== i):
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
        affine = np.eye(4)
        image3D_epoch_np = image3D_epoch.detach().numpy()
        img = nib.Nifti1Image(image3D_epoch_np, affine)

        #get image nr. from path file name
        path = data_i['path'][0]


        imgNr= int(re.search(r"\d{4}", path).group())


        filename = f"3DImage{imgNr}SameImageDiffMasks{i%221}try2.nii.gz"

        nib.save(img, filename = os.path.join(opt.checkpoints_dir, opt.name,'web','images', filename))
        #nib.save(img,filename= '/home/sastocke/2Dslicesfor3D/'+filename)

        # start new stacking for the next 3D image
        image3D_epoch = torch.empty(221,512,512)
        generated = model(data_i, mode='inference')
        image3D_epoch[0,:,:] = generated[0,0,:,:]
    elif(True):
        print(f'adding')
        #Add to the stack of 3D
        image3D_epoch[i%221,:,:] = generated[0,0,:,:]
    
    print(f'path: {path}')
    print(f'path of the data_i: {data_i["path"][0]}')





webpage.save()
