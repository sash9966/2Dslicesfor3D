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

ref_img = sitk.ReadImage('/home/sastocke/data/testimages128/ct_1129_image.nii.gz')


opt = TestOptions().parse()
name = opt.name
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
print(f'length of dtalaoder: {len(dataloader)}')
# test
for i, data_i in enumerate(dataloader):

    print(f'i: {i}')
    generated = model(data_i, mode='inference').detach().cpu()
    
    img_path = data_i['path']

    print(f'shape of generated: {generated.shape}')
    img = sitk.GetImageFromArray(generated[0,0,:,:,:])
    img.CopyInformation(ref_img)
    path = data_i['gtname'][0]
    imgNr= int(re.search(r"\d{4}", path).group())
    filename = f"3DImage{name}{imgNr}Synthetic.nii.gz"
    sitk.WriteImage(img, os.path.join(opt.checkpoints_dir, opt.name,'web','images', filename))
    print(f'saved image: {filename}')

    


#Save last image


print(f'done')
    





webpage.save()
