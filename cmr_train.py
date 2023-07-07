
import os
import sys
from collections import OrderedDict
from options.train_options import TrainOptions
import data
from data.base_dataset import repair_data
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from trainers.pix2pix_trainer import Pix2PixTrainer
from util import html
from util.util import tensor2im, tensor2label
import torch
import nibabel as nib
import numpy as np


# parse options
opt = TrainOptions().parse()

#BUG: Unsure if for larger crop size this should be changed, seems to work without!
if opt.crop_size == 256:
     opt.resnet_n_downsample = 5
     opt.resnet_n_blocks = 2
else:
    opt.resnet_n_downsample = 4
    opt.resnet_n_blocks = 2



# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataloader = data.create_dataloader(opt)

# create trainer for our model
trainer = Pix2PixTrainer(opt)

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))

# create tool for visualization
visualizer = Visualizer(opt)

    

for epoch in iter_counter.training_epochs():
    print('epoch', epoch)
    iter_counter.record_epoch_start(epoch)

    

    print(f'lenght of dataloader: {len(dataloader)}')
    for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):

        #First initalisation

        #look at data that's loaded:
        # print(f'i: {i}')
        #print(f' data_i: {data_i.keys()}')
        # print the value of the key with 'image'

        # data_i dict_keys(['label', 'image', 'instance', 'dist', 'path', 'gtname', 'index', 'segpair_slice'])
        # print(f' data_i: {data_i["image"]}')
        # print(f' image shape: {data_i["image"].shape}')
        # print(f'gt name is: {data_i["gtname"]}')
        # print(f'path is: {data_i["path"]}')

        #Set initial path:

        #When paths change, a new 3D volume is being generated


        iter_counter.record_one_iteration()

        #print(f'type of the image, is it PIL or torch tensor: {type(data_i['label'])}') 
        #print(f'data_i label type: {type(data_i)}')

        # Training
        # train generator
        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data_i)

        # train discriminator
        trainer.run_discriminator_one_step(data_i)

        #print(f'trainer contains which type of generated image: {type(trainer.get_latest_generated)}')
        #print(f'trainer containes generated {trainer.get_latest_generated().shape}')

        # Visualizations
        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                                losses, iter_counter.time_per_iter)
            visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

        if iter_counter.needs_displaying():
            visuals = OrderedDict([('input_label', data_i['label']),
                                   ('synthesized_image', trainer.get_latest_generated()),
                                   ('real_image', data_i['image'])])
            visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

            #Save 3D stacked image


        #stack the images togehter to create a 3D image of the generated images



        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, iter_counter.total_steps_so_far))
        
            trainer.save('latest')
        

        if(i==0):
            path = data_i['path']
            #Expected 3D
            image3D_epoch = torch.empty(221,512,512)
            print(f'initial done')
            print(f'image3D_epoch: {image3D_epoch.shape}')


        elif(path != data_i['path'] and iter_counter.needs_displaying ):
            #save old 3D stacked, should be 221 images stacked together
            print(f'stacked image shape: {image3D_epoch.shape()}')
            #Override for the new 3D stacked image
            img = nib.Nifti1Image(image3D_epoch, affine)

            #get image nr. from path file name
            imgNr= data['path'][3:7]
            filename = "3Depoch{epoch}Image{imgNr}.nii.gz"
            nib.save(nifti_image, os.path.join(opt.checkpoints_dir, opt.name, 'images', filename))

            # start new stacking for the next 3D image
            image3D_epoch = torch.empty(221,512,512)
            image3D_epoch[0,:,:] = trainer.get_latest_generated()[0,0,:,:]
            path = data_i['path']
        
        else:
            print(f'image_3d_epoch shape= {image3D_epoch.shape}')
            print(f'latest generated is: {trainer.get_latest_generated().shape}')

            image3D_epoch[i%221,:,:] = trainer.get_latest_generated()[0,0,:,:]



    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()
    if epoch % opt.save_epoch_freq == 0 or \
       epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        trainer.save('latest')
        trainer.save(epoch)
    #check the shape of the stacked image
    #convert to numpy arry and save with nib
    image3D_epoch = image3D_epoch.cpu().detach().numpy()
    print(f' shape of image3D_epoch: {image3D_epoch.shape}')
    nifti_image = nib.Nifti1Image(image3D_epoch, affine=np.eye(4))
    




print('Training was successfully finished.')
