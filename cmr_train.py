
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
import SimpleITK as sitk
import matplotlib.pyplot as plt
from tqdm import tqdm
from util import util
ref_img = sitk.ReadImage('/scratch/users/sastocke/data/data/images/ct_1001_image.nii.gz')[:,:,:3]
# parse options
opt = TrainOptions().parse()

#BUG: Unsure if for larger crop size this should be changed, seems to work without!
if opt.crop_size == 256:
     opt.resnet_n_downsample = 5
     opt.resnet_n_blocks = 2
else:
    opt.resnet_n_downsample = 4
    opt.resnet_n_blocks = 2
    
opt.use_vae = False

print(f'3D testing!!')
name_of_try = opt.name

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
torch.cuda.empty_cache()

if (os.path == '/home/users/sastocke/2Dslicesfor3D'):
    opt.checkpoints_dir = '/scratch/users/sastocke/results/3dfusetry'


print(f'path where saving will go is: {opt.checkpoints_dir}')

for epoch in iter_counter.training_epochs():
    print('epoch', epoch)
    iter_counter.record_epoch_start(epoch)

    


    for i, data_i in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}"), start=iter_counter.epoch_iter):

 

        iter_counter.record_one_iteration()
        latest = None
        if(epoch>10 and i%221 != 0 ):
            latest = trainer.get_latest_generated().detach()

        # Training
        # train generator
        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data_i,latest)

        # train discriminator
        trainer.run_discriminator_one_step(data_i,latest)


        # Visualizations
        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                                losses, iter_counter.time_per_iter)
            visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

        if iter_counter.needs_displaying():
            #show the 3D image
            latest_image = trainer.get_latest_generated()
            # print(f'latest_image shape: {latest_image.shape}')
            # print(f'label shape: {data_i["label"].shape}')
            # print(f'real image shape: {data_i["image"].shape}')

            #plot the three images side by side
            #convert to numpy array
            latest_image = latest_image.detach().cpu().numpy()
            real_image = (data_i['image']).detach().cpu().numpy()
            label = (data_i['label']).detach().cpu().numpy()


            #plot the three images side by side


            #plot image
            #get current path:
            path = os.getcwd()            
            plt.close('all')
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))

            for j in range(3):
                axs[0].imshow(label[0,j,:,:])
                axs[0].axis('off')
                axs[0].set_title('Input Label')
                axs[1].imshow(latest_image[0,0,j,:,:],cmap='gray')
                axs[1].axis('off')
                axs[1].set_title('Synthesized Image')
                axs[2].imshow(real_image[0,j,:,:],cmap='gray')
                axs[2].axis('off')
                axs[2].set_title('Real Image')
                plt.savefig(f'/scratch/users/sastocke/results/3dfusetry{name_of_try}{epoch}_{i}_plotdepth{j}.png')

                # visuals = OrderedDict([('input_label', label[:,:,:,j]),
                #     ('synthesized_image', latest_image[:,:,j,:,:]),
                # #     ('real_image', real_image[:,:,:,j])])

                # visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)
                
                # util.save_image(latest_image_np[0,0,i,:,:], '/home/sastocke/2Dslicesfor3D/checkpoints/{name_of_try}/web/images/latestsynthetic_epoch{epoch}_{i}.png')
                # util.save_image(real_image_np[0,:,:,i], '/home/sastocke/2Dslicesfor3D/checkpoints/{name_of_try}/web/images/real_epoch_{epoch}_{i}.png')
                # util.save_image(label_np[0,:,:,i], '/home/sastocke/2Dslicesfor3D/checkpoints/{name_of_try}/web/images/label_epoch{epoch}_{i}.png')
                

                
            
            img = sitk.GetImageFromArray(latest_image[0,0,:,:,:])
            img.CopyInformation(ref_img)
            sitk.WriteImage(img, f'/scratch/users/sastocke/results/3dfusetry/{name_of_try}latestsynthetic{epoch}.nii.gz')
            #Save 3D stacked image



        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')



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
    
    




print('Training was successfully finished.')
