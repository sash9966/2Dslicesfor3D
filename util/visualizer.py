"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import PIL 
import nibabel as nib

import numpy as np

import os
import ntpath
import time
from . import util
from . import html
import scipy.misc
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x

class Visualizer():
    def __init__(self, opt):
        self.opt = opt
        self.tf_log = opt.isTrain and opt.tf_log
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        if self.tf_log:
            import tensorflow as tf
            self.tf = tf
            self.log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'logs')
            self.writer = tf.summary.create_file_writer(self.log_dir)

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        if opt.isTrain:
            self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)

    #|visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, step):


        # convert tensors to numpy arrays
        #check if visuals are tensors
        # if (visuals[''] == 'torch.Tensor'):
        #     visuals = self.convert_visuals_to_numpy(visuals)
        
        if self.tf_log: # show images in tensorboard output
            with self.writer.as_default():
                for label, image_numpy in visuals.items():
                    if len(image_numpy.shape) >= 4:
                        image_numpy = image_numpy[0]
                    # Convert the image to [0, 255] and uint8
                    image_numpy = (image_numpy * 255).astype(np.uint8)
                    # Use tf.summary.image to log the image
                    self.tf.summary.image(label, image_numpy[np.newaxis], step=step)
            self.writer.flush()


        if self.use_html: # save images to a html file
            for label, image_numpy in visuals.items():
                if isinstance(image_numpy, list):
                    for i in range(len(image_numpy)):
                        img_path = os.path.join(self.img_dir, 'epoch%.3d_iter%.3d_%s_%d.png' % (epoch, step, label, i))
                        util.save_image(image_numpy[i], img_path)
                else:
                    img_path = os.path.join(self.img_dir, 'epoch%.3d_iter%.3d_%s.png' % (epoch, step, label))


                    #Safe and handle images different for generated vs orginal image 
                    if(image_numpy.shape[0] == image_numpy.shape[1]):
                        #show PIL image
                        pil_img = PIL.Image.fromarray(image_numpy)
                        pil_img.save(img_path)
                    elif(image_numpy.shape[1] == image_numpy.shape[2]):
                        #plt.imshow(image_numpy[0,:,:])
                        #plt.title('image, probably generated or full')
                        #plt.show()
                        util.save_image(image_numpy[0,:,:], img_path)

                    
                    # if len(image_numpy.shape) >= 4:
                    #     print(f' len is called!')
                    #     image_numpy = image_numpy[0,0,:,:]      

                    

            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=5)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, image_numpy in visuals.items():
                    if isinstance(image_numpy, list):
                        for i in range(len(image_numpy)):
                            img_path = 'epoch%.3d_iter%.3d_%s_%d.png' % (n, step, label, i)
                            ims.append(img_path)
                            txts.append(label+str(i))
                            links.append(img_path)
                    else:
                        img_path = 'epoch%.3d_iter%.3d_%s.png' % (n, step, label)
                        ims.append(img_path)
                        txts.append(label)
                        links.append(img_path)
                if len(ims) < 10:
                    webpage.add_images(ims, txts, links, width=self.win_size)
                else:
                    num = int(round(len(ims)/2.0))
                    webpage.add_images(ims[:num], txts[:num], links[:num], width=self.win_size)
                    webpage.add_images(ims[num:], txts[num:], links[num:], width=self.win_size)
            webpage.save()

    # errors: dictionary of error labels and values
    def plot_current_errors(self, errors, step):
        if self.tf_log:
            with self.writer.as_default():
                for tag, value in errors.items():
                    value = value.mean().float().item()  # Convert tensor to a Python scalar
                    self.tf.summary.scalar(tag, value, step=step)
            self.writer.flush()


    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t, fid=None):
        if fid is None:
            message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        else:
            message = '(epoch: %d, iters: %d, time: %.3f, FID: %.3f) ' % (epoch, i, t, fid)
        for k, v in errors.items():
            #print(v)
            #if v != 0:
            v = v.mean().float()
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def print_eval_fids(self, epoch, curr_fid, best_fid):
        message = '(epoch: %d, curr FID: %.3f, previous best FID: %.3f) ' % (epoch, curr_fid, best_fid)
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def convert_visuals_to_numpy(self, visuals):
        for key, t in visuals.items():
            tile = self.opt.batchSize > 2
            if 'input_label' == key:
                t = util.tensor2label(t, self.opt.label_nc + 2, tile=tile)
            else:
                t = util.tensor2im(t, tile=tile)
            visuals[key] = t
        return visuals

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):        
        visuals = self.convert_visuals_to_numpy(visuals)        
        
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = os.path.join(label, '%s.png' % (name))
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path, create_dir=True)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)
        