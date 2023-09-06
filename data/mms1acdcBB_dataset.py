"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from data.base_dataset import BaseDataset



import os
import nibabel as nib
import util.cmr_dataloader as cmr
import util.cmr_transform as cmr_tran
from torch.utils.data import DataLoader
from torchvision.transforms import Compose


TR_CLASS_MAP_MMS_SRS= {'BG': 0,'LV_Bloodpool': 1, 'LV_Myocardium': 2,'RV_Bloodpool': 3, 'Test' : 4}
TR_CLASS_MAP_MMS_DES= {'BG': 0,'LV_Bloodpool': 1, 'LV_Myocardium': 2,'RV_Bloodpool': 3, 'Test' : 4}

class Mms1acdcBBDataset(BaseDataset):
    """ Dataset that loads images from directories
        Use option --label_dir, --image_dir, --instance_dir to specify the directories.
        The images in the directories are sorted in alphabetical order and paired in order.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = BaseDataset.modify_commandline_options(parser, is_train)
       
        # parser.set_defaults(label_nc=4)
        parser.set_defaults(output_nc=1)
        # parser.set_defaults(crop_size=128)
        parser.set_defaults(no_instance=True)
        parser.set_defaults(add_dist=False)
        
        # parser.add_argument('--label_dir', type=str, required=False, default = "/home/sastocke/2Dslicesfor3D/data/masks",
        #                     help='path to the directory that contains label images')
        # parser.add_argument('--image_dir', type=str, required=False, default ="/home/sastocke/2Dslicesfor3D/data/images" ,
        #                     help='path to the directory that contains photo images')
        parser.add_argument('--label_dir', type=str, required=False, default = "/home/sastocke/data/alltrainingdata/data/segmentation",
                            help='path to the directory that contains label images')
        parser.add_argument('--image_dir', type=str, required=False, default ="/scratch/users/sastocke/data/data/testimage/ct_1029_image.nii.gz" ,
                            help='path to the directory that contains photo images')
        # parser.add_argument('--label_dir', type=str, required=False, default = "/home/sastocke/data/SynthesizedTest",
        #                     help='path to the directory that contains label images')
        # parser.add_argument('--image_dir', type=str, required=False, default ="/home/sastocke/data/testimages" ,
        #                     help='path to the directory that contains photo images')
        
        # parser.add_argument('--label_dir_B', type=str, required=False, default = "/Users/saschastocker/Desktop/Data/StyleTransfer/segmentationTestFullResolution",
        #                     help='path to the directory that contains label images')
        

        # parser.add_argument('--image_dir_B', type=str, required=False, default ="/Users/saschastocker/Desktop/Data/StyleTransfer/imageTestFullResolution" ,
        #                     help='path to the directory that contains photo images')
        # parser.add_argument('--instance_dir', type=str, default='',
        #                     help='path to the directory that contains instance maps. Leave black if not exists')
        # parser.add_argument('--acdc_dir', type=str, required=False, default = "/Users/saschastocker/Desktop/Data/StyleTransfer/SlicedMRI/patient101_frame01.nii/",
        #                     help='path to the directory that contains label images')
                        
        return parser

    def get_paths(self, opt):
        """
        To prepare and get the list of files
        """

        #SA_mask_list = sorted(os.listdir(os.path.join(opt.label_dir)))

        #Single file loading:
        self.img_list = []
        self.msk_list = []
        self.filename_pairs = []

        self.img_list += [opt.image_dir]
        self.msk_list += [opt.label_dir]
        self.filename_pairs += [(opt.image_dir, opt.label_dir)]

        # if(opt.phase == 'test'):
        #     #For test we will generate images with different mask but paired with one patient image for the background.
        #     single_image = opt.image_dir
        #     SA_image_list = [single_image] * len(SA_mask_list)
        # else:
        #     SA_image_list = sorted(os.listdir(os.path.join(opt.image_dir)))

        # print(f'length of SA_image_list: {len(SA_image_list)}')
        # print(f'length of SA_mask_list: {len(SA_mask_list)}')



        # # assert len(SA_mask_list_B) == len(SA_image_list_B) 
        # if(opt.phase != 'test'):

        #     assert len(SA_image_list) == len(SA_mask_list)


        # SA_filename_pairs = [] 

        # for i in range(len(SA_image_list)):
        #     SA_filename_pairs += [(os.path.join(opt.image_dir,SA_image_list[i]), os.path.join(opt.label_dir, SA_mask_list[i]))]

        
        # self.img_list = SA_image_list
        # self.msk_list = SA_mask_list
        # self.filename_pairs = SA_filename_pairs

        # #print the file names and their content


        return self.filename_pairs, self.img_list, self.msk_list



    def initialize(self, opt):
        self.opt = opt
        print(f'filename pairs trying to be read from options: {self.opt}')
        self.filename_pairs, _, _  = self.get_paths(self.opt)


   

        if opt.isTrain:
            train_transforms = Compose([
                # cmr_tran.Resample(self.opt.target_res,self.opt.target_res), #1.33
                # cmr_tran.CenterCrop2D((self.opt.crop_size,self.opt.crop_size)),
                
                # cmr_tran.RandomRotation(degrees=90),
                # cmr_tran.RandomRotation(p=0.5),
                cmr_tran.RandomRotation90(p=0.7),
                
                cmr_tran.ToTensor(),
                cmr_tran.NormalizeMinMaxpercentile(range=(-1,1), percentiles=(1,99)),
                # cmr_tran.NormalizeLabel(),
                # cmr_tran.NormalizeMinMaxRange(range=(-1,1)),
                
                # cmr_tran.PercentileBasedRescaling(out_min_max=(-1,1), percentiles=(1,99)),  #TODO: make sure the normalization is performed on the volume data not slice-by-slice
                # cmr_tran.RandomElasticTorchio(num_control_points  = (8, 8, 4), max_displacement  = (20, 20, 0), p=0.5),
                # cmr_tran.ClipScaleRange(min_intensity= 0, max_intensity=1),
                # cmr_tran.ClipTanh(),
                # cmr_tran.ClipScaleRange(),
                # cmr_tran.ClipNormalize(min_intensity= 0, max_intensity=4000),
                # cmr_tran.ClipZscoreMinMax(min_intensity= 0, max_intensity=4000),
                
                # cmr_tran.RandomHorizontalFlip2D(p=0.7),
                # cmr_tran.RandomVerticalFlip2D(p=0.7),
                cmr_tran.UpdateLabels(source=TR_CLASS_MAP_MMS_SRS, destination=TR_CLASS_MAP_MMS_DES)

            ])
        else:
            train_transforms = Compose([
                # cmr_tran.Resample(self.opt.target_res,self.opt.target_res), #1.33
                # cmr_tran.CenterCrop2D((self.opt.crop_size,self.opt.crop_size)),
                # cmr_tran.RandomDilation_label_only(kernel_shape ='elliptical', kernel_size = 3, iteration_range = (1,2) , p=0.5),
                # cmr_tran.RandomRotation(degrees=90),
                # cmr_tran.RandomRotation(p=0.5),
                
                cmr_tran.ToTensor(),
                cmr_tran.NormalizeMinMaxpercentile(range=(-1,1), percentiles=(1,99)),
                # cmr_tran.NormalizeMinMaxRange(range=(-1,1)),
                

                # cmr_tran.PercentileBasedRescaling(out_min_max=(-1,1), percentiles=(1,99)),  #TODO: make sure the normalization is performed on the volume data not slice-by-slice
                # cmr_tran.RandomElasticTorchio_label_only(num_control_points  = (8, 8, 4), max_displacement  = (14, 14, 1), p=1),
                
                # cmr_tran.RandomElasticTorchio(num_control_points  = (8, 8, 4), max_displacement  = (20, 20, 0), p=0.5),
                # cmr_tran.ClipScaleRange(min_intensity= 0, max_intensity=4000),
                # cmr_tran.ClipTanh(),
                # cmr_tran.ClipScaleRange(),
                # cmr_tran.RandomHorizontalFlip2D(p=0.5),
                # cmr_tran.RandomVerticalFlip2D(p=0.5),
                cmr_tran.UpdateLabels(source=TR_CLASS_MAP_MMS_SRS, destination=TR_CLASS_MAP_MMS_DES)

            ])
        
        #if(opt.phase == 'test'):
            #self.cmr_dataset(cmr.MRI2DSegmentationDataset(self.msk_list, transform = train_transforms, slice_axis=2, canonical = False))

        self.cmr_dataset = cmr.MRI2DSegmentationDataset(self.filename_pairs, transform = train_transforms, slice_axis=2, canonical = False)
        
        
        size = len(self.cmr_dataset)
        self.dataset_size = size


    def __getitem__(self, index):
        # Label Image
        data_input = self.cmr_dataset[index]
        # if using instance maps
        if self.opt.no_instance:
            instance_tensor = 0
        else:
            instance_tensor = data_input["gt"] # the label map equals the instance map for this dataset
        if not self.opt.add_dist:
            dist_tensor = 0
        input_dict = {'label': data_input['gt'],
                      'image': data_input['input'],
                      'instance': instance_tensor,
                      'dist': dist_tensor,
                      'path': data_input['filename'],
                      'gtname': data_input['gtname'],
                      'index': data_input['index'],
                      'segpair_slice': data_input['segpair_slice'],
                      }

        return input_dict
    
    def __len__(self):
        return self.cmr_dataset.__len__()