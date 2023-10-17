import os
import glob

import sys

# Append the path to the Python bindings in your build directory
sys.path.append('/home/users/sastocke/SimpleElastix/build/SimpleITK-build/Wrapping/Python')

import SimpleITK as sitk



def affine_register(ref_seg_fn, target_seg_fn, ref_img_fn, out_fn):
    ref_seg = sitk.ReadImage(ref_seg_fn)
    target_seg = sitk.ReadImage(target_seg_fn)
    ref_img = sitk.ReadImage(ref_img_fn)
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(target_seg)
    p_map_1 = sitk.GetDefaultParameterMap('translation')
    p_map_1['Metric'] = ['AdvancedMeanSquares']
    p_map_2 = sitk.GetDefaultParameterMap('affine')
    p_map_2['Metric'] = ['AdvancedMeanSquares'] 
    p_map_3 = sitk.GetDefaultParameterMap('bspline')
    p_map_3['Metric'] = ['AdvancedMeanSquares']
    p_map_3['MaximumNumberOfIterations'] = ['256']
    p_map_3['FinalGridSpacingInPhysicalUnits'] = []
    p_map_3["MaximumNumberOfSamplingAttempts"] = ['4']
    p_map_3["FinalGridSpacingInVoxels"] = ['18']
    p_map_3['FinalBSplineInterpolationOrder'] = ['2']
    elastixImageFilter.SetParameterMap(p_map_1)
    elastixImageFilter.AddParameterMap(p_map_2)
    elastixImageFilter.AddParameterMap(p_map_3)
    elastixImageFilter.SetMovingImage(ref_seg)
    elastixImageFilter.Execute()

    parameter_map = elastixImageFilter.GetTransformParameterMap()
    warp_ref = sitk.Transformix(ref_img, parameter_map)
    sitk.WriteImage(warp_ref, out_fn)

if __name__ == '__main__':
    seg_ref = '/scratch/users/sastocke/data/data/testmask128/ct_1129_image.nii.gz'
    img_ref = '/scratch/users/sastocke/data/data/testnormimages128/ct_1129_image.nii.gz'
    seg_curr_dir = "/scratch/users/fwkong/SharedData/Synthesized_correction_128"
    out_dir = "/scratch/users/fwkong/SharedData/Synthesized_correction_128_ref"
    seg_curr = glob.glob(os.path.join(seg_curr_dir, 'ct*.nii.gz'))
    for fn in seg_curr:
        out_fn = os.path.join(os.path.dirname(fn), 'ref_' + os.path.basename(fn).split('_')[-1])
        print(out_fn)
        if not os.path.exists(out_fn):
            affine_register(seg_ref, fn, img_ref, out_fn)