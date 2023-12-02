# Synthetic Image Generation to Aid Segmentation of Congenital Heart Disease Patient Images

Welcome to the GitHub repository dedicated to advancing the segmentation of congenital heart disease (CHD) patient images through synthetic image generation. This project aims to address the challenges posed by the rarity of CHD and the resultant data scarcity in machine learning applications, by generating synthetic yet anatomically accurate CT images of CHD patients.
This work builds on the work of Amirrajab et al. Please check out their github and their paper: https://github.com/sinaamirrajab/CardiacPathologySynthesis, https://arxiv.org/abs/2209.04223 

## Branches Overview
- **Master**: This branch contains the fully functional 2D image generation architecture.
- **3dfinalgantesting**: Here, you'll find the working 3D architecture that represents the culmination of our research in extending the styleSPADE GANs into full 3D image generation. 

## Getting Started
To start training the models, simply run the `cmr_train.py` script. Ensure that the configuration files are set up according to your requirements for a seamless training experience.

## Visual Demonstrations
We have included two GIFs to visually demonstrate our process:

![Masks Used in Image Generation](./gifs/mask_axia_view.gif)
![Resulting Synthesized Images](./gifs/real_image_axial_view.gif)

These GIFs illustrate the masks used in our image generation process and the resulting synthesized images, showcasing the accuracy and realism we achieve.

Thank you for your interest in our project, and we hope our research contributes to significant advancements in the field of medical imaging and treatment of congenital heart disease.
