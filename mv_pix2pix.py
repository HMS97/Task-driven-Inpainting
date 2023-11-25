import cv2
import shutil

import os
import pathlib

from path import Path


for i in Path('/home/huimingsun/Desktop/rs_gan/pytorch-CycleGAN-and-pix2pix-master/results/LPN_RESULT/LPN_pix2pix/test_latest/images').files():
    if 'fake_B' in i :
        pathlib.Path('/home/huimingsun/Desktop/rs_gan/RS_INPAINTING/data/CVUSA/val_pt/pix2pix_street/',i.stem[:7]).mkdir(parents=True, exist_ok=True) 

        print(os.path.join('/home/huimingsun/Desktop/rs_gan/RS_INPAINTING/data/CVUSA/val_pt/pix2pix_street/',i.stem[:7],i.stem[:7] + '.jpg'))
        shutil.copy(i,os.path.join('/home/huimingsun/Desktop/rs_gan/RS_INPAINTING/data/CVUSA/val_pt/pix2pix_street/',i.stem[:7],i.stem[:7] + '.jpg'))


# pathlib.Path('/home/huimingsun/Desktop/rs_gan/RS_INPAINTING/data/CVUSA/val_pt/pix2pix_street_all/').mkdir(parents=True, exist_ok=True) 

# for i in Path('/home/huimingsun/Desktop/rs_gan/pytorch-CycleGAN-and-pix2pix-master/results/LPN_RESULT/LPN_pix2pix/test_latest/images').files():
#     if 'fake_B' in i :
#         shutil.copy(i,os.path.join('/home/huimingsun/Desktop/rs_gan/RS_INPAINTING/data/CVUSA/val_pt/pix2pix_street_all/',i.stem[:7] + '.jpg'))
