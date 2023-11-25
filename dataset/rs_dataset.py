
from path import Path
from torch.utils.data import TensorDataset, DataLoader, Dataset,SubsetRandomSampler
from torchvision.datasets import ImageFolder
import cv2
import numpy as np
from PIL import Image
from random import randint,sample
import os
import cv2
from torchvision import transforms
from util.util  import aug,aug_plus
import random 

class RS_Dataset(ImageFolder):
    # split image into 5 parts each part's partion is 0.6
    def __init__(self,root, clean_dir,mask_dir,Image_transform, Mask_transform, maskmix = False):
        super(RS_Dataset, self).__init__(root)
        self.Image_transform = Image_transform
        self.Mask_transform = Mask_transform
        self.maskmix = maskmix
        self.clean_path = Path(clean_dir)
        self.mask_path = Path(mask_dir)
    



    def __getitem__(self, index):
        input_path =  self.imgs[index][0]
        class_label = self.imgs[index][1]
        # data_A = self.transform(np.array(Image.open(path)))
        # data_B = self.transform(np.array(Image.open(self.get_clean(path))))

        MASK = np.zeros((256,256,1),dtype=np.float)
        data_B = self.Image_transform(np.array(Image.open(self.get_clean(input_path))))
        data_A = np.array(Image.open(self.get_clean(input_path)))
        mask = np.array( ((cv2.imread(self.get_mask(input_path),0)>10)*255).astype(np.uint8) )
        data_A = cv2.resize(data_A,(256,256))
        mask = cv2.resize(mask,(256,256))
        if  self.maskmix:
            mask = Image.fromarray(mask)
            mask_replace = aug_plus(mask,self.Mask_transform)>0

            data_A[mask_replace[0,:,:]==1] = 255
            MASK = mask_replace


        else:
            data_A[mask==255] = 255
            MASK = self.Mask_transform(mask)

        # print(data_A.shape, mask.shape, MASK.shape)
        # print(mask.shape)
        # print(mask[0,:,:].numpy().shape)
        # data_A = cv2.bitwise_not(data_A, data_A, mask=mask[0,:,:].numpy())

        data_A = self.Image_transform(data_A)

        return {'cloud': data_A, 'clean': data_B, 'mask': MASK, 'class_label':class_label, 'path': input_path}
    
        

    # def get_y(self,x): return  self.trainB_path + os.path.join(*x.split('/')[-3:])
    def get_clean(self,x):
        return   os.path.join(self.clean_path ,os.path.join(*str(x).split('/')[-3:]))
    def get_mask(self,x): 
        return  os.path.join(self.mask_path , os.path.join(*str(x).split('/')[-3:]))


    



class RSC_Dataset(Dataset):
    # split image into 5 parts each part's partion is 0.6
    def __init__(self,root, clean_dir,mask_dir,Image_transform, Mask_transform, maskmix = False):
        # super(RS_Dataset, self).__init__(root)
        self.Image_transform = Image_transform
        self.Mask_transform = Mask_transform
        self.maskmix = maskmix
        self.clean_path = Path(root)
        self.mask_path = Path(mask_dir)
    
        # print(len(self.clean_path.files()))

    def __len__(self):
        return len(self.clean_path.files())

    def __getitem__(self, index):
       
        input_path = self.clean_path.files()[index]
        MASK = np.zeros((256,256,1),dtype=np.float)
        data_B = self.Image_transform(np.array(Image.open(input_path)))
        # data_A = np.array(Image.open(self.get_clean(input_path)))
        # mask = np.array( ((cv2.imread(self.get_mask(input_path),0)>10)*255).astype(np.uint8) )
        # data_A = cv2.resize(data_A,(256,256))
        # mask = cv2.resize(mask,(256,256))
        label_list = ['a','b','c','d','e','f','g']
        class_label = label_list.index(self.clean_path.files()[index].name[0])

        return {'clean': data_B,  'class_label':class_label, 'path': input_path}
    
        

    
