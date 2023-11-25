from dataset.rs_dataset import RS_Dataset 
from CRNet import Mygan
from torch.utils.data import TensorDataset, DataLoader, Dataset,SubsetRandomSampler
from tqdm import tqdm
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr
from Loss.ssim_loss import SSIM
from datetime import date
import torch
from path import Path
import os
from config import cfg
import json
from torchvision import transforms
from torch import nn
import warnings
warnings.filterwarnings("ignore",)
import json

transform = transforms.Compose(
                [   transforms.ToPILImage(),                  
                    transforms.Resize((cfg.SIZE,cfg.SIZE)),
                    transforms.ToTensor(),
                    # transforms.Normalize(cfg.MEAN,cfg.STD)
                    ])
mask_transform = transforms.Compose(
                [   
                    transforms.ToTensor(),
                    ])

clean_dir = Path('./data/rsscn7_clean')
mask_dir = Path('./data/rsscn7_map')                    
train_dataset = RS_Dataset(root = './data/rsscn7_thick/train_dataset', clean_dir=clean_dir, mask_dir=mask_dir,  Image_transform = transform ,Mask_transform = mask_transform)
valid_dataset = RS_Dataset(root = './data/rsscn7_thick/test_dataset',  clean_dir=clean_dir, mask_dir=mask_dir,  Image_transform = transform ,Mask_transform = mask_transform,maskmix = False )
  

train_loader = DataLoader(train_dataset,  batch_size= cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKER, pin_memory = True )
valid_loader = DataLoader(valid_dataset,  batch_size= cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKER, pin_memory = True )
data_loader = {'train_loader':train_loader, 'valid_loader':valid_loader}



def train(cfg):
    print(json.dumps(cfg, indent=2))
    epochs = cfg.EPOCH
    continue_epoch = 0


    model = Mygan(  cfg = cfg )
    model.setup() 


    model.load_networks('80')

    # model.load_networks('last')

    data_loader['train_loader'] = model.Accelerate(data_loader['train_loader'])

    correct = 0
    for index, data in enumerate(tqdm(data_loader['valid_loader'])):
        model.save_test = True
        model.save_image = True
        _,_,_,correct = model.test(data,correct)         # unpack data from dataset and apply preprocessing

    print( 100. * correct / len(data_loader['valid_loader'].dataset), len(data_loader['valid_loader'].dataset))
if __name__ == '__main__':

    cfg.SAVE_DIR = f'./saved_models/AUGMIX_cancel_change_model_normalized_class_weight_5_3_3_2023-02-03'

    train(cfg) 