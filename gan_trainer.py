
from CRNet import Mygan
from dataset.rs_dataset import RS_Dataset 
from torch.utils.data import TensorDataset, DataLoader, Dataset,SubsetRandomSampler
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from datetime import date
import torch
from path import Path
from evaluation import evaluation
import os
from config import cfg
import json
from torchvision import transforms
import warnings
import json
from datetime import date
today = date.today()
warnings.filterwarnings("ignore", category=FutureWarning)

# transform = transforms.Compose(
#                 [   transforms.ToPILImage(),                  
#                     transforms.Resize((cfg.SIZE,cfg.SIZE)),
#                     transforms.ToTensor(),
#                     transforms.Normalize(cfg.MEAN,cfg.STD)
#                     ])

transform = transforms.Compose(
                [   transforms.ToPILImage(),                  
                    transforms.Resize((cfg.SIZE,cfg.SIZE)),
                    # transforms.AugMix(),
                    transforms.ToTensor(),
                    # transforms.Normalize(cfg.MEAN,cfg.STD)
                    ])


#adding AUGMIX to the transform
# from augmix import AugMix

mask_transform = transforms.Compose(
                [   
                    # transforms.ToPILImage(),                  
                    transforms.ToTensor(),
                    ])
                    

clean_dir = Path('data/rsscn7_clean')
mask_dir = Path('data/rsscn7_map')                    
train_dataset = RS_Dataset(root = 'data/rsscn7_thick/train_dataset', clean_dir=clean_dir, mask_dir=mask_dir,  Image_transform = transform ,Mask_transform = mask_transform, maskmix = False  )
valid_dataset = RS_Dataset(root = 'data/rsscn7_thick/test_dataset',  clean_dir=clean_dir, mask_dir=mask_dir,  Image_transform = transform ,Mask_transform = mask_transform, maskmix = False )

train_loader = DataLoader(train_dataset,  batch_size= cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKER, pin_memory = True )
valid_loader = DataLoader(valid_dataset,  batch_size= cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKER, pin_memory = True )
data_loader = {'train_loader':train_loader, 'valid_loader':valid_loader}




def train(cfg):
    print(json.dumps(cfg, indent=2))
    epochs = cfg.EPOCH
    continue_epoch = 0

    if cfg.load_epoch!=None:
        continue_epoch = int(cfg.load_epoch)

    # continue_epoch = 0
    
    model = Mygan(  cfg = cfg )
    data_loader['train_loader'] = model.Accelerate(data_loader['train_loader'])
    model.setup() 
    if cfg.load_epoch:
        model.load_networks(cfg.load_epoch)
    for epoch in range(continue_epoch + 1,continue_epoch + epochs+1):
        model.isTrain = True
        for index, data in enumerate(tqdm(data_loader['train_loader'])):
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters(epoch)
        if epoch%20 == 0:
            model.save_networks(str(epoch))
        
        model.save_networks('last')
        model.update_learning_rate(epoch)

    for index, data in enumerate(tqdm(data_loader['valid_loader'])):
        model.save_test = True
        model.save_image = True
        model.test(data)         # unpack data from dataset and apply preprocessing



def init():
    print(json.dumps(cfg, indent=2))
    fgepochs = cfg.EPOCH
    continue_epoch = 0
    model = Mygan(  cfg = cfg )
    model.setup() 

    return model

# def inception_train(inc_model,data):
#     inc_model.isTrain = True
#     inc_model.set_input(data)         # unpack data from dataset and apply preprocessing
#     inc_model.optimize_parameters()
#     inc_model.update_learning_rate()

#     if epoch%50 == 0:
#         model.save_networks(str(epoch)+'without_maskmix')

# def incepetion_train(cfg):
    


#     for epoch in range(continue_epoch + 1,continue_epoch + epochs+1):
#         for index, data in enumerate(tqdm(data_loader['train_loader'])):
           
        
        
#             model.save_networks('last')

if __name__ == '__main__':

    # cfg.CLASS_WEIGHT = 5
    # cfg.D1_LAYERS = 2
    # cfg.D2_LAYERS = 3
    # cfg.SAVE_DIR = f'saved_models/class_weight_5_pachgan_{cfg.D1_LAYERS}_{cfg.D2_LAYERS }_'+str(today)
    # train(cfg) 
    
    # cfg.CLASS_WEIGHT = 5
    # cfg.D1_LAYERS = 3
    # cfg.D2_LAYERS = 3
    # cfg.load_epoch = 'last'
    # cfg.SAVE_DIR = f'saved_models/normalized_class_weight_{cfg.CLASS_WEIGHT}_{cfg.D1_LAYERS}_{cfg.D2_LAYERS }_'+str(today) 
    # train(cfg) 
        

    # cfg.CLASS_WEIGHT = 0
    # cfg.D1_LAYERS = 3
    # cfg.D2_LAYERS = 3
    # cfg.SAVE_DIR = f'saved_models/normalized_class_weight_{cfg.CLASS_WEIGHT}_{cfg.D1_LAYERS}_{cfg.D2_LAYERS }_'+str(today) 
    # train(cfg)     

    # cfg.CLASS_WEIGHT = 5
    # cfg.D1_LAYERS = 3
    # cfg.D2_LAYERS = 3
    # cfg.load_epoch = 'last'

    # cfg.SAVE_DIR = f'saved_models/normalized_class_weight_{cfg.CLASS_WEIGHT}_{cfg.D1_LAYERS}_{cfg.D2_LAYERS }_'+str(today) 
    # train(cfg)   


    # evaluation(cfg.DATASET_NAME, os.path.join(cfg.SAVE_DIR,'predict_images'))
    # cfg.CLASS_WEIGHT = 10
    # cfg.D1_LAYERS = 3
    # # evaluation(cfg.DATASET_NAME, os.path.join(cfg.SAVE_DIR,'predict_images'))
    # cfg.CLASS_WEIGHT = 10
    # cfg.D1_LAYERS = 3
    # cfg.D2_LAYERS = 3
    # cfg.SAVE_DIR = f'saved_models/cancel_nochange_model_normalized_class_weight_{cfg.CLASS_WEIGHT}_{cfg.D1_LAYERS}_{cfg.D2_LAYERS }_'+str(today) 
    # train(cfg)   


    
    # cfg.CLASS_WEIGHT = 7
    # cfg.D1_LAYERS = 3
    # cfg.D2_LAYERS = 3
    # cfg.SAVE_DIR = f'saved_models/cancel_nochange_model_normalized_class_weight_{cfg.CLASS_WEIGHT}_{cfg.D1_LAYERS}_{cfg.D2_LAYERS }_'+str(today) 
    # train(cfg) 

    # cfg.CLASS_WEIGHT = 20
    # cfg.D1_LAYERS = 3
    # cfg.D2_LAYERS = 3
    # cfg.SAVE_DIR = f'saved_models/cancel_nochange_model_normalized_class_weight_{cfg.CLASS_WEIGHT}_{cfg.D1_LAYERS}_{cfg.D2_LAYERS }_'+str(today) 
    # train(cfg)   

    # cfg.CLASS_WEIGHT = 1
    # cfg.D1_LAYERS = 3
    # cfg.D2_LAYERS = 3

    # cfg.SAVE_DIR = f'saved_models/change_model_normalized_class_weight_{cfg.CLASS_WEIGHT}_{cfg.D1_LAYERS}_{cfg.D2_LAYERS }_'+str(today) 
    # train(cfg)     

    # cfg.CLASS_WEIGHT = 5
    # cfg.D1_LAYERS = 3
    # cfg.D2_LAYERS = 4
    # cfg.SAVE_DIR = f'saved_models/cancel_change_model_normalized_class_weight_{cfg.CLASS_WEIGHT}_{cfg.D1_LAYERS}_{cfg.D2_LAYERS }_'+str(today) 
    # train(cfg)  

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--perceptual', type=int, default=10, help='value for cfg.PERCEPTUAL')
    parser.add_argument('--g_d2', type=int, default=10, help='value for cfg.G_D2')
    parser.add_argument('--g_d1', type=int, default=10, help='value for cfg.G_D1')
    args = parser.parse_args()

    cfg.PERCEPTUAL = args.perceptual
    cfg.G_D2 = args.g_d2
    cfg.G_D1 = args.g_d1
    cfg.SAVE_DIR = f'saved_models/different_{cfg.G_D2}_{cfg.G_D1}_{cfg.PERCEPTUAL }_'+str(today) 
    train(cfg)  



# if __name__ == '__main__':
#     train(cfg) 
#     # evaluation(cfg.DATASET_NAME, os.path.join(cfg.SAVE_DIR,'predict_images'))
