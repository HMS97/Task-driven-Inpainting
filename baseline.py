from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset,SubsetRandomSampler
from torchvision import models
import time
from tqdm import tqdm
import numpy as np
from PIL import Image
from path import Path
def train(PARAMS, model, criterion, device, train_loader, optimizer, epoch):
    t0 = time.time()
    model.train()
    correct = 0

    for batch_idx, (img, target) in enumerate(tqdm(train_loader)):
        img,  target = img.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(img)

        loss = criterion(output, target )
        loss.backward()
        optimizer.step()
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
    

    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} , {:.2f} seconds'.format(
        epoch, batch_idx * len(img), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item(),time.time() - t0))


def test(PARAMS, model,criterion, device, test_loader,optimizer,epoch,best_acc):
    model.eval()
    test_loss = 0
    correct = 0

    example_images = []
    with torch.no_grad():
        # for batch_idx, (img, target,path) in enumerate(tqdm(test_loader)):
        for batch_idx, (img, target,path) in enumerate(tqdm(test_loader)):

            img, target = img.to(device), target.to(device)
            output = model(img)

            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            # Save the first input tensor in each test batch as an example image
            # for val in range(len(pred.flatten())):
            #     if (target[val] != pred[val]):
            #         print(path[val], pred[val], target[val] , classes[pred[val]] ) 

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
 

    current_acc = 100. * correct / len(test_loader.dataset)
    return current_acc

def main(data_path = None):

    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--model', type=str, default = 'resnet50')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--evaluate_model', type=str)
    parser.add_argument('--dataset', type=str, default='rsscn7')

    args = parser.parse_args()

    PARAMS = {'DEVICE': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                'bs': 8,
                'epochs':50,
                'lr': 0.0006,
                'momentum': 0.5,
                'log_interval':10,
                'criterion':'cross_entropy',
                'model_name': args.model,
                'dataset': args.dataset,
                }


    # Training settings
    train_transform = transforms.Compose(
                    [ 
                        transforms.RandomHorizontalFlip(),
                         transforms.ColorJitter(0.4, 0.4, 0.4),
                        transforms.Resize((256,256)),
                        transforms.ToTensor(),
                        # transforms.Normalize([0.4850, 0.4560, 0.4060], [0.2290, 0.2240, 0.2250])
                        # transforms.Normalize([0.5]*3, [0.5]*3)
                        
                        ])
    test_transform = transforms.Compose(
                    [ 
                        transforms.Resize((256,256)),
                        transforms.ToTensor(),
                        # transforms.Normalize([0.4850, 0.4560, 0.4060], [0.2290, 0.2240, 0.2250])
                        transforms.Normalize([0.5]*3, [0.5]*3)

                        ])

    class RS_Dataset(Dataset):
        def __init__(self, root, transform=None, partion = 0.6, size = 224 ,):
            # super(RS_Dataset, self).__init__(root, transform)
            self.transform = transform
            self.partion = partion
            self.size = size
            self.width,self.length = int(self.size*self.partion),int(self.size*self.partion)
            self.imgs = Path(root).files()
            self.classes = ['a','b','c','d','e','f','g']

        def __getitem__(self, index):
            img = Image.open(self.imgs[index])
            img = self.transform(img)
            label = self.find_classes(self.imgs[index].name[0])
            path = self.imgs[index]
            return img, label, path

        def __len__(self):
            return len(self.imgs)
        #convert string to label in pytorch dataset

        def find_classes(self, CLASS):
            
            class_to_idx = {classes[i]: i for i in range(len(classes))}
            return class_to_idx[CLASS]


    if args.dataset == 'rsscn7':

        train_dataset = RS_Dataset(root=data_path,transform = test_transform)
        test_dataset = RS_Dataset(root=data_path,transform = test_transform)


    elif args.dataset == 'ucm':
        train_dataset = datasets.ImageFolder(root='data/ucm/train_dataset/',transform = train_transform)
        test_dataset = datasets.ImageFolder(root='data/ucm/test_dataset/',transform = test_transform)
    train_loader = DataLoader(train_dataset,  batch_size=PARAMS['bs'], shuffle=True, num_workers=4, pin_memory = True )
    test_loader =  DataLoader(test_dataset, batch_size=PARAMS['bs'], shuffle=True,  num_workers=4, pin_memory = True  )

    global classes
    classes = train_dataset.classes
    num_classes = len(train_dataset.classes)
    if PARAMS['model_name'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.classifier[-1] =  nn.Linear(in_features=4096, out_features=num_classes, bias=True)
    elif PARAMS['model_name'] == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.fc =  nn.Linear(in_features=2048, out_features=num_classes, bias=True)
    elif PARAMS['model_name'] == 'alexnet':
        model = models.alexnet(pretrained=True)
        model.classifier[-1] =  nn.Linear(in_features=4096, out_features=num_classes, bias=True)    

    
   
    model = model.to(PARAMS['DEVICE'])   
    optimizer = optim.SGD(model.parameters(), lr=PARAMS['lr'], momentum=PARAMS['momentum'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 7, gamma = 0.9)
    criterion =  F.cross_entropy
    acc = 0

    num_classes = 7
    model = models.vgg16(pretrained=True)
    model.classifier[-1] =  nn.Linear(in_features=4096, out_features=num_classes, bias=True)
    evaluate_model = './saved_models/rsscn7_2021-05-16_vgg16_93.43_baseline.pth'
    model.load_state_dict(torch.load(evaluate_model))
    model = model.cuda().eval()
    # model = torch.load('/home/huimingsun/HUIMING_NAS/backup/rs_gan/proposed_no_seg_trian_0502/clean_model/project_rs/evaluate_models/rsscn7_2020-07-07_vgg16_93.79_baseline.pth')

    # print(test_dataset.class_to_idx)
    acc = test(PARAMS, model,criterion, PARAMS['DEVICE'], test_loader, optimizer, 0, acc)
    print(f'the evalutaion acc is {acc}')
    return acc

if __name__ == '__main__':
    main('./saved_models/AUGMIX_cancel_change_model_normalized_class_weight_5_3_3_2023-02-03/predict_images')
