
import torch

from model.BASNetv7 import BBnet,SBBnet
from model.Unet import UnetGenerator
from torchvision import transforms
from torch import nn
from torch.nn import init
from torch.optim import lr_scheduler
# import neptune
from util.show_result import plot_generated
import numpy as np
import lpips
import os
from path import Path
from Loss.Dice import DiceLoss
import pathlib
from network import *
from PIL import Image
from Loss.ssim_loss import SSIM
import cv2
import torch.nn.functional as F
from accelerate import Accelerator
import neptune.new as neptune
from neptune.new.types import File
from torchvision import models

dice_loss = DiceLoss()
ssim = SSIM()


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss



def iou_numpy(outputs: np.array, labels: np.array):
    SMOOTH = 1e-6
    outputs = outputs.squeeze(1)
    labels = labels.squeeze(1)

    intersection = np.logical_and(outputs , labels).sum((1, 2))
    union = np.logical_or(outputs , labels).sum((1, 2))
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    
    thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10
    
    return thresholded.mean()




class Mygan(nn.Module):
    
    def __init__(self,  cfg ):
        super(Mygan,self).__init__()
        """Initialize the pix2pix class.
        """
        self.shuffle = 'train'
        self.cfg = cfg
        self.iteration = 0
        self.start_epoch = 0        
        self.count_times = 0
        self.iou_value = 0
        self.device = torch.device('cuda:{}'.format(self.cfg.GPU_IDS[0])) if self.cfg.GPU_IDS else torch.device('cpu')  # get device name: CPU or GPU

        self.lpips = lpips.LPIPS(net='vgg', verbose = False).to(self.device)
        self.optimizers = []
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.test_trans = transforms.Normalize([0.4850, 0.4560, 0.4060], [0.2290, 0.2240, 0.2250])
        self.loss_names = ['class', 'SSIM', 'G_GAN_D1','G_GAN_D2',  'G_L1', 'D1_real', 'D1_fake', 'D2_real', 'D2_fake','Dice','Gre_L1','Gre_Pec']
        self.suffix = 'predict_images'
        self.class_criterion =  F.cross_entropy
        self.acc_parallel = self.cfg.PARRALEL
        self.fixed = True
        # self.netS = init_net(UnetGenerator(self.cfg.SEG_IN, self.cfg.SEG_OUT, self.cfg.NUM_DOWNS), init_type='normal', init_gain=0.02, gpu_ids = self.cfg.GPU_IDS)


        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        
        # self.model_names = ['G', 'D1', 'D2']
        self.model_names = ['G']

        # define networks (both generator and discriminator)
        # self.netG = init_net(BBnet(in1_chs=self.cfg.GEN_IN1,in1_out=self.cfg.GEN_OUT1,in2_chs=self.cfg.GEN_IN2,in2_out=self.cfg.GEN_OUT2, classifier = self.cfg.CLASSIFIER), init_type='normal', init_gain=0.02, gpu_ids = self.cfg.GPU_IDS)
        self.netG = init_net(BBnet(in1_chs=self.cfg.GEN_IN1,in1_out=self.cfg.GEN_OUT1,in2_chs=self.cfg.GEN_IN2,in2_out=self.cfg.GEN_OUT2, ), init_type='normal', init_gain=0.02, gpu_ids = self.cfg.GPU_IDS)


        self.netD1 = define_D(input_nc = self.cfg.D1_IN, ndf = self.cfg.D1_NDF, netD = self.cfg.D1_TYPE, gpu_ids = self.cfg.GPU_IDS, n_layers_D = self.cfg.D1_LAYERS )

        self.netD2 = define_D(input_nc = self.cfg.D2_IN, ndf = self.cfg.D2_NDF, netD = self.cfg.D2_TYPE, gpu_ids = self.cfg.GPU_IDS, n_layers_D = self.cfg.D2_LAYERS )

        num_classes = 7
        model = models.vgg16(pretrained=True)
        model.classifier[-1] =  nn.Linear(in_features=4096, out_features=num_classes, bias=True)
        model.load_state_dict(torch.load('./saved_models/rsscn7_2021-05-16_vgg16_93.43_baseline.pth'))
        self.netC = model.eval()
        # self.netC = torch.load('/home/huimingsun/HUIMING_NAS/backup/rs_gan/RS_INPAINTING/CVPR_REBATTLE/GLNET/saved_models/rsscn7_2021-05-16_vgg16_93.43_baseline.pth').eval()
        # self.netC.classifier[-1] =  nn.Linear(in_features=4096, out_features=7, bias=True)
            
        # for param in self.model.parameters():
        #     if self.fixed:
        #         param.requires_grad = False
                
        # define loss functions
        self.criterionGAN = GANLoss('lsgan').to(self.device)
        self.criterionL1 = torch.nn.L1Loss()
        # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.cfg.LR, betas=(0.5, 0.999))
        self.optimizer_D1 = torch.optim.Adam(self.netD1.parameters(), lr=self.cfg.LR, betas=(0.5, 0.999))
        self.optimizer_D2 = torch.optim.Adam(self.netD2.parameters(), lr=self.cfg.LR, betas=(0.5, 0.999))
        # self.optimizer_C = torch.optim.Adam(self.netDC.parameters(), lr=self.cfg.LR, betas=(0.5, 0.999))

        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D1)
        self.optimizers.append(self.optimizer_D2)
        # self.optimizers.append(self.optimizer_C)

       

        for name in self.loss_names:
            if isinstance(name, str):
                setattr(self, 'loss_' + name, 0)
                setattr(self,  name, 0)



    def record_loss(self):
        """ update loss according train and valid status"""
        inv_normalize = transforms.Normalize(
        mean= [-m/s for m, s in zip(self.cfg.MEAN, self.cfg.STD)],
        std= [1/s for s in self.cfg.STD]
        )        
        tensor2np = lambda x:  np.array(transforms.ToPILImage()(inv_normalize(x[0].cpu()) ))


        if self.iteration % self.cfg.SHOW_INTERVAL == 0:

            # print(tensor2np(self.real_A)[0], tensor2np_R(self.Re)[0])
            image_dic = {'input':tensor2np(self.real_A),
                        'target':tensor2np(self.real_B),
                        # 'fake_C':tensor2np(self.fake_C),
                        'seg':  np.array(transforms.ToPILImage()( self.mask[0].cpu() )),
                        'coarse': tensor2np(self.coarse), 
                        're':tensor2np(self.Re)
            }
            plot_generated(image_dic)
            # if self.cfg.UPLOAD:
                # neptune.log_image(f'{self.shuffle}_images', 'example.png')

    def Accelerate(self,dataloader):
        self.acc_parallel = True
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.netG, self.optimizer_G, dataloader = self.accelerator.prepare(self.netG, self.optimizer_G, dataloader)
        self.netD1,self.netD2,self.netC = self.accelerator.prepare(self.netD1,self.netD2,self.netC)

        return dataloader

    def setup(self ):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.cfg.ISTRAIN:
            self.schedulers = [get_scheduler(optimizer, self.cfg.LR_POLICY, self.cfg.STEP_DEC) for optimizer in self.optimizers]
      

    def save_networks(self, which_epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (which_epoch, name)
                save_path = os.path.join(self.cfg.SAVE_DIR , save_filename).replace('\\', '/')
                net = getattr(self, 'net' + name)
                optimize = getattr(self, 'optimizer_' + name)
                pathlib.Path(os.path.join(self.cfg.SAVE_DIR )).mkdir(parents=True, exist_ok=True) 

                if len(self.cfg.GPU_IDS) > 0 and torch.cuda.is_available():
                    torch.save({'net': net.cpu().state_dict(), 'optimize': optimize.state_dict()}, save_path)
                    net.cuda(self.cfg.GPU_IDS[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    # helper loading function that can be used by subclasses
    def load_networks(self, which_epoch):
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (which_epoch, name)
                load_path = os.path.join(self.cfg.SAVE_DIR , load_filename)
                print('load model from ', load_path)
                net = getattr(self, 'net' + name)
                optimize = getattr(self, 'optimizer_' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path.replace('\\', '/'), map_location=str(self.device))
                # optimize.load_state_dict(state_dict['optimize'])
                net.load_state_dict(state_dict['net'])


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        # AtoB = self.direction == 'AtoB'
        self.real_A = input['cloud'].to(self.device,dtype = torch.float)
        self.real_B = input['clean'].to(self.device,dtype = torch.float)
        self.real_C = input['mask'].to(self.device,dtype = torch.float)
        self.mask = input['mask'].to(self.device,dtype = torch.float)
        self.class_label  = input['class_label'].to(self.device)
        self.paths = input['path']

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def train(self):
        """Make models eval mode during train time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()
    

    def get_G_loss(self):
        fake_AB = torch.cat((self.real_A, self.coarse), 1)
        pred_fake = self.netD1(fake_AB)
        self.loss_G_GAN_D1 = self.criterionGAN(pred_fake, True)* self.cfg.G_D1
        self.loss_G_L1 = self.criterionL1(self.coarse, self.real_B) * self.cfg.COARSE_G_L1

        # Second, G(A) should fake the D2 with Re results 
        # fake_AB = torch.cat((self.real_A*(1-self.mask), self.Re*self.mask), 1)
        fake_AB = torch.cat((self.real_A*(1-self.mask), self.Re*self.mask), 1)

        pred_fake = self.netD2(fake_AB)
        self.loss_G_GAN_D2 = self.criterionGAN(pred_fake, True)* self.cfg.G_D2

        self.loss_Gre_Pec = self.lpips(self.Re, self.real_B)* self.cfg.PERCEPTUAL
        self.loss_Gre_L1 = self.criterionL1(self.Re, self.real_B) * self.cfg.REFINE_G_L1
        self.loss_class = self.class_criterion(self.class_pred, self.class_label)*self.cfg.CLASS_WEIGHT

        # self.loss_SSIM = self.fusion_loss(self.Re, self.d2, self.d3, self.d4, self.real_B)*self.cfg.SSIM
        # self.loss_SSIM = 0
        self.loss_G = self.loss_G_GAN_D1 + self.loss_G_L1 + self.loss_Gre_Pec + self.loss_Gre_L1  + self.loss_G_GAN_D2 + self.loss_class 
        return self.loss_G
        
    def test(self,input, correct = None):
        """
        Forward function used in test time.
        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        inv_normalize = transforms.Normalize(
                mean= [-m/s for m, s in zip(self.cfg.MEAN, self.cfg.STD)],
                std= [1/s for s in self.cfg.MEAN]
                )        
        with torch.no_grad():
            """Run forward pass; called by both functions <optimize_parameters> and <test>."""
            self.netG.eval()
            self.set_input(input)

            self.mask,self.coarse,self.Re = self.forward()

            if  correct != None:            
                

                # self.Re = self.Re* self.mask + self.real_B* (1-self.mask)
                # self.coarse = self.coarse* self.mask + self.real_B* (1-self.mask)

                # self.class_pred = self.netC((self.Re-0.5)/ 0.5) 
                pred = self.class_pred.max(1, keepdim=True)[1] # get the index of the max log-probability
                self.test_loss = F.cross_entropy(self.class_pred, self.class_label)
                correct += pred.eq(self.class_label.view_as(pred)).sum().item()

                # print(self.class_label.view_as(pred))
                # print(pred.eq(self.class_label.view_as(pred)).sum().item())
        
            if self.cfg.SAVE_IMGS:
                inv_normalize = transforms.Normalize(
                mean= [-m/s for m, s in zip(self.cfg.MEAN, self.cfg.STD)],
                std= [1/s for s in self.cfg.STD]
                )        
                # tensor2np = lambda x, j:  np.array(transforms.ToPILImage()(inv_normalize(x[j].cpu()) ))
                tensor2np = lambda x, j:  np.array(transforms.ToPILImage()(x[j].cpu() ))

                

                for j in range(self.real_B.shape[0]):
                    # pathlib.Path(os.path.join(self.cfg.SAVE_DIR , self.suffix+ '_coarse' )).mkdir(parents=True, exist_ok=True) 
                    pathlib.Path(os.path.join(self.cfg.SAVE_DIR , self.suffix )).mkdir(parents=True, exist_ok=True) 
                    pathlib.Path(os.path.join(self.cfg.SAVE_DIR , self.suffix + '_coarse' )).mkdir(parents=True, exist_ok=True) 

                    pathlib.Path(os.path.join(self.cfg.SAVE_DIR , 'seg_mask',)).mkdir(parents=True, exist_ok=True) 

                    Image.fromarray(tensor2np(self.Re,j)).save(os.path.join(self.cfg.SAVE_DIR , self.suffix , Path(self.paths[j]).name))
                    Image.fromarray(tensor2np(self.coarse,j)).save(os.path.join(self.cfg.SAVE_DIR , self.suffix + '_coarse' , Path(self.paths[j]).name))

                    # print(self.mask[j].cpu().squeeze().numpy().shape)
                    cv2.imwrite(os.path.join(self.cfg.SAVE_DIR , 'seg_mask', Path(self.paths[j]).name),self.mask[j].cpu().squeeze().numpy()*255)


        return self.mask, self.coarse, self.Re, correct



    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # self.seg = self.netS(self.real_A)  # G(A)
        # self.mask = (self.seg > self.cfg.SEG_THRD).to(torch.float32)

        self.coarse,self.Refine = self.netG(self.real_A, self.real_A, self.mask)  # G(A)
        self.Re, self.d2, self.d3, self.d4 = self.Refine
        # self.Re = self.Re* self.mask + self.real_B* (1-self.mask)
        # self.class_pred = self.netC((self.Re-0.5)/ 0.5)
        self.class_pred = self.netC(self.Re)
  






        return self.mask,self.coarse,self.Re
    



    def fusion_loss(self, d1, d2, d3, d4,  target):

        loss, loss0, loss1, loss2, loss3, loss4, loss5, loss6 = 0,0,0,0,0,0,0,0
        loss0  = 1 - ssim(d1, target)
        loss1  = 1 - ssim(d2, target)
        loss2  = 1 - ssim(d3, target)
        loss3  = 1 - ssim(d4, target)
        # loss4  = 1 - ssim(d5, target)
        # loss5  = 1 - ssim(d6, target)

        loss += loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6 
        return loss 

    def backward_D1(self):
        """Calculate GAN loss for the discriminator"""

        # Fake; stop backprop to the generator by detaching coarse
        # print((self.seg.detach() > 0.5).to(torch.int16) )
        # print( torch.logical_not((self.seg.detach() > 0.5).to(torch.int16)) )
    
        self.fake_C  = self.coarse * self.mask + self.real_A.detach() * (1-self.mask)

        fake_AB = torch.cat((self.real_A, self.fake_C), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD1(fake_AB.detach())
        self.loss_D1_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD1(real_AB)
        self.loss_D1_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D1 = (self.loss_D1_fake + self.loss_D1_real) * self.cfg.D1
        self.loss_D1.backward()  




    def backward_D2(self):
        """Calculate GAN loss for the discriminator"""

      
        self.fake_C  = self.Re * self.mask + self.real_A.detach() * (1-self.mask)

        fake_AB = torch.cat((self.real_A, self.fake_C), 1) 
        
        pred_fake = self.netD2(fake_AB.detach())
        self.loss_D2_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD2(real_AB)
        self.loss_D2_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D2 = (self.loss_D2_fake + self.loss_D2_real) * self.cfg.D2
        self.loss_D2.backward()  



    def backward_G(self, extra_loss = None):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the D1 and update coarse network with coarse results 
        fake_AB = torch.cat((self.real_A, self.coarse), 1)
        pred_fake = self.netD1(fake_AB)
        self.loss_G_GAN_D1 = self.criterionGAN(pred_fake, True)* self.cfg.G_D1
        self.loss_G_L1 = self.criterionL1(self.coarse, self.real_B) * self.cfg.COARSE_G_L1

        # Second, G(A) should fake the D2 with Re results 
        # fake_AB = torch.cat((self.real_A*(1-self.mask), self.Re*self.mask), 1)
        fake_AB = torch.cat((self.real_A*(1-self.mask), self.Re*self.mask), 1)

        pred_fake = self.netD2(fake_AB)
        self.loss_G_GAN_D2 = self.criterionGAN(pred_fake, True)* self.cfg.G_D2

        self.loss_Gre_Pec = self.lpips(self.Re, self.real_B)* self.cfg.PERCEPTUAL
        self.loss_Gre_L1 = self.criterionL1(self.Re, self.real_B) * self.cfg.REFINE_G_L1
        self.loss_class = self.class_criterion(self.class_pred, self.class_label)*self.cfg.CLASS_WEIGHT

        # self.loss_SSIM = self.fusion_loss(self.Re, self.d2, self.d3, self.d4, self.real_B)*self.cfg.SSIM
        # self.loss_SSIM = 0
        self.loss_G = self.loss_G_GAN_D1 + self.loss_G_L1 + self.loss_Gre_Pec + self.loss_Gre_L1  + self.loss_G_GAN_D2 + self.loss_class 

        # combine loss and calculate gradients
        if self.acc_parallel:
            self.accelerator.backward(self.loss_G)    
        else:
            self.loss_G.backward()
        
    def backward_C(self):
        self.loss_class.backward()



    def optimize_parameters(self, epoch, extra_loss = None):
        if self.cfg.ISTRAIN!=True:
            with torch.no_grad():
                self.eval()
                self.forward()                   # compute fake images: G(A)
        else:
            self.train()
            self.forward()


            # update D1
            self.set_requires_grad(self.netD1, True)  # enable backprop for D
            self.optimizer_D1.zero_grad()     # set D's gradients to zero
            self.backward_D1()                # calculate gradients for D
            self.optimizer_D1.step()          # update D's weights

            # update D2
            self.set_requires_grad(self.netD2, True)  # enable backprop for D
            self.optimizer_D2.zero_grad()     # set D's gradients to zero
            self.backward_D2()                # calculate gradients for D
            self.optimizer_D2.step()          # update D's weights

            # update G
            self.set_requires_grad(self.netD1, False)  # D requires no gradients when optimizing G
            self.set_requires_grad(self.netD2, False)  # D requires no gradients when optimizing G

            self.optimizer_G.zero_grad()        # set G's gradients to zero
            self.backward_G(extra_loss)                   # calculate graidents for G
            self.optimizer_G.step()# udpate G's weights

        self.iteration += 1
        self.record_loss()


    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


    def update_learning_rate(self, epoch ):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']

        for scheduler in self.schedulers:
            if self.cfg.LR_POLICY == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']

        print(f'current epoch {epoch}  learning rate %.7f -> %.7f' % (old_lr, lr))


