import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torchvision import transforms
# from resnet_model import *
from .resnet_model import *
from .Unet import UnetGenerator


class BASNet(nn.Module):
    def __init__(self,n_channels,n_classes, res = False, seg = False):
        super(BASNet,self).__init__()
        self.seg = seg
        self.res = res

        resnet = models.resnet34(pretrained=True)


        ## -------------Encoder--------------


        self.inconv = nn.Conv2d(n_channels,64,3,padding=1)
        self.inbn = nn.InstanceNorm2d(64)
        self.inrelu = nn.ReLU(inplace=True)

        #stage 1
        self.encoder1 = resnet.layer1 #224
        #stage 2
        self.encoder2 = resnet.layer2 #112
        #stage 3
        self.encoder3 = resnet.layer3 #56
        #stage 4
        self.encoder4 = resnet.layer4 #28

        self.pool4 = nn.MaxPool2d(2,2,ceil_mode=True)



        ## -------------Bridge--------------

        #stage Bridge
        self.convbg_1 = nn.Conv2d(512,512,3,dilation=2, padding=2) # 7
        self.bnbg_1 = nn.InstanceNorm2d(512)
        self.relubg_1 = nn.ReLU(inplace=True)
        self.convbg_m = nn.Conv2d(512,512,3,dilation=2, padding=2)
        self.bnbg_m = nn.InstanceNorm2d(512)
        self.relubg_m = nn.ReLU(inplace=True)
        self.convbg_2 = nn.Conv2d(512,512,3,dilation=2, padding=2)
        self.bnbg_2 = nn.InstanceNorm2d(512)
        self.relubg_2 = nn.ReLU(inplace=True)

        ## -------------Decoder--------------

        #stage 4d
        self.conv4d_1 = nn.Conv2d(1024,512,3,padding=1) # 32
        self.bn4d_1 = nn.InstanceNorm2d(512)
        self.relu4d_1 = nn.ReLU(inplace=True)

        self.conv4d_2 = nn.Conv2d(512,256,3,padding=1)
        self.bn4d_2 = nn.InstanceNorm2d(256)
        self.relu4d_2 = nn.ReLU(inplace=True)

        #stage 3d
        self.conv3d_1 = nn.Conv2d(512,256,3,padding=1) # 64
        self.bn3d_1 = nn.InstanceNorm2d(256)
        self.relu3d_1 = nn.ReLU(inplace=True)

        self.conv3d_2 = nn.Conv2d(256,128,3,padding=1)
        self.bn3d_2 = nn.InstanceNorm2d(128)
        self.relu3d_2 = nn.ReLU(inplace=True)

        #stage 2d

        self.conv2d_1 = nn.Conv2d(256,128,3,padding=1) # 128
        self.bn2d_1 = nn.InstanceNorm2d(128)
        self.relu2d_1 = nn.ReLU(inplace=True)


        self.conv2d_2 = nn.Conv2d(128,64,3,padding=1)
        self.bn2d_2 = nn.InstanceNorm2d(64)
        self.relu2d_2 = nn.ReLU(inplace=True)

        #stage 1d
        self.conv1d_1 = nn.Conv2d(128,64,3,padding=1) # 256
        self.bn1d_1 = nn.InstanceNorm2d(64)
        self.relu1d_1 = nn.ReLU(inplace=True)

        self.conv1d_2 = nn.Conv2d(64,64,3,padding=1)
        self.bn1d_2 = nn.InstanceNorm2d(64)
        self.relu1d_2 = nn.ReLU(inplace=True)


        ## -------------Bilinear Upsampling--------------
        self.upscore6 = nn.Upsample(scale_factor=32,mode='bilinear')###
        self.upscore5 = nn.Upsample(scale_factor=16,mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=8,mode='bilinear')
        self.upscore3 = nn.Upsample(scale_factor=4,mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')
                        
        ## -------------Side Output--------------
        self.outconvb = nn.Conv2d(512,n_classes,3,padding=1)
        self.outconv6 = nn.Conv2d(512,n_classes,3,padding=1)
        self.outconv5 = nn.Conv2d(512,n_classes,3,padding=1)
        self.outconv4 = nn.Conv2d(256,n_classes,3,padding=1)
        self.outconv3 = nn.Conv2d(128,n_classes,3,padding=1)
        self.outconv2 = nn.Conv2d(64,n_classes,3,padding=1)
        self.outconv1 = nn.Conv2d(64,n_classes,3,padding=1)


        self.last_res = nn.Conv2d(3,n_classes,3,padding=1)




    def forward(self,x,seg=None,edge=None):

        hx = x
      
        ## -------------Encoder-------------
        hx = self.inconv(hx)
        hx = self.inbn(hx)
        hx = self.inrelu(hx)

        h1 = self.encoder1(hx) # 256
        h2 = self.encoder2(h1) # 128
        h3 = self.encoder3(h2) # 64
        h4 = self.encoder4(h3) # 32


        ## -------------Bridge-------------
        hx = self.relubg_1(self.bnbg_1(self.convbg_1(h4))) # 8   513*8*8
        hx = self.relubg_m(self.bnbg_m(self.convbg_m(hx)))
        hx = self.relubg_2(self.bnbg_2(self.convbg_2(hx)))


        ## -------------Decoder-------------

        hx = self.relu4d_1(self.bn4d_1(self.conv4d_1(torch.cat((hx,h4),1))))
        hd4 = self.relu4d_2(self.bn4d_2(self.conv4d_2(hx)))
        del h4
        hx = self.upscore2(hd4) # 32 -> 64

        hx = self.relu3d_1(self.bn3d_1(self.conv3d_1(torch.cat((hx,h3),1))))
        hd3 = self.relu3d_2(self.bn3d_2(self.conv3d_2(hx)))
        del h3
        hx = self.upscore2(hd3) # 64 -> 128

        hx = self.relu2d_1(self.bn2d_1(self.conv2d_1(torch.cat((hx,h2),1))))
        hd2 = self.relu2d_2(self.bn2d_2(self.conv2d_2(hx)))
        del h2
        hx = self.upscore2(hd2) # 128 -> 256

        hx = self.relu1d_1(self.bn1d_1(self.conv1d_1(torch.cat((hx,h1),1))))
        hd1 = self.relu1d_2(self.bn1d_2(self.conv1d_2(hx)))

        d1 = self.outconv1(hd1) # 256

        if self.res == True:

            d4 = self.outconv4(hd4)
            d4 = self.upscore4(d4) # 32->256

            d3 = self.outconv3(hd3)
            d3 = self.upscore3(d3) # 64->256

            d2 = self.outconv2(hd2)
            d2 = self.upscore2(d2) # 128->256

            hd1 = d1 + x[:,:3]
            d1 = self.last_res(hd1)
            return  d1, d2, d3, d4 
        else:
            return  d1


class BBnet(nn.Module):
    def __init__(self,in1_chs=4,in1_out=3,in2_chs=4,in2_out=3, num_classes=7, seg = False,fixed = True,classifier = 'vgg16'):
        super(BBnet,self).__init__()
        self.corase = BASNet(in1_chs,in1_out)
        self.refine = BASNet(in2_chs,in2_out, res = True)
        # self.refine = UnetGenerator (in2_chs,in2_out,8)
        self.fixed = True
        self.classifier_name = classifier
        # self.classifier = torch.load('/home/huimingsun/Desktop/rs_gan/RS_INPAINTING/CVPR_REBATTLE/proposed_no_seg_trian_0502/clean_model/project_rs/evaluate_models/rsscn7_2020-07-07_vgg16_93.79_baseline.pth')

        # if self.classifier_name == 'vgg16':
        # self.classifier = torch.load('/home/huimingsun/Desktop/rs_gan/RS_INPAINTING/CVPR_REBATTLE/proposed_no_seg_trian_0502/clean_model/project_rs/evaluate_models/rsscn7_2020-07-07_vgg16_93.79_baseline.pth')
        # elif self.classifier.classifier_name == 'resnet50':
        #     self.classifier = torch.load('/home/huimingsun/Desktop/rs_gan/RS_INPAINTING/CVPR_REBATTLE/proposed_no_seg_trian_0502/clean_model/project_rs/evaluate_models/rsscn7_resnet50_clean_94.21_baseline.pth')
        # elif self.classifier.classifier_name == 'alexnet':
        # self.classifier = torch.load('/home/huimingsun/Desktop/rs_gan/RS_INPAINTING/CVPR_REBATTLE/proposed_no_seg_trian_0502/clean_model/project_rs/evaluate_models/rsscn7_alexnet_clean_91.57_baseline.pth')

 
    def forward(self,x,input_copy=None,seg=None):
        # Normal_data = transforms.Normalize([0.4850, 0.4560, 0.4060], [0.2290, 0.2240, 0.2250])
        re_x = torch.cat((x,seg),1)
        # x = self.corase(torch.cat((x,seg),1))
        # re_x  = input_copy* (seg) + input_copy * (1-seg)
        # if seg != None  :
        #     re_x = torch.cat((re_x,seg),1)
        # del input_copy,seg
        re_x = self.refine(re_x)
        # for visual consistency
        return x, re_x,



# if __name__ == "__main__":
#     data = torch.randn((3,3,256,256))
#     seg = torch.randn((3,1,256,256))

#     train_augmentation = transforms.Compose([transforms.ToPILImage(),
#                                             transforms.Resize(8),
#                                             transforms.ToTensor(),
#     ])
#     sample = nn.Upsample(scale_factor = 1/32,mode='bilinear')
#     net = BASNet(3,3)
#     print(net(data,seg).shape)