from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np
import torch.nn as nn

from Modules import resnet,cait
import dataloader

class ResNet(nn.Module):
    def __init__(self, model_name, zero_init_residual=True,pretrained=True, device=None):
        super(ResNet, self).__init__()
        self.Extractor = resnet.get_convnet(model_name, zero_init_residual=zero_init_residual, pretrained=pretrained)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, 200)
        self.device = device
    
    def initialize(self, center_initialize = True):
        if center_initialize:
            # initialing classifer weight by the average of the corresponding class training data feature
            print("initialing classifer weight by feature mean ...")
            dataset = dataloader.CUB200(is_train=True)
            classes_ids = np.array(dataset.classes_id) 

            transform = transforms.Compose(dataset.base_transform) 
            flip_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=1.), *dataset.base_transform])
            
            self.eval()
            for class_num in range(200):
                idxes = np.where(classes_ids == class_num)[0]
                imgs = []
                for idx in idxes:
                    imgs.append(transform(dataset.imgs[idx]).unsqueeze(0))
                    imgs.append(flip_transform(dataset.imgs[idx]).unsqueeze(0))
                imgs = torch.from_numpy(np.concatenate(imgs)).to(self.device)

                with torch.no_grad():
                    normalized_feature = torch.nn.functional.normalize(self.extract_feature(imgs), dim=1)
                    self.fc.weight[class_num] = torch.mean(normalized_feature, dim = 0)
        else:
            nn.init.kaiming_normal_(self.fc.weight, nonlinearity="linear")
        nn.init.constant_(self.fc.bias, 0.)
    
    def forward(self, x):
        x = self.Extractor(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def extract_feature(self, x):
        x = self.Extractor(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class CaiT(nn.Module):
    def __init__(self, model_name,pretrained=True, device=None):
        super(CaiT, self).__init__()
        self.Extractor = cait.get_convnet(model_name, pretrained=pretrained)
        self.fc = nn.Linear(384, 200)
        self.device = device
    
    def initialize(self,center_initialize=False):
        if center_initialize:
            # initialing classifer weight by the average of the corresponding class training data feature
            print("initialing classifer weight by feature mean ...")
            dataset = dataloader.CUB200(is_train=True)
            classes_ids = np.array(dataset.classes_id) 

            transform = transforms.Compose(dataset.base_transform) 
            flip_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=1.), *dataset.base_transform])
            
            self.eval()
            for class_num in range(200):
                idxes = np.where(classes_ids == class_num)[0]
                imgs = []
                for idx in idxes:
                    imgs.append(transform(dataset.imgs[idx]).unsqueeze(0))
                    imgs.append(flip_transform(dataset.imgs[idx]).unsqueeze(0))
                imgs = torch.from_numpy(np.concatenate(imgs)).to(self.device)

                with torch.no_grad():
                    normalized_feature = torch.nn.functional.normalize(self.extract_feature(imgs), dim=1)
                    self.fc.weight[class_num] = torch.mean(normalized_feature, dim = 0)
        else:
            nn.init.kaiming_normal_(self.fc.weight, nonlinearity="linear")
        nn.init.constant_(self.fc.bias, 0.)
    
    def forward(self, x):
        x = self.Extractor.forward_features(x)
        x = self.fc(x)
        return x

    def extract_feature(self, x):
        x = self.Extractor.forward_features(x)
        return x