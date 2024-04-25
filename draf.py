# -*- coding: utf-8 -*-

import os
import sys
import time
import comet_ml
import torch
import torch.nn
import argparse
from PIL import Image
from tensorboardX import SummaryWriter
import numpy as np

from validate import validate
from data import create_dataloader
from earlystop import EarlyStopping
from networks.trainer import Trainer
from options.train_options import TrainOptions
from sklearn.metrics import accuracy_score, confusion_matrix


"""Currently assumes jpg_prob, blur_prob 0 or 1"""
def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.dataroot = '{}/{}/'.format(val_opt.dataroot, val_opt.val_split)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.jpg_method = ['pil']
    if len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]

    return val_opt



opt = TrainOptions().parse()
opt.dataroot = '{}/{}/'.format(opt.dataroot, opt.train_split)
opt.detect_method = 'Shading'

model = Trainer(opt)

data_loader = create_dataloader(opt)

opt.mode 

opt.dataroot

from tqdm import tqdm
for a,b,c in os.walk(r'mydataset/ShadingDB/shading/val/0_real'):
    if len(c)>0:
        for i in c:
            f = i.replace('_', '.jpg')
            os.rename(os.path.join(a,i), os.path.join(a,f))

remove = []
for a,b,c in os.walk(r'mydataset/ShadingDB/shading'):
    if len(c)>0:
        for i in c:
            f = os.path.join(a,i)
            f = f.replace('shading', 'rgb')
            if not os.path.isfile(f):
                print(f)
                #os.remove(os.path.join(a,i))
                


for param in model.model.origin.parameters():
    print(param.requires_grad)
    
    
    
import torch
from networks.resnet import resnet50

state_dict = torch.load('weights/model_1.pth',map_location=torch.device('cpu'))
state_dict = state_dict['model']
model1 = resnet50()
model1 = model1.load_state_dict(state_dict)

for last_layer_name in model1.state_dict():
    print(last_layer_name)
    
    
for i in state_dict['model']:
    print(i)
    




model.model.origin.layer4[1].conv1


model.model.head[0]



for name, param in model.model.named_parameters():
    print(name, param.requires_grad)

import torch
a = torch.load('weights/auxiliary/dct_mean')





