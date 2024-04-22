import os
import torch
import torch.nn as nn
from networks.resnet import resnet50
from networks.resnet_shading import resnet50_shading


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def unnormalize(tens, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # assume tensor of shape NxCxHxW
    return tens * torch.Tensor(std)[None, :, None, None] + torch.Tensor(
        mean)[None, :, None, None]


def get_model(opt):
    if opt.detect_method.lower()  in ["cnndetection","cnnsport","dire"]:
        if opt.isTrain:
            model = resnet50(pretrained=True)
            model.fc = nn.Linear(2048, 1)
            torch.nn.init.normal_(model.fc.weight.data, 0.0, opt.init_gain)
            return model
        else:
            return resnet50(num_classes=1)

    elif opt.detect_method.lower() == "shading":
        if opt.isTrain:
            model = resnet50_shading(num_classes=1)
            return model
        else:
            return resnet50_shading(num_classes=1, pretrained=False)
    
    else:
        raise ValueError(f"Unsupported model_type: {opt.detect_method}")
        
        
        
        










