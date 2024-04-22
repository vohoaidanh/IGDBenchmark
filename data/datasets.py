import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from random import random, choice
from io import BytesIO
from PIL import Image
from PIL import ImageFile
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import resize
from .process import processing


ImageFile.LOAD_TRUNCATED_IMAGES = True

def dataset_folder(opt, root):
    if opt.mode == 'binary':
        return binary_dataset(opt, root)
    if opt.mode == 'filename':
        return FileNameDataset(opt, root)

    raise ValueError('opt.mode needs to be [binary, filename, shading].')


class shading_dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, opt, split='train', rgb_dir='rgb', shading_dir = 'shading' ):
        """
        Parameters
        ----------
        opt : TYPE
            DESCRIPTION.
        split : [train, test, val]
            DESCRIPTION. The default is 'train'.
        rgb_dir : dir of RGB images
            DESCRIPTION. The default is 'rgb'.
        shading_dir : dir of shading images
            DESCRIPTION. The default is 'shading'.

        Returns Dataset
        -------

        """
        
        self.opt = opt
        self.root = opt.dataroot
        self.rgb_dir = rgb_dir
        self.shading_dir = shading_dir
        self.split = split
        
        real_rgb_name = os.listdir(os.path.join(self.root, self.rgb_dir, self.split, '0_real'))
        real_label_list = [0 for _ in range(len(real_rgb_name))]
        
        real_rgb_list = [os.path.join(self.root, self.rgb_dir, self.split, '0_real',i) \
                         for i in real_rgb_name if i.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
            
        fake_rgb_name = os.listdir(os.path.join(self.root, self.rgb_dir, self.split, '1_fake'))
        fake_rgb_list = [os.path.join(self.root, self.rgb_dir, self.split, '1_fake',i) \
                         for i in fake_rgb_name if i.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]        
        
        
        fake_label_list = [1 for _ in range(len(fake_rgb_name))]
                    
        self.input = real_rgb_list + fake_rgb_list
        self.shading = [i.replace(self.rgb_dir, self.shading_dir) for i in self.input]
        self.labels = real_label_list + fake_label_list

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        rgb  = Image.open(self.input[idx]).convert('RGB')
        shading = Image.open(self.shading[idx]).convert('RGB')
        
        target  = self.labels[idx]
        
        if (not self.opt.isTrain) and (not self.opt.isVal):
            rgb = custom_augment(rgb, self.opt)
            shading = custom_augment(shading, self.opt)
            
        if self.opt.detect_method.lower() in ['shading']:
            rgb = processing(rgb,self.opt,'imagenet')
            shading = processing(shading,self.opt,'imagenet')
            
        else:
            raise ValueError(f"Unsupported model_type: {self.opt.detect_method}")
        
        return rgb, shading, target

def binary_dataset(opt, root):
    if opt.isTrain:
        crop_func = transforms.RandomCrop(opt.cropSize)
    elif opt.no_crop:
        crop_func = transforms.Lambda(lambda img: img)
    else:
        crop_func = transforms.CenterCrop(opt.cropSize)

    if opt.isTrain and not opt.no_flip:
        flip_func = transforms.RandomHorizontalFlip()
    else:
        flip_func = transforms.Lambda(lambda img: img)
    if not opt.isTrain and opt.no_resize:
        rz_func = transforms.Lambda(lambda img: img)
    else:
        rz_func = transforms.Lambda(lambda img: custom_resize(img, opt))

    dset = datasets.ImageFolder(
            root,
            transforms.Compose([
                rz_func,
                transforms.Lambda(lambda img: data_augment(img, opt)),
                crop_func,
                flip_func,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]))
    return dset


class FileNameDataset(datasets.ImageFolder):
    def name(self):
        return 'FileNameDataset'

    def __init__(self, opt, root):
        self.opt = opt
        super().__init__(root)

    def __getitem__(self, index):
        # Loading sample
        path, target = self.samples[index]
        return path


def data_augment(img, opt):
    img = np.array(img)

    if random() < opt.blur_prob:
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sig)

    if random() < opt.jpg_prob:
        method = sample_discrete(opt.jpg_method)
        qual = sample_discrete(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}
def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)


rz_dict = {'bilinear': Image.BILINEAR,
           'bicubic': Image.BICUBIC,
           'lanczos': Image.LANCZOS,
           'nearest': Image.NEAREST}
def custom_resize(img, opt):
    interp = sample_discrete(opt.rz_interp)
    return TF.resize(img, opt.loadSize, interpolation=rz_dict[interp])

def custom_augment(img, opt):
    
    # print('height, width:'+str(height)+str(width))
    # resize
    if opt.noise_type=='resize':
        if opt.detect_method=='Fusing':
            height, width = img.shape[0], img.shape[1]
            img = resize(img, (int(height/2), int(width/2)))
        else:
            height, width = img.height, img.width
            img = torchvision.transforms.Resize((int(height/2),int(width/2)))(img) 

    img = np.array(img)
    # img = img[0:-1:4,0:-1:4,:]
    if opt.noise_type=='blur':
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sig)

    if opt.noise_type=='jpg':
        
        method = sample_discrete(opt.jpg_method)
        qual = sample_discrete(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)
    
    return Image.fromarray(img)

def loadpathslist(root,flag):
    classes =  os.listdir(root)
    paths = []
    if not '1_fake' in classes:
        for class_name in classes:
            imgpaths = os.listdir(root+'/'+class_name +'/'+flag+'/')
            for imgpath in imgpaths:
                paths.append(root+'/'+class_name +'/'+flag+'/'+imgpath)
        return paths
    else:
        imgpaths = os.listdir(root+'/'+flag+'/')
        for imgpath in imgpaths:
            paths.append(root+'/'+flag+'/'+imgpath)
        return paths