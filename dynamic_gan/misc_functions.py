import os
from os import path, listdir, scandir

import torch
import torch.nn
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

# user it.resizer(filename)
def imglist_to_tensors(img_fnames, image_size, test_size):
    tensors = []
    
    for filename in img_fnames[:test_size]:
        image = Image.open(filename.strip())

        scaled = TF.resize(image, size=image_size) # scale image
        minDim = min(scaled.size) # find min dimension
        cropped = TF.center_crop(scaled, output_size=minDim) # use min dimension to crop
    
        # transform from image to tensor
        toTensor = transforms.Compose([transforms.PILToTensor()])
        out_tensor = toTensor(cropped)

        tensors.append(out_tensor)
    
    return tensors

def imgdir_to_tensors(img_dir, image_size, test_size):
    if not path.exists(img_dir) or not path.isdir(img_dir):
        return None
    
    img_fnames = []

    for f in scandir(img_dir):
        fpath = path.join(img_dir, f)
        if not path.isfile(fpath):
            continue
        
        img_fnames.append(fpath)
        if len(img_fnames) >= test_size:
            break

    return imglist_to_tensors(img_fnames, image_size, test_size)

def to_batches(dataset, batch_size):
    img_batches = []
    shift = 0
    while shift < len(dataset):
        stride = min(batch_size, len(dataset)-shift)
        remainder = batch_size - stride

        img_size = dataset[0].size()[1]

        batch = torch.Tensor(batch_size, 3, img_size, img_size) 
        
        batch_list = dataset[shift : shift+stride] + dataset[:remainder]

        for i, tensor in enumerate(batch_list):
            batch[i] = tensor

        img_batches.append(
            batch
        )
        shift += stride
    return img_batches


