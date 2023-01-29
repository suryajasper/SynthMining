import os
import torch
import torch.nn
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

import image_transform as it # adarsh code


# user it.resizer(filename)
def to_tensor(file_list, image_size):
    img_dir_name = './dynamic_gan/celeba/img_align_celeba'
    tensors = []
    file1 = open(file_list, 'r')
    for filename in file1.readlines()[:1000]:
        filename = filename.strip()
        out_tensor = torch.zeros(3)
        image = Image.open(os.path.join(img_dir_name,filename))
        scaled = TF.resize(image, size=image_size) # scale image
        minDim = min(scaled.size) # find min dimension
        cropped = TF.center_crop(scaled, output_size=minDim) # use min dimension to crop
    
        # transform from image to tensor
        to_tensor = transforms.Compose([transforms.PILToTensor()])
        out_tensor = to_tensor(cropped)

        tensors.append(out_tensor)
    file1.close()
    return tensors

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


