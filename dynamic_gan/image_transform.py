import torch
import torch.nn
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

# takes a list of files to convert
def resizer(file_list, image_size):
    out_tensor = torch.zeros(3)
    file1 = open(file_list, 'r')
    jpegs = file1.readlines()
    print(jpegs)
    for x in jpegs:
        y = x.split('\n')
        image = Image.open(y[0])
        scaled = TF.resize(image, size=image_size) # scale image
        minDim = min(scaled.size) # find min dimension
        cropped = TF.center_crop(scaled, output_size=minDim) # use min dimension to crop
        
        # transform from image to tensor
        to_tensor = transforms.Compose([transforms.PILToTensor()])
        out_tensor = to_tensor(cropped)

    file1.close()
    return out_tensor
    

def image_displays(in_tensor):
        transform = transforms.ToPILImage()
        fig1 = plt.imshow(transform(in_tensor))
        plt.show()
        

print(image_displays(resizer('testjpegs.txt', 64))) # test image resize and conversion; display 1

