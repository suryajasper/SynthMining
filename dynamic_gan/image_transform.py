import torch
import torch.nn
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def resizer(file_name):
    return_array = []
    file1 = open(file_name, 'r')
    jpegs = file1.readlines()
    print(jpegs)
    for x in jpegs:
        y = x.split('\n')
        image = Image.open(y[0])
        pil_to_torch = transforms.PILToTensor()
        output_array = pil_to_torch(image)
        sizing = list(output_array.size())
        z = 100000
        for i in range(1,3):
            if sizing[i] < z:
                z = sizing[i]
        resizer = transforms.Resize([z, z])
        output_array = resizer(output_array)
        output_array = torch.flatten(output_array)
        tuple1 = z,output_array
        return_array.append(tuple1)
    file1.close()
    return(return_array)
    

def image_displays(array_input):
        torch_to_pil = transforms.ToPILImage()
        fig1 = plt.imshow(torch_to_pil(array_input[1]))
        plt.show()
print(resizer("testjpegs"))


