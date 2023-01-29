import torch
import random as rd
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets

a = [2,5,34,3]
b = [2,5,3,4]

rd.shuffle(a)
rd.shuffle(b)
print(a)
print(b)


    