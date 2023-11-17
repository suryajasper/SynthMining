import os

import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy

pretrained_pkl = 'FFHQ-pretrained.pkl'

def load_network():
    print(f'Loading model from {pretrained_pkl}')
    model = legacy.load_network_pkl(pretrained_pkl)
    print(model['training_set_kwargs'])

if __name__ == '__main__':
    load_network()