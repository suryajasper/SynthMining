import math
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
import numpy as np

class ScaledLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
    ):
        super(ScaledLinear, self).__init__()
        
        self.scale = math.sqrt(1 / in_features)

        self.linear = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        self.bias = self.linear.bias
        self.linear.bias = None

    def forward(self, x):
        x = self.linear(x)

        return self.scale * x + self.bias

class ScaledConv2D(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        kernel_size=4,
        stride=1,
        padding=0,
    ):
        super(ScaledConv2D, self).__init__()

        self.scale = math.sqrt(1 / (in_features * pow(kernel_size, 2)))

        self.conv2d = nn.Conv2d(in_features, out_features, kernel_size, stride, padding)
        nn.init.xavier_uniform_(self.conv2d.weight)
        nn.init.zeros_(self.conv2d.bias)

        self.bias = self.conv2d.bias
        self.conv2d.bias = None
    
    def forward(self, x):
        x = self.conv2d(x)
        
        return x + self.bias.reshape((1, self.bias.shape[0], 1, 1))

class AdaptiveInstanceNorm(nn.Module):
    def __init__(
        self, 
        num_channels, 
        w_dim,
    ):
        super().__init__()

        self.instance_norm = nn.InstanceNorm2d(num_channels)

        self.style_weight = ScaledConv2D(w_dim, num_channels)
        self.style_bias  = ScaledConv2D(w_dim, num_channels)
    
    def forward(self, x, latent_style):
        style_weight = self.style_weight(latent_style).unsqueeze(2).unsqueeze(3)
        style_bias = self.style_bias(latent_style).unsqueeze(2).unsqueeze(3)

        x_mean = torch.mean(x, dim=1, keepdim=True)
        x_stdev = torch.sqrt( torch.mean(x ** 2, dim=1, keepdim=True, dtype=torch.FloatTensor) )

        return style_weight * (x - x_mean) / x_stdev + style_bias

class FCMappingLayer(nn.Module):
    def __init__(
        self,
        z_dim,
        w_dim,
    ):
        super(FCMappingLayer, self).__init__()

        # self.ada_in = AdaptiveInstanceNorm(w_dim, num_channels)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=False)

        self.z_to_w = ScaledConv2D(
            in_features=z_dim,
            out_features=w_dim,
        )

        self.fully_connected = ScaledConv2D(
            in_features=w_dim,
            out_features=w_dim,
        )
    
    def forward(self, z):
        z = self.z_to_w(z)

        for _ in range(7):
            z = self.leaky_relu(z)
            z = self.fully_connected(z)
        
        return z

class NoiseLayer(nn.Module):
    def __init__(
        self,
        num_channels,
    ):
        super().__init__()

        self.num_channels = num_channels
        self.initialize_weights()

    def initialize_weights(self):
        weight_tensor = torch.zeros((1, self.num_channels, 1, 1))
        self.weights = nn.Parameter(weight_tensor)
    
    def forward(self, x):
        noise = torch.randn((x.shape[0], 1, x.shape[1], x.shape[2]))
        return x + self.weights + noise

class SynthesisBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        w_dim,
    ):
        super(SynthesisBlock, self).__init__()

        self.conv2d1 = ScaledConv2D(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=3,
            stride=2,
            padding=0,
        )

        self.inject_noise = NoiseLayer(num_channels=out_channels)

        self.ada_in = AdaptiveInstanceNorm(
            w_dim=w_dim, 
            num_channels=out_channels
        )

        self.conv2d2 = ScaledConv2D(
            in_features=out_channels,
            out_features=out_channels,
            kernel_size=4,
            stride=1,
            padding=0
        )

        self.activation = nn.LeakyReLU(
            negative_slope=0.2, 
            inplace=False
        )
    
    def forward(self, x, y):
        x = self.conv2d1(x)
        x = self.inject_noise(x)
        x = self.activation(x)
        x = self.ada_in(x, y)

        x = self.conv2d2(x)
        x = self.inject_noise(x)
        x = self.activation(x)
        x = self.ada_in(x, y)

        return x


