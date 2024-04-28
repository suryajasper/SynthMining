import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import math

class Generator(nn.Module):
    def __init__(self,
                 n_categories,
                 noise_size,
                 channels=3,
                 feature_map_size=64,
                 embedding_size=50,
                 img_size=64):  # Assuming default power of 2 as 64x64 image size

        super(Generator, self).__init__()

        ngf = feature_map_size
        power = int(math.log2(img_size))
        
        self.label_embedding = nn.Embedding(n_categories, embedding_size)
        
        layers = []
        prev_channels = noise_size + embedding_size  # Initial input channels
        out_channels = (ngf * 8)
        
        for i in range(power-1):
            layers.append(nn.ConvTranspose2d(
                in_channels=prev_channels, 
                out_channels=out_channels,  # out_channels is still itself
                kernel_size=4, 
                stride=2, 
                padding=1, 
                bias=False
            ))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(True))
            prev_channels = out_channels  # Update input channels for the next layer
            out_channels //= 2
        
        # Output layer
        layers.append(nn.ConvTranspose2d(
            in_channels=prev_channels, 
            out_channels=channels, 
            kernel_size=4, 
            stride=2, 
            padding=1, 
            bias=False
        ))
        layers.append(nn.Tanh())

        self.main = nn.Sequential(*layers)

    def forward(self, noise, label):
        label_embedding = self.label_embedding(label).view(label.size(0), -1, 1, 1)
        combined_input = torch.cat((noise, label_embedding), 1)
        output = self.main(combined_input)
        return output

# Discriminator currently has same isues*
# Kernel size 4x4 can't be larger than input 1x1

class Discriminator(nn.Module):
    def __init__(self,
                 n_categories,
                 channels=3,
                 feature_map_size=64,
                 embedding_size=50,
                 img_size=64):
        super(Discriminator, self).__init__()

        ndf = feature_map_size
        power = int(math.log2(img_size))
        
        self.label_embedding = nn.Embedding(n_categories, embedding_size)
        
        layers = []
        prev_channels = channels + embedding_size  # Initial input channels
        for i in range(power-1):
            out_channels = (ndf * 8) // (2 ** i)
            layers.append(nn.Conv2d(
                in_channels=prev_channels, 
                out_channels=out_channels, 
                kernel_size=4, 
                stride=2, 
                padding=1, 
                bias=False
            ))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev_channels = out_channels  # Update input channels for the next layer

        # Output layer
        self.output_conv = nn.Conv2d(
            in_channels=prev_channels,
            out_channels=1,
            kernel_size=2, 
            stride=2, 
            padding=0,
        )
        
        self.sigmoid = nn.Sigmoid()

        self.main = nn.Sequential(*layers)

    def forward(self, input, label):
        label_embedding = self.label_embedding(label).view(label.size(0), -1, 1, 1)
        label_embedding = label_embedding.expand(-1, -1, input.size(2), input.size(3))  # Expand along height and width dimensions
        combined_input = torch.cat((input, label_embedding), 1)
        # print('before main', combined_input.shape)
        x = self.main(combined_input)
        # print('after main', x.shape)
        x = self.output_conv(x)
        x = x.flatten()
        # print('after output conv', x.shape)
        x = self.sigmoid(x)
        # print('after sigmoid', x.shape)
        return x
