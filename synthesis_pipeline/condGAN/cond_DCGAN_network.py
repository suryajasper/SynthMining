import torch
import torch.nn as nn
import torch.nn.parallel

# ** GENERATOR MODEL (takes noise; returns tensor) ** 
class Generator(nn.Module):
    def __init__(self,
                 n_categories,
                 noise_size,
                 channels=3,
                 feature_map_size=64,
                 embedding_size=50):

        super(Generator, self).__init__()
        
        ngf = feature_map_size
        
        self.label_embedding = nn.Embedding(n_categories, embedding_size)
        
        self.main = nn.Sequential(
            # input is Z, going into a convolution 
            nn.ConvTranspose2d(
                in_channels=embedding_size + noise_size, 
                out_channels=ngf * 8, 
                kernel_size=4, 
                stride=1, 
                padding=0, 
                bias=False
            ),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (gen_dimesions*8) x 4 x 4
            nn.ConvTranspose2d(
                in_channels=ngf * 8, 
                out_channels=ngf * 4,
                kernel_size=4, 
                stride=2, 
                padding=1, bias=False
            ),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (gen_dimesions*4) x 8 x 8
            nn.ConvTranspose2d(
                in_channels=ngf * 4, 
                out_channels=ngf * 2,
                kernel_size=4, 
                stride=2, 
                padding=1, bias=False
            ),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (gen_dimesions*2) x 16 x 16
            nn.ConvTranspose2d(
                in_channels=ngf * 2, 
                out_channels=ngf, 
                kernel_size=4,
                stride= 2, 
                padding=1, 
                bias=False
            ),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (gen_dimesions) x 32 x 32
            nn.ConvTranspose2d(
                in_channels=ngf, 
                out_channels=channels, 
                kernel_size=4, 
                stride=2, 
                padding=1, 
                bias=False
            ),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, noise, label):
        label_embedding = self.label_embedding(label).view(label.size(0), -1, 1, 1)
        combined_input = torch.cat((noise, label_embedding), 1)
        output = self.main(combined_input)
        return output


# ** DISCRIMINATOR MODEL (takes tensor; returns probability) **
class Discriminator(nn.Module):
    def __init__(self,
                 n_categories,
                 channels=3,
                 feature_map_size=64,
                 embedding_size=50):
        super(Discriminator, self).__init__()
        
        ndf = feature_map_size
        
        self.label_embedding = nn.Embedding(n_categories, embedding_size)
        
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(
                in_channels=channels + embedding_size, 
                out_channels=ndf, 
                kernel_size=4, 
                stride=2, 
                padding=1, 
                bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndvf) x 32 x 32
            nn.Conv2d(
                in_channels=ndf, 
                out_channels=ndf * 2, 
                kernel_size=4, 
                stride=2, 
                padding=1, 
                bias=False
            ),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(
                in_channels=ndf * 2, 
                out_channels=ndf * 4, 
                kernel_size=4, 
                stride=2, 
                padding=1, 
                bias=False
            ),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(
                in_channels=ndf * 4, 
                out_channels=ndf * 8, 
                kernel_size=4, 
                stride=2, 
                padding=1, 
                bias=False
            ),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(
                in_channels=ndf * 8, 
                out_channels=1, 
                kernel_size=4, 
                stride=1, 
                padding=0, 
                bias=False
            ),
            nn.Sigmoid()
        )

    def forward(self, input, label):
        label_embedding = self.label_embedding(label).view(label.size(0), -1, 1, 1)
        expanded_labels = label_embedding.expand(label.size(0), label_embedding.size(1), input.size(2), input.size(3))
        combined_input = torch.cat((input, expanded_labels), 1)
        return self.main(combined_input)

