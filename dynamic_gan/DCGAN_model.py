from __future__ import print_function

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data 
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np

import misc_functions as mf

# ** quality of life imports ** 
# import matplotlib.pyplot as plt
# import matplotlib.animation as animationd

# ** RNG ** 
# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


# ** MODEL INITIALIZTION **
# Number of channels in the training images. COLOR CHANNELS
channels = 3 # (nc) <MAKE CONSTANT>

# Size of z latent vector (i.e. size of generator input)
noise_size = 156 # (nz) <RANDOM 1D TENSOR>

# Size of training images, feature maps in generator and discriminator respectively (LEN^2)
image_size = 64 # (ngf, ndf, image_size) <CHANGE TO LEN FROM ADARSH>

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# ** WEIGHT INITIALIZATION **
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ** GENERATOR MODEL (takes noise; returns tensor) ** 
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution 
            nn.ConvTranspose2d(
                in_channels=noise_size, 
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
                padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (gen_dimesions*4) x 8 x 8
            nn.ConvTranspose2d(
                in_channels=ngf * 4, 
                out_channels=ngf * 2,
                kernel_size=4, 
                stride=2, 
                padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (gen_dimesions*2) x 16 x 16
            nn.ConvTranspose2d(
                in_channels=ngf * 2, 
                out_channels=ngf, 
                kernel_size=4,
                stride= 2, 
                padding=1, 
                bias=False),
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
    
    # forward pass 
    def forward(self, input):
        return self.main(input)


# ** DISCRIMINATOR MODEL (takes tensor; returns probability) **
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(
                in_channels=channels, 
                out_channels=ndf, 
                kernel_size=4, 
                stride=2, 
                padding=1, 
                bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndvf) x 32 x 32
            nn.Conv2d(
                in_channels=ndf, 
                out_channels=ndf * 2, 
                kernel_size=4, 
                stride=2, 
                padding=1, 
                bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(
                in_channels=ndf * 2, 
                out_channels=ndf * 4, 
                kernel_size=4, 
                stride=2, 
                padding=1, 
                bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(
                in_channels=ndf * 4, 
                out_channels=ndf * 8, 
                kernel_size=4, 
                stride=2, 
                padding=1, 
                bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(
                in_channels=ndf * 8, 
                out_channels=1, 
                kernel_size=4, 
                stride=1, 
                padding=0, 
                bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# Device Specification
device = torch.device('cpu')

# ** INSTANTIATING G() AND D()
# Create models
netG = Generator().to(device)
netD = Discriminator().to(device)

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.02.
netG.apply(weights_init)
netD.apply(weights_init)

# Print the model
print(netG)
print(netD)

# ** LOSS AND OPTIMIZATION ** 

# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, noise_size, 1, 1, device=device)

# hyperparameters 
# Learning rate for optimizers
lr = 0.0002
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

### ** TRAINING FRAMEWORK ** 
# real and fake label values
real_label = 1.
fake_label = 0.

# Number of training epochs
num_epochs = 40

## ** STREAMING IN TRAINING DATA ** 
# G(z) <ON REAL>
file_list = "./trainer_set.txt"
test_size = 7019
real_set = mf.to_tensor(file_list, image_size=image_size, test_size=test_size)

# convert real_set into batches
batch_size = 20
batchs = len(real_set) // batch_size
img_batches = mf.to_batches(real_set, batch_size)


## ** RECORD KEEPING ** 
img_list = [] 
G_losses = [] 
D_losses = []
iters = 0

# ** TRAINING LOOP ** 
print("Starting Training Loop...")
# For each epoch (training cycle)
for epoch in range(num_epochs):
    # For each batch in the img_batches
    for i, batch in enumerate(img_batches): # image_convert is the list of images as 1D tensors 
        
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z))) 
        ###########################

        ## TRAIN D INDEPENDANTLY (all real batch)
        netD.zero_grad() # reset gradients 
        # format batch
        real_cpu = batch.to(device)
        
        b_size = real_cpu.size(0)
        
        # target tensor
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device) 
        # Forward pass real batch through D
        
        output = netD(real_cpu).view(-1)
        print('Discriminator from real target: ', label)
        print('Discriminator from real test: ', output)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        # average of each batch
        D_x = output.mean().item()

        ## TRAIN G INDEPENANTLY (all-fake batch)
        # Generate batch of latent vectors
        noise = torch.randn(batch_size, noise_size, 1, 1, device=device) # random 1D tensor
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach())
        print('Generator from fake target: ', label)
        print('Generator from fake test: ', output)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output.squeeze(1).squeeze(1).squeeze(1), label )
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################

        ## REINFORCE D; TRAIN D AGAINST G (all real into all fake; maximize error)
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        print('Real Values: ', label)
        print('Generator from fake test: ', output)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 10 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch+8, num_epochs, i, len(img_batches),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(img_batches)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1




