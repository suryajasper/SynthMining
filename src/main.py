from src.Discriminator import Discriminator
from src.Generator import Generator
import torch.utils.data as tdata
import torchvision
from src.misc import *
from matplotlib import pyplot as plt
import numpy as np


# setup machine
processor = "cpu"
if torch.cuda.is_available():
    # use NVIDIA. CPU if not detected
    print("Cuda detected; using GPU.")
    processor = "cuda"
device = torch.device(processor)


# MODEL INITIALIZTION
# Number of channels in the training images. COLOR CHANNELS
channels = 3  # (nc) <MAKE CONSTANT>
# Size of z latent vector (i.e. size of generator input)
noise_size = 156  # (nz) <RANDOM 1D TENSOR>
# Size of training images,
image_size = 64
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64

generator = Generator(noise_size, ngf, channels).to(device)
discriminator = Discriminator(noise_size, ndf, channels).to(device)
generator.apply(weights_init)
discriminator.apply(weights_init)

# debug
# print(generator)
# print(discriminator)

# load data
# label = 0
transform_fn = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # convert to 3 layer for grayscale or b&w img
    transforms.Resize(image_size),  # resize to 64x64
    transforms.CenterCrop(image_size),  # crop img at center. Not sure why needed
    transforms.ToTensor(),  # convert PIL to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# load mnist data, specify which label to use then subset
# data_mnist = torchvision.datasets.MNIST(root="./nosync_data/mnist/", train=True, download=True, transform=transform_fn)
data_mnist = torchvision.datasets.FashionMNIST(root="./nosync_data/mnist/", train=True, download=True, transform=transform_fn)
# indices = torch.where(data_mnist.targets == label)[0]
# train_data_mnist = torch.utils.data.Subset(data_mnist, indices)
train_data_mnist = data_mnist

# ** LOSS AND OPTIMIZATION **
# Initialize BCELoss function
criterion = torch.nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, noise_size, 1, 1, device=device)

# hyperparameters
# Learning rate for optimizers
lr = 0.0002
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Setup Adam optimizers for both G and D
optimizerG = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

### ** TRAINING FRAMEWORK **
# real and fake label values
real_label = 1.
fake_label = 0.

# Number of training epochs
num_epochs = 20

## ** STREAMING IN TRAINING DATA **
# G(z) <ON REAL>
batch_size = 128
train_size = len(train_data_mnist)//batch_size*batch_size  # make sure there's enough data in a batch

# Create the dataloader
rand_indices = np.random.choice(len(train_data_mnist), train_size, replace=False)
trunc_dataset = tdata.Subset(dataset=train_data_mnist, indices=rand_indices)
dataloader = tdata.DataLoader(dataset=trunc_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

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
    for i, data in enumerate(dataloader):  # image_convert is the list of images as 1D tensors
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################

        ## TRAIN D INDEPENDANTLY (all real batch)
        discriminator.zero_grad()  # reset gradients
        # format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        # target tensor
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D

        output = discriminator(real_cpu).view(-1)
        # print('Discriminator from real target: ', label)
        # print('Discriminator from real test: ', output)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        # average of each batch
        D_x = output.mean().item()

        ## TRAIN G INDEPENANTLY (all-fake batch)
        # Generate batch of latent vectors
        noise = torch.randn(batch_size, noise_size, 1, 1, device=device)  # random 1D tensor
        # Generate fake image batch with G
        fake = generator(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = discriminator(fake.detach())
        # print('Generator from fake target: ', label)
        # print('Generator from fake test: ', output)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output.squeeze(1).squeeze(1).squeeze(1), label)
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
        generator.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = discriminator(fake).view(-1)
        # print('Real Values: ', label)
        # print('Generator from fake test: ', output)
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
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 100 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
            with torch.no_grad():
                comp_buffer = generator(fixed_noise).detach()
                fake = comp_buffer.cuda()  # if torch.cuda.is_available() else comp_buffer.cpu()

            gen_img = torchvision.utils.make_grid(fake, padding=2, normalize=True)
            torchvision.utils.save_image(gen_img, './nosync_data/generated/clothes/e{}_i{}.png'.format(epoch, iter))
            img_list.append(torchvision.utils.make_grid(fake, padding=2, normalize=True))

        iters += 1

plt.plot(G_losses, label="generator loss")
plt.plot(D_losses, label="discriminator loss")
plt.legend()
plt.show()
