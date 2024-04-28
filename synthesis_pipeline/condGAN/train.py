import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as tdata
import torchvision.transforms as transforms
import torchvision.utils as vutils

from DiffAugment_pytorch import DiffAugment
from dataset import MultiClassDataset
from cond_DCGAN_network import Generator, Discriminator
from utils import torch_utils

# parse arguments
parser = argparse.ArgumentParser(
    prog='Conditional GAN Training Pipeline',
    description='Low-shot Training pipeline for Conditional DCGAN implementation using differentiable augmentation',
)
parser.add_argument('--data', required=True, type=str)
parser.add_argument('--image_size', required=True, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--noise_size', default=154, type=int)
parser.add_argument('-d', '--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
parser.add_argument('--no_diffaug', action='store_true')
parser.add_argument('--fixed_seed', action='store_true')
args = parser.parse_args()

if args.fixed_seed:
    print('Using Fixed Seed')
    torch.manual_seed(69)
else:
    manualSeed = random.randint(1, 10000)
    print("Using Random Seed: ", manualSeed)
    random.seed(manualSeed)

# set device for training
device = torch.device(args.device)

policy = 'color,translation,cutout'

### ** TRAINING FRAMEWORK ** 
# real and fake label values
real_label = 1.
fake_label = 0.

# Number of training epochs
num_epochs = 1000

batch_size = args.batch_size

dataset = MultiClassDataset(data_path=args.data,
                            category_max=batch_size,
                            transform=transforms.Compose([
                                transforms.Resize(args.image_size),
                                transforms.CenterCrop(args.image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))

dataloader = tdata.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

loss_fn = nn.BCELoss()

lr = 0.0002
beta1 = 0.5

# initialize networks
netG = Generator(n_categories=dataset.num_labels, noise_size=args.noise_size, img_size=args.image_size).to(device)
netD = Discriminator(n_categories=dataset.num_labels, img_size=args.image_size).to(device)

print(netG)
print(netD)

# randomly initialize network weights
netG.apply(torch_utils.weights_init)
netD.apply(torch_utils.weights_init)

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

fixed_noise = torch.randn(batch_size, args.noise_size, 1, 1, device=device)
fixed_labels = torch.randint(0, dataset.num_labels, (batch_size,), device=device)

img_list = [] 
G_losses = [] 
D_losses = []
iters = 0

print("Starting Training Loop...")
for epoch in range(num_epochs):
    for i, (img_batch, label_batch) in enumerate(dataloader):        
        netD.zero_grad()
        
        b_size = img_batch.size(0)
        real_batch = img_batch.to(device)
        labels = label_batch.to(device)

        # Real batch
        expected = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        output = netD(DiffAugment(real_batch, policy=policy), labels).flatten()
        errD_real = loss_fn(output, expected)
        errD_real.backward()
        D_x = output.mean().item()

        # Fake batch
        noise = torch.randn(b_size, args.noise_size, 1, 1, device=device)
        fake_batch = netG(noise, labels)
        
        expected.fill_(fake_label)
        output = netD(DiffAugment(fake_batch.detach(), policy=policy), labels).flatten()
        errD_fake = loss_fn(output, expected)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        
        errD = errD_real + errD_fake
        optimizerD.step()

        # Update G network
        netG.zero_grad()
        expected.fill_(real_label)  # Use real labels for generator loss
        output = netD(DiffAugment(fake_batch, policy=policy), labels).flatten()
        errG = loss_fn(output, expected)
        errG.backward()
        D_G_z2 = output.mean().item()
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
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                comp_buffer = netG(fixed_noise, fixed_labels).detach()
                fake_batch = comp_buffer.cuda()  # if torch.cuda.is_available() else comp_buffer.cpu()
            
            gen_img = vutils.make_grid(fake_batch, padding=2, normalize=True)
            vutils.save_image(gen_img, f'./out/run4/e{epoch}_i{iters}.png')
            img_list.append(gen_img)

        iters += 1
