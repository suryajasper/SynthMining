from torch import nn


class Discriminator(nn.Module):
    def __init__(self, noise, ndf, imgchannel):
        super().__init__()
        self.noise_size = noise
        self.ndf = ndf
        self.imgchannel = imgchannel

        # setup neuro_network architect, this will return a function
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(
                in_channels=self.imgchannel,
                out_channels=self.ndf,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndvf) x 32 x 32
            nn.Conv2d(
                in_channels=self.ndf,
                out_channels=self.ndf * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(
                in_channels=self.ndf * 2,
                out_channels=self.ndf * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(
                in_channels=self.ndf * 4,
                out_channels=self.ndf * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(
                in_channels=self.ndf * 8,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False),
            nn.Sigmoid()
        )

    def forward(self, indata):
        return self.main(indata)
