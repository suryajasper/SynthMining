from torch import nn


# ** GENERATOR MODEL (takes noise; returns tensor) **
class Generator(nn.Module):
    def __init__(self, noise, ngf, imgchannel):
        super().__init__()
        self.noise_size = noise
        self.ngf = ngf
        self.imgchannel = imgchannel

        # setup neuro_network architect, this will return a function
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(
                in_channels=self.noise_size,
                out_channels=self.ngf * 8,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (gen_dimesions*8) x 4 x 4
            nn.ConvTranspose2d(
                in_channels=self.ngf * 8,
                out_channels=self.ngf * 4,
                kernel_size=4,
                stride=2,
                padding=1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (gen_dimesions*4) x 8 x 8
            nn.ConvTranspose2d(
                in_channels=self.ngf * 4,
                out_channels=self.ngf * 2,
                kernel_size=4,
                stride=2,
                padding=1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (gen_dimesions*2) x 16 x 16
            nn.ConvTranspose2d(
                in_channels=self.ngf * 2,
                out_channels=self.ngf,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. (gen_dimesions) x 32 x 32
            nn.ConvTranspose2d(
                in_channels=self.ngf,
                out_channels=self.imgchannel,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.Tanh()  # hyperbolic tangent. Not sure what it does
            # state size. (nc) x 64 x 64
        )

    # forward pass
    def forward(self, indat):
        return self.main(indat)
