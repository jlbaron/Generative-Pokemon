'''
generator model for the GAN
will use convolution layers to upsample from seed
'''
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, nz, ngf, nc, device="cpu"):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False, device=device),
            nn.BatchNorm2d(ngf * 8, device=device),
            nn.ReLU(),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False, device=device),
            nn.BatchNorm2d(ngf * 4, device=device),
            nn.ReLU(),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False, device=device),
            nn.BatchNorm2d(ngf * 2, device=device),
            nn.ReLU(),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False, device=device),
            nn.BatchNorm2d(ngf, device=device),
            nn.ReLU(),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False, device=device),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )
    def forward(self, x):
        return self.main(x)
