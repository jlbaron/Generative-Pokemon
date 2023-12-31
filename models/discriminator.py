'''
discriminator to judge generator outputs
will downsample image
'''
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, nc, ndf, device="cpu"):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # layers of conv, batchnorm, and leakyrelu
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False, device=device),
            nn.LeakyReLU(0.2),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False, device=device),
            nn.BatchNorm2d(ndf * 2, device=device),
            nn.LeakyReLU(0.2),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False, device=device),
            nn.BatchNorm2d(ndf * 4, device=device),
            nn.LeakyReLU(0.2),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False, device=device),
            nn.BatchNorm2d(ndf * 8, device=device),
            nn.LeakyReLU(0.2),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False, device=device),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.main(x)
        