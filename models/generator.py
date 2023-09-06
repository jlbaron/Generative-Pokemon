'''
generator model for the GAN
will use convolution layers to upsample from seed
'''
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # layers of convtranspose2d, batchnorm2d, relu
        )
    def forward(self, x):
        return self.main(x)
