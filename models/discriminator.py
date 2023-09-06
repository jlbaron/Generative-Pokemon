'''
discriminator to judge generator outputs
will downsample image
'''
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # layers of conv, batchnorm, and leakyrelu
        )
    def forward(self, x):
        return self.main(x)
        