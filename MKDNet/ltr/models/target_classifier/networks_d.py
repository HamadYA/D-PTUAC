
import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
from torch.nn.utils import spectral_norm
import pdb
# from util.util import SwitchNorm2d
import torch.nn.functional as F

class Discriminator1(nn.Module):
    def __init__(self):
        super(Discriminator1, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(256*25, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity
class Discriminator2(nn.Module):
    def __init__(self):
        super(Discriminator2, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(256*9, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity