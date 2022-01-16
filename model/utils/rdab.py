import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init

from model.utils.ca import CALayer
from model.utils.drb import DRB

##  Residual Dual-shuffle Attention Block ####
class RDAB(nn.Module):
    def __init__(self, n_blocks, n_features, reductions = 16):
      super(RDAB, self).__init__()

      self.msb = DRB(n_features)
      self.ca = CALayer(n_features, reductions)

      m = []

      for i in range(n_blocks):
        m.append(DRB(n_features))
        if (i == n_blocks-1):
          m.append(nn.Sigmoid())

      self.body = nn.Sequential(*m)
      self.relu = nn.ReLU()

    def forward(self, x):
      x_msb = self.body(x)
      x_ca = self.ca(x)

      out = x_ca + x_msb

      return self.relu(out)