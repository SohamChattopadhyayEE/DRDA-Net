import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init

from model.utils.layers import Shuffle3x3, Shuffle5x5

# Dual-shuffle Residual Block (DRB)
class DRB(nn.Module):
    def __init__(self, n_features):
      super(DRB, self).__init__()

      self.features = n_features
      self.groups = 4 #self.features / 3

      self.shuffle3x3 = Shuffle3x3(in_channels = self.features, out_channels = self.features, groups = self.groups)
      self.shuffle5x5 = Shuffle5x5(in_channels = self.features, out_channels = self.features, groups = self.groups)

      self.outConv = nn.Conv2d(2*self.features, self.features, kernel_size = 3, stride = 1, padding = 1)


      self.relu6 = nn.ReLU6()

    def forward(self, x):
      x_shuff3x3 = self.shuffle3x3(x)
      x_shuff5x5 = self.shuffle5x5(x)
      out = torch.cat((x_shuff3x3, x_shuff5x5), 1)
      out = self.outConv(out)

      return self.relu6(out)