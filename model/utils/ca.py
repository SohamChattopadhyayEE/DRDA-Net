import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init

## Channel attention ###
class CALayer(nn.Module):
    def __init__(self, channels, reduction=16):
      super(CALayer, self).__init__()
      self.GlobalAvgPool = nn.AdaptiveAvgPool2d((1,1))
      self.conv_du = nn.Sequential(
          nn.Conv2d(channels, channels//reduction, 1, padding = 0, bias = True),
          nn.ReLU(),
          nn.Conv2d(channels//reduction, channels, 1, padding = 0, bias = True),
          nn.Sigmoid()
      )

    def forward(self, x):
      y = self.GlobalAvgPool(x)
      y = self.conv_du(y)

      return x*y