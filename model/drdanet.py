import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init

from model.utils.rdab import RDAB

## Dense-Residual Dual-shuffle Attention Net (DRDANet) ##

class DRDANet(nn.Module):
    def __init__(self, num_blocks = 4, num_classes = 2, num_channels = 3):
      super(DRDANet, self).__init__()

      self.conv_in = nn.Sequential(nn.Conv2d(num_channels, 32, kernel_size = 3, stride = 1, padding = 1),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU())
      self.maxpool = nn.MaxPool2d((2,2))

      self.dsrcab_1 = RDAB(n_blocks = num_blocks, n_features = 32, reductions = 16 // 2)

      self.dsrcab_2 = RDAB(n_blocks = num_blocks, n_features = 64, reductions = 16)
      self.conv2 = nn.Sequential(nn.Conv2d(160, 128, 3, stride = 1, padding = 1),
                                 nn.BatchNorm2d(128),
                                 nn.ReLU())

      self.dsrcab_3 = RDAB(n_blocks = num_blocks, n_features = 128, reductions = 16)
      self.conv3 = nn.Sequential(nn.Conv2d(352, 256, 3, stride = 1, padding = 1),
                                 nn.BatchNorm2d(256),
                                 nn.ReLU())
      
      self.dsrcab_4 = RDAB(n_blocks = num_blocks, n_features = 256, reductions = 32)
      self.conv4 = nn.Sequential(nn.Conv2d(736, 512, 3, stride = 1, padding = 1),
                                 nn.BatchNorm2d(512),
                                 nn.ReLU())
      
      self.conv_f = nn.Sequential(nn.Conv2d(512, 256, 2 , stride = 2, padding = 0),
                                 nn.BatchNorm2d(256),
                                 nn.ReLU()) 
      self.fc1 = nn.Linear(256*7*7, 2048)
      self.fc2 = nn.Linear(512, num_classes)

      self.avgpool = nn.AdaptiveAvgPool2d((1,1))

      self.clf = nn.Sigmoid()



    def forward(self, x):
      x = self.conv_in(x) # N x 32 x 224 x 224
      x1 = self.maxpool(x) # N x 32 x 112 x 112

      x2 = self.dsrcab_1(x1) # N x 32 x 112 x 112
      x2 = torch.cat((x1, x2), 1) # N x 64 x 112 x 112
      x2 = self.maxpool(x2) # N x 64 x 56 x 56

      x3 = self.dsrcab_2(x2) # N x 64 x 56 x 56
      x3 = torch.cat((x3, x2), 1) # N x 128 x 56 x 56
      x1_1 = self.maxpool(x1) # N x 32 x 56 x 56
      x3 = torch.cat((x3, x1_1), 1) # N x (128+32) = 160 x 56 x 56
      x3 = self.conv2(x3) # N x 128 x 56 x 56
      x3 = self.maxpool(x3) # N x 128 x 28 x 28

      x4 = self.dsrcab_3(x3) # N x 128 x 28 x 28
      x4 = torch.cat((x4, x3), 1) # N x 256 x 28 x 28
      x2_1 = self.maxpool(x2) # N x 64 x 28 x 28
      x4 = torch.cat((x4, x2_1), 1) # N x (256+64) = 320 x 28 x 28
      x1_2 = self.maxpool(x1_1) # N x 32 x 28 x 28
      x4 = torch.cat((x4, x1_2), 1) # N x (320+32) = 352 x 28 x 28
      x4 = self.conv3(x4) # N x 256 x 28 x 28
      x4 = self.maxpool(x4) # N x 256 x 14 x 14

      x5 = self.dsrcab_4(x4) # N x 256 x 14 x 14
      x5 = torch.cat((x5, x4), 1) # N x 512 x 14 x 14
      x3_1 = self.maxpool(x3) # N x 128 x 14 x 14
      x5 = torch.cat((x5, x3_1), 1) # N x (512+128) = 640 x 14 x 14
      x2_2 = self.maxpool(x2_1) # N x 64 x 14 x 14
      x5 = torch.cat((x5, x2_2), 1) # N x (640+64) = 704 x 14 x 14
      x1_3 = self.maxpool(x1_2) # N x 32 x 14 x 14
      x5 = torch.cat((x5, x1_3), 1) # N x (704+32) = 736 x 14 x 14
      x5 = self.conv4(x5) # N x 512 x 14 x 14

      out = self.maxpool(x5) # N x 512 x 7 x 7
      out = self.avgpool(out) # N x 512 x 1 x 1
      #out = self.conv_f(x5) # N x 256 x 7 x 7
      out = torch.flatten(out, 1) # N x 512
      #out = torch.flatten(out, 1) # N x (256 x 7 x 7)
      #out = self.fc1(out)
      out = self.fc2(out)
      out = self.clf(out)

      return out