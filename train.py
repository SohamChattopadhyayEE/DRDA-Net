import os
import matplotlib.pyplot as plt
import numpy as np
import warnings
import argparse

import torch
import torchvision
import torchvision.transforms as t
import torch.nn as nn
import torchvision.models as model

from model.drdanet import DRDANet

warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description='DRDA-Net')
# Paths
parser.add_argument('-t_path','--train_path', type=str, default='/dataset/Train',
                    help="path to the files of training images")
parser.add_argument('-v_path','--validation_path', type=str, default='/dataset/Validation',
                    help="path to the files of validation images")
parser.add_argument('-p_path','--plot_path', type=str, default='/dataset',
                    help="path to the convergence plots")
parser.add_argument('-m_path','--model_path', type=str, default='/dataset',
                    help="path to the model.pt")

# Model parameters
parser.add_argument('-bs','--bs', type=int, default=30,
                    help="batch size")
parser.add_argument('-n','--n_class', type=int, default=2,
                    help="number of classes")
parser.add_argument('-lr','--lr', type=float, default=0.0001,
                    help="number of classes")
parser.add_argument('-e','--epoch', type=int, default=300,
                    help="number of epochs")
parser.add_argument('-n_e_block','--num_elementary_blocks', type=int, default=7,
                    help="number of elementary blocks")
args = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_path=args.train_path
val_path=args.validation_path
plot_path=args.plot_path
snapshot_path=args.model_path


model_name='DRDANet'
batch_s = args.bs # MaxTill 84.87-->25    MaxWith 85.714--->50,  MaxWith 86.55--->50,  MaxWith 89.915--->30
normalize = t.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.529, 0.524, 0.525])

transform=t.Compose([t.Resize((224,224)), #MaxTill 84.87  MaxWith 85.714,   MaxWith:  89.915
                     t.RandomHorizontalFlip(), #MaxTill 84.87    MaxWith 85.714,   MaxWith:  89.915
                     t.RandomVerticalFlip(), #MaxTill 84.87    MaxWith 85.714,   MaxWith:  89.915
                     t.RandomRotation(0,10), #MaxWith 85.714,  MaxWith:  89.915
                     t.GaussianBlur(kernel_size = 3, sigma=(0.1, 2.0)), # MaxWith:  89.915
                     t.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=2, fill=0),
                     t.RandomAffine(degrees=(-10,10), translate=(0.1,0.1), 
                                    scale=(0.9,1.1), shear=(-5,5)), #MaxWith 85.714, MaxWith:  89.915
                     t.ToTensor()
                     #normalize
                     ])
dset_train=torchvision.datasets.ImageFolder(root=train_path,transform=transform)

test_trans=t.Compose([t.Resize((224,224)),t.ToTensor()])
dset_val=torchvision.datasets.ImageFolder(root=val_path,transform=test_trans)

train_loader=torch.utils.data.DataLoader(dset_train,batch_size=batch_s,shuffle=True,num_workers=16)
val_loader=torch.utils.data.DataLoader(dset_val,batch_size=batch_s,shuffle=False,num_workers=16)

num_classes = args.n_class
lr = args.lr
num_epoch = args.epoch
num_elementary_blocks = args.num_elementary_blocks # MaxWith: 85.714--->4,  MaxWith: 86.55--->7,  MaxWith: 86.55--->9, MaxWith:  89.915--->7

net = DRDANet(num_blocks = num_elementary_blocks, num_classes = num_classes)
net = net.to(device)#cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr = lr)

for epoch in range(num_epoch):
  print('Epoch: ', epoch+1)
  train_loss = 0.0
  correct = total = 0
  for i, data in enumerate(train_loader):
    net.train()
    optimizer.zero_grad()
    images, labels = data
    images, labels = images.to(device), labels.to(device)
    outputs = net(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    train_loss += loss.item()*images.size(0)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
  print('Train loss: ', train_loss/len(dset_train))
  print('Train accuracy: ', 100*correct/total)

  net.eval()
  with torch.no_grad():
    val_loss = 0.0
    correct = total = 0
    for val_data in val_loader:
      val_images, val_labels = val_data
      val_images, val_labels = val_images.to(device), val_labels.to(device)
      val_output = net(val_images)
      loss = criterion(val_output, val_labels)
      val_loss += loss.item()*val_images.size(0)
      _, predicted = torch.max(val_output.data, 1)
      total += val_labels.size(0)
      correct += (predicted == val_labels).sum().item()
    print('Validation loss: ', val_loss/len(dset_val))
    print('Validation accuracy: ', 100*correct/total)