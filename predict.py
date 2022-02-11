import os
import matplotlib.pyplot as plt
import numpy as np
import warnings
import argparse
import cv2

import torch
import torchvision
import torchvision.transforms as t
import torch.nn as nn
import torchvision.models as model

from model.drdanet import DRDANet

warnings.filterwarnings('ignore')
class prediction():
    def __init__(self,model = DRDANet(7,2), wt = 'weights/model_DRDANet7.pt'):
        self.wt = wt 
        self.net = model
        load_model=self.wt
        optimizer = torch.optim.Adam(self.net.parameters(), lr = 0.0001)

        if os.path.exists(load_model):
            checkpoint=torch.load(load_model, map_location=torch.device('cpu'))
            self.net.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            print("model loaded successfully")
            print('starting training after epoch: ',checkpoint['epoch'])

    def predict(self, file):
        img = cv2.imread(file)
        img = cv2.resize(img, (224,224))
        img = torch.tensor(img)
        img = img.permute(2,0,1)
        img = img.unsqueeze(0)
        img = img/225
        net = self.net.cpu()
        output = net(img)
        _, predicted = torch.max(output.data, 1)
        pred = predicted.cpu().detach()
        pred = np.array(pred[0])
        if pred == 1 :
            return "Benign"
        else :
            return "Malignant"

if __name__ == "__main__":
    file = 'Sample_data/SOB_B_A-14-22549AB-400-005.png'
    model = DRDANet(7,2)
    pred = prediction(model)
    out = pred.predict(file)
    print(out)

