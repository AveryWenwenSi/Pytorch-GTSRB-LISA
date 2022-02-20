import os
import pdb
import re
import numpy as np
from pprint import pprint
import tensorflow as tf

import torch
import torch.nn as nn
import torch.nn.functional as F
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

class TorchYadav(nn.Module):
    def __init__(self):
        super().__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        # torch.nn.Dropout(p=0.5, inplace=False)
        self.conv1 = nn.Conv2d(3, 3, 1)
        self.conv2 = nn.Conv2d(3, 32, 5, padding=2)
        self.conv3 = nn.Conv2d(32, 32, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout()
        self.conv4 = nn.Conv2d(32, 64, 5, padding=2)
        self.conv5 = nn.Conv2d(64, 64, 5, padding=2)
        # self.dropout5 = nn.Dropout()
        self.conv6 = nn.Conv2d(64, 128, 5, padding=2)
        self.conv7 = nn.Conv2d(128, 128, 5, padding=2)
        # self.dropout7 = nn.Dropout()
        self.fc1 = nn.Linear(14336, 1024)
        # self.fc_dropout1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        # self.fc_dropout2 = nn.Dropout()
        self.fc3 = nn.Linear(1024, 43)
        self.pred = nn.Softmax()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.pool(self.conv3(x)))
        # x = self.dropout3(x)
        x1 = F.relu(self.conv4(x))
        x1 = F.relu(self.pool(self.conv5(x1)))
        # x1 = self.dropout5(x1)
        x2 = F.relu(self.conv6(x1))
        x2 = F.relu(self.pool(self.conv7(x2))) # "conv6"
        # x2 = self.dropout7(x2)
        x = x.permute(0, 2, 3, 1)
        x11 = torch.flatten(x, 1)
        x1 = x1.permute(0, 2, 3, 1)
        x22 = torch.flatten(x1, 1)
        x2 = x2.permute(0, 2, 3, 1)
        x33 = torch.flatten(x2, 1)
        flat = torch.cat((x11, x22, x33),dim=1)
        flat = F.relu(self.fc1(flat))
        flat = F.relu(self.fc2(flat))
        flat1 = self.fc3(flat)
        pred = self.pred(flat1)
        return pred


import cv2

def preprocess_yadav(image):
    image = image.astype(np.uint8)
    image = image/255. - 0.5
    return image.astype(np.float32)

def read_img(path):
    '''
    Reads the image at path, checking if it was really loaded
    '''
    img = cv2.imread(path)
    assert img is not None, "No image found at %s"%path
    return img

srcimgs = '/home/wenwens/DeepDetector/GtsrbCnn/StopSigns/CleanStop'
imgnames = []
for x in os.listdir(srcimgs):
    if x.lower().endswith(".jpg") or x.lower().endswith(".png"):
        imgnames.append(x)

Torch_model = TorchYadav()
Torch_model.load_state_dict(torch.load("TorchYadav.pth"))
Torch_model.eval()

cnt, tcorrect = 0, 0
for image in imgnames:
    inputs = read_img(os.path.join(srcimgs, image))
    inputs = preprocess_yadav(inputs)
    
    torch_inputs = torch.Tensor(inputs).permute(2, 0, 1)
    inputs = np.expand_dims(inputs, 0)
    torch_inputs = torch.unsqueeze(torch_inputs, 0)
    cnt += 1

    torch_model_out = Torch_model(torch_inputs)
    torch_c = torch.argmax(torch_model_out, 1).item()
    if torch_c == 14:
        tcorrect += 1

print ("pytorch performance: ", tcorrect / cnt, tcorrect)

