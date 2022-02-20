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
# os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3"

from scipy.misc import imread

def torch_flatten(x):
    ls = x.shape
    num_features = ls[1]*ls[2]*ls[3]
    x = torch.reshape(x, (-1, num_features))
    return x

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 8, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 128, 6, stride=2, padding=0)
        self.conv3 = nn.Conv2d(128, 128, 5, stride=1, padding=0)
        self.dense = nn.Linear(512, 18)
        self.pred = nn.Softmax()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        #print(x.shape)
        x = F.relu(self.conv2(x))
        #print(x.shape)
        x = F.relu(self.conv3(x))
        #print(x.shape)
        x = x.permute(0, 2, 3, 1)
        x = torch.flatten(x, 1)
        flat = F.relu(self.dense(x))
        pred = self.pred(flat)
        return pred

def load_norm_img_from_source(src):
    img = imread(src, mode='RGB')
    assert img is not None, "No image found at filepath %s"%src
    return img/255.0

srcimgs = '/home/wenwens/DeepDetector/LisaCnn/StopSigns/CleanStop/'
imgnames = []
for x in os.listdir(srcimgs):
    if x.lower().endswith(".jpg") or x.lower().endswith(".png"):
        imgnames.append(x)


Torch_model = Net()
Torch_model.load_state_dict(torch.load("lisa_model.pth"))
Torch_model.eval()
cnt, tcorrect = 0, 0

for image in imgnames:
    inputs = load_norm_img_from_source(os.path.join(srcimgs, image))
    torch_inputs = torch.Tensor(inputs).permute(2, 0, 1)
    torch_inputs = torch.unsqueeze(torch_inputs, 0)
    cnt += 1

    torch_model_out = Torch_model(torch_inputs)
    torch_c = torch.argmax(torch_model_out, 1).item()
    if torch_c == 14:
        tcorrect += 1

print ("pytorch performance: ", tcorrect / cnt, tcorrect)


