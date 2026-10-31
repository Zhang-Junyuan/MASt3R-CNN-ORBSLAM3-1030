"""Convolution over organized MASt3R point maps, not an unordered XYZ array.
Channels: RGB(3), normalized XYZ(3), confidence(1), rigid reprojection
residual(1), pair point consistency(1), correspondence support(1).
The bootstrap checkpoint is synthetic-trained and is not a validated detector.
"""
import torch
from torch import nn
class Block(nn.Sequential):
    def __init__(self,cin,cout,d=1):
        super().__init__(nn.Conv2d(cin,cin,3,padding=d,dilation=d,groups=cin),nn.ReLU(),nn.Conv2d(cin,cout,1),nn.ReLU())
class DynamicPointCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(10,24,1),nn.ReLU(),Block(24,32),Block(32,32,2),Block(32,24,4),nn.Conv2d(24,1,1))
    def forward(self,x):return self.net(x)
