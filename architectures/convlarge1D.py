#!coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class GaussianNoise(nn.Module):
    
    def __init__(self, std):
        super(GaussianNoise, self).__init__()
        self.std = std

    def forward(self, x):
        zeros = torch.zeros(x.size())
        n = Variable(torch.normal(zeros_, std=self.std))        
        return x + n

class CNN_block(nn.Module):

    def __init__(self, in_plane, out_plane, kernel_size, padding, activation):
        super(CNN_block, self).__init__()

        self.act = activation
        self.conv = nn.Conv1d(in_plane, 
                              out_plane, 
                              kernel_size, 
                              padding=padding)

        self.bn = nn.BatchNorm1d(out_plane)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
        
class ChannelAttention1(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention1, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool1d(1)

        self.fc1   = torch.nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = torch.nn.ReLU()
        self.fc2   = torch.nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)   
    

class ChannelAttention2(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention2, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool1d(1)

        self.fc1   = torch.nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = torch.nn.ReLU()
        self.fc2   = torch.nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)    

class CNN(nn.Module):
                      
    def __init__(self, block, num_blocks, num_classes=12, drop_ratio=0.0, feanum=6):
        super(CNN, self).__init__()

        self.in_plane = feanum
        self.out_plane = 128
        self.gn = GaussianNoise(0.15)
        self.act = nn.LeakyReLU(0.1)
        
        #self.ca1 = ChannelAttention1(self.in_plane)                                      
        self.layer1 = self._make_layer(block, num_blocks[0], 128, 3, padding=1)
        self.mp1 = nn.MaxPool1d(4, stride=4, padding=0)
        self.drop1 = nn.Dropout(drop_ratio)
        
        #self.ca2 = ChannelAttention2(128)      
        self.layer2 = self._make_layer(block, num_blocks[1], 256, 3, padding=1)
        self.mp2 = nn.MaxPool1d(4, stride=4, padding=0)
        self.drop2 = nn.Dropout(drop_ratio)
        
        #self.ca3 = ChannelAttention2(256)      
        self.layer3 = self._make_layer(block, num_blocks[2], 
                                       [512, 256, self.out_plane], 
                                       [3, 1, 1], 
                                       padding=0)
        self.ap3 = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(self.out_plane, num_classes)
                          
    def _make_layer(self, block, num_blocks, planes, kernel_size, padding=1):
        if isinstance(planes, int):
            planes = [planes]*num_blocks
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size]*num_blocks
        layers = []
        for plane, ks in zip(planes, kernel_size):
            layers.append(block(self.in_plane, plane, ks, padding, self.act))
            self.in_plane = plane
        return nn.Sequential(*layers)

    def forward(self, x):
        #out = self.ca1(x)*x       # New added        
        #out = self.layer1(out)     
        out = self.layer1(x)     
        out = self.mp1(out)
        out = self.drop1(out)
        #out = self.ca2(out)*out   # New added
        out = self.layer2(out)
        out = self.mp2(out)
        out = self.drop1(out)
        #out = self.ca3(out)*out   # New added
        out = self.layer3(out)
        out = self.ap3(out)

        out = out.view(out.size(0), -1)
        return self.fc1(out)

def convLarge1D(num_classes, drop_ratio=0.0, feanum=6):
    return CNN(CNN_block, [3,3,3], num_classes, drop_ratio, feanum)

#def test():
#    print('--- run conv_large test ---')
#    x = torch.randn(2,3,32,32)
#    for net in [convLarge1D(10)]:
#        print(net)
#        y = net(x)
#        print(y.size())
