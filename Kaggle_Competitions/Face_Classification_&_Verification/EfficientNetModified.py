''' https://medium.com/analytics-vidhya/image-classification-with-efficientnet-better-performance-with-computational-efficiency-f480fdb00ac6
The article above also helped with the network construction'''
'''EfficientNet in PyTorch.
Paper: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks".
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MBConvBlock(nn.Module):
    '''expand + depthwise + pointwise + squeeze-excitation'''

    def __init__(self, input_channel, out_channel, expansion, stride):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        #Expansion Phase
        channel = expansion * input_channel
        self.conv1 = nn.Conv2d(input_channel, channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, stride=stride, padding=1, groups=channel, bias=False)
        self.bn2 = nn.BatchNorm2d(channel)
        self.conv3 = nn.Conv2d(channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel)

        if stride == 1 and input_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channel),
            )
        else:
            self.shortcut = nn.Sequential()
        # SE layers
        self.fc1 = nn.Conv2d(out_channel, out_channel//16, kernel_size=1)
        self.fc2 = nn.Conv2d(out_channel//16, out_channel, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        shortcut = out
        if self.stride == 1:
            shortcut = self.shortcut(x)
        # w = F.avg_pool2d(out, out.size(2))
        # w = F.relu(self.linear1(w))
        # # w = self.fc2(w).sigmoid()
        # w = self.swish_activation(self.linear2(w))
        # out = out * w + shortcut
        out = self.squeeze(out, shortcut)
        return out

    def swish_activation(self, x):
        return x.sigmoid()

    # Squeeze-Excitation
    def squeeze(self, x, shortcut):
        s = F.avg_pool2d(x, x.size(2))
        s = F.relu(self.fc1(s))
        s = self.swish_activation(self.fc2(s))
        out = x * s + shortcut
        return out


class EfficientNet(nn.Module):
    def __init__(self, expansion, filters, stride_array, num_blocks, num_classes=2300):
        super(EfficientNet, self).__init__()
        #self.cfg = cfg
        self.expansion = expansion
        self.filters = filters
        self.stride_array = stride_array
        self.num_blocks = num_blocks
        self.num_feats = 3
        self.input_channel = 32
        max_lin_size = 4096
        self.conv1 = nn.Conv2d(self.num_feats, self.input_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.input_channel)
        self.layers = self.pattern_maker(self.input_channel)
        self.linear = nn.Linear(self.filters[-1], max_lin_size)
        #self.linear = nn.Linear(self.filters[-1], num_classes)
        self.linear2 = nn.Linear(max_lin_size, max_lin_size)
        self.bn2 = nn.BatchNorm1d(max_lin_size)
        self.linear3 = nn.Linear(max_lin_size, num_classes)

    def pattern_maker(self, input_channel):
        layers = []
        # for expansion, out_channel, num_blocks, stride in self.cfg:
        #     strides = [stride] + [1]*(num_blocks-1)
        #     for stride in strides:
        #         layers.append(MBConvBlock(input_channel, out_channel, expansion, stride))
        #         input_channel = out_channel
        print(self.stride_array)
        for i in range(len(self.stride_array)):
            for stride in self.stride_array[i]:
                layers.append(MBConvBlock(input_channel, self.filters[i], self.expansion[i], stride))
                input_channel = self.filters[i]
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = out.view(out.size(0), -1)
        #out = self.linear(out)
        out = F.relu(self.bn2(self.linear(out)))
        out = F.relu(self.bn2(self.linear2(out)))
        out = self.linear3(out)
        return out


# def EfficientNetB0():
#     # # (expansion, out_channel, num_blocks, stride)
#     # cfg = [(1,  16, 1, 2),
#     #        (6,  24, 2, 1),
#     #        (6,  40, 2, 2),
#     #        (6,  80, 3, 2),
#     #        (6, 112, 3, 1),
#     #        (6, 192, 4, 2),
#     #        (6, 320, 1, 2)]
#     return EfficientNet(arguments)

def EfficientNetB0():
    # def __init__(self):
    expansion = [1, 6, 6, 6, 6, 6, 6]
    filters = [16, 24, 40, 80, 112, 192, 320]
    num_blocks = [1, 2, 2, 3, 3, 4, 1]
    stride_vals = [2, 1, 2, 2, 1, 2, 2]
    stride_array = []
    for idx in range(len(stride_vals)):
        stride_array.append([stride_vals[idx]] + [1] * (num_blocks[idx] - 1))
    print(stride_array)
    #arguments = {'expansion': expansion, 'filters': filters, 'num_blocks': num_blocks, 'stride_array': stride_array}
    return EfficientNet(expansion, filters, stride_array, num_blocks)

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight.data)