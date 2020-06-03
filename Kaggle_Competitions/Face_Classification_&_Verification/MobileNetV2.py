'''MobileNetV2 in PyTorch.
See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


# class Block(nn.Module):
#     '''expand + depthwise + pointwise'''
#     def __init__(self, in_planes, out_planes, expansion, stride):
#         super(Block, self).__init__()
#         self.stride = stride
#
#         planes = expansion * in_planes
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
#         self.bn3 = nn.BatchNorm2d(out_planes)
#
#         self.shortcut = nn.Sequential()
#         if stride == 1 and in_planes != out_planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.BatchNorm2d(out_planes),
#             )
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out = out + self.shortcut(x) if self.stride==1 else out
#         return out

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

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.stride == 1:
            shortcut = self.shortcut(x)
            out = out + shortcut
        return out

class MobileNet(nn.Module):
    def __init__(self, block_input, expansion, hidden_sizes, stride_array, num_classes=2300):
        super(MobileNet, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        #hidden_sizes = [16, 24, 32, 64, 96, 160, 320]
        self.input_channels = hidden_sizes[0]
        num_feats = 3
        max_out_nuerons = 1280
        self.conv1 = nn.Conv2d(num_feats, hidden_sizes[2], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self.layer_maker(block_input, stride_array[0], hidden_sizes[0], expansion[0])
        self.layer2 = self.layer_maker(block_input, stride_array[1], hidden_sizes[1], expansion[1])
        self.layer3 = self.layer_maker(block_input, stride_array[2], hidden_sizes[2], expansion[2])
        self.layer4 = self.layer_maker(block_input, stride_array[3], hidden_sizes[3], expansion[3])
        self.layer5 = self.layer_maker(block_input, stride_array[4], hidden_sizes[4], expansion[4])
        self.layer6 = self.layer_maker(block_input, stride_array[5], hidden_sizes[5], expansion[5])
        self.layer7 = self.layer_maker(block_input, stride_array[6], hidden_sizes[6], expansion[6])
        self.conv2 = nn.Conv2d(hidden_sizes[-1], max_out_nuerons, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(max_out_nuerons)
        self.linear = nn.Linear(max_out_nuerons, num_classes)

    # def _make_layers(self, in_planes):
    #     layers = []
    #     for expansion, out_planes, num_blocks, stride in self.cfg:
    #         strides = [stride] + [1]*(num_blocks-1)
    #         for stride in strides:
    #             layers.append(MBConvBlock(in_planes, out_planes, expansion, stride))
    #             in_planes = out_planes
    #     return nn.Sequential(*layers)

    def layer_maker(self, block_input, stride_size, hidden_size, expansion):
        layers = []
        for stride in stride_size:
            layers.append(block_input(self.input_channels, hidden_size, expansion, stride))
            self.input_channels = hidden_size
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # out = self.layers(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        closs_out = torch.flatten(out, -1)
        out = out.view(out.size(0), -1)
        # out = torch.flatten(out, 1)
        label_out = self.linear(out)
        return closs_out, label_out

def MobileNetV2():
    # # (expansion, out_planes, num_blocks, stride)
    # cfg = [(1, 16, 1, 1),
    #        (6, 24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
    #        (6, 32, 3, 2),
    #        (6, 64, 4, 2),
    #        (6, 96, 3, 1),
    #        (6, 160, 3, 2),
    #        (6, 320, 1, 1)]
    expansion = [1, 6, 6, 6, 6, 6, 6]
    hidden_sizes = [16, 24, 32, 64, 96, 160, 320]
    num_blocks = [1, 2, 3, 4, 3, 3, 1]
    stride_values = [1, 1, 2, 2, 1, 2, 1]
    stride_array = []
    for idx in range(len(stride_values)):
        stride_array.append([stride_values[idx]] + [1] * (num_blocks[idx] - 1))
    return MobileNet(MBConvBlock, expansion, hidden_sizes, stride_array)