import torch
import torch.nn as nn
import torch.nn.functional as F

'''ResNet Block structure is used to build all of the layers.'''
class ResNetBlock(nn.Module):
    def __init__(self, input_channel_size, channel_size, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channel_size, channel_size, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channel_size)
        self.conv2 = nn.Conv2d(channel_size, channel_size, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channel_size)
        '''Describes the shortcut scenario whent he input channel size and channel size do not match. Gives us the transition
           to the larger conv layer sizes.'''
        if stride != 1 or input_channel_size != channel_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channel_size, channel_size, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel_size)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
'''The Resnet Class brings together all of the configuration settings and building blocks to formulate the entire model.'''
class ResNet(nn.Module):
    def __init__(self, block_input, hidden_sizes, stride_array, num_classes=2300):
        super(ResNet, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.input_channels = hidden_sizes[0] #64 channels
        self.stride_array = stride_array
        self.conv1 = nn.Conv2d(3, hidden_sizes[0], kernel_size=7, stride=1, padding=3, bias=False) #Decided to change stride to 1
        self.bn1 = nn.BatchNorm2d(hidden_sizes[0])
        self.relu = nn.ReLU(inplace=True)
        self.ResNet_1 = self.ResNet_block_maker(block_input, stride_array[0], hidden_sizes[0]) #Resnet Layer Construction
        self.ResNet_2 = self.ResNet_block_maker(block_input, stride_array[1], hidden_sizes[1])
        self.ResNet_3 = self.ResNet_block_maker(block_input, stride_array[2], hidden_sizes[2])
        self.ResNet_4 = self.ResNet_block_maker(block_input, stride_array[3], hidden_sizes[3])
        self.linear_label = nn.Linear(hidden_sizes[-1], num_classes)

    '''Function used to make the layers for the ResNet model structure. These layers are the repeating pattern that help 
       construct the meat of the model. '''
    def ResNet_block_maker(self, block_input, stride_size, hidden_size):
        layers = []
        for stride in stride_size:
            layers.append(block_input(self.input_channels, hidden_size, stride))
            self.input_channels = hidden_size
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.ResNet_1(out) #ResNet Block 1 forward pass
        out = self.ResNet_2(out) #ResNet Block 2 forward pass
        out = self.ResNet_3(out) #ResNet Block 3 forward pass
        out = self.ResNet_4(out) #ResNet Block 4 forward pass
        out = F.avg_pool2d(out, kernel_size=(4, 4))
        closs_out = torch.flatten(out, -1) #used to calculate the verification (used as embeddings)
        out = torch.flatten(out, 1)
        label_out = self.linear_label(out)  #linear classification for num_classes output
        return closs_out, label_out
'''Configuration paramaters for the ResNet model selected. Decided to go with ResNet 34 which has the below configuration
   settings for the model. The ResNet 34 model has 4 ResNet block layers with the block sizes, number of blocks per layer
   ,and specified strides shown below.'''
def ResNet34():
    hidden_sizes = [64, 128, 256, 512] #ResNet Block Layer sizes
    num_blocks = [3, 4, 6, 3] #Number of Blocks specified
    stride_values = [1, 2, 2, 2]  #Initial Stride Values per block
    stride_array = []
    for idx in range(len(stride_values)):  #Create the stride values for all Conv Layers in the ResNet blocks
        stride_array.append([stride_values[idx]] + [1] * (num_blocks[idx] - 1))
    return ResNet(ResNetBlock, hidden_sizes, stride_array)

''' Weight initialization for the model. Taken from recitation 6.'''
def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight.data)


