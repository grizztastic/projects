from PIL import Image
import torch
from torchvision.transforms import Normalize, ToTensor
from torch.utils.data import Dataset
'''Test Image Data loader to read in the images for the verification testing. Was modified from the Recitation 6 code.'''
class TestImageDataset(torch.utils.data.Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img = Image.open(self.file_list[index])
        img = ToTensor()(img)
        img = Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(img) #normalize the image for better results
        return img

