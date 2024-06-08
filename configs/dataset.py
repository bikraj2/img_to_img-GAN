from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset

import configs.config as config 
class MapDataset(Dataset):
    def __init__(self,root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
        print(self.list_files)
    def __len__(self):
        return len(self.list_files)
    def __getitem__(self,index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir,img_file)
        image_data = Image.open(img_path)
        image = np.array(image_data)
        w = image_data.width
        w = w//2
        input_image = image[:,:w,:]
        target_image = image[:,w:,:]
        augmentations = config.both_transform(image=input_image,image0=target_image)
        input_image ,target_image =augmentations["image"],augmentations["image0"]
        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_input(image=target_image)["image"]
        return input_image,target_image
class TestDataset(Dataset):
    def __init__(self,root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
        print(self.list_files)
    def __len__(self):
        return len(self.list_files)
    def __getitem__(self,index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir,img_file)
        image_data = Image.open(img_path)
        image_data = image_data.convert("RGB")
        image = np.array(image_data)
        w = image_data.width
        w = w//2
        input_image = image
        target_image = image 
        augmentations = config.both_transform(image=input_image,image0=target_image)
        input_image ,target_image =augmentations["image"],augmentations["image0"]
        input_image = config.transform_only_input(image=input_image)["image"]
        return input_image,target_image
