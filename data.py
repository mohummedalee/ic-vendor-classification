import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from config import CFG

import keras
import torch
from torchvision.io import read_image


class ICImagesDataset(Dataset):
    def __init__(self, annotations, directory, transform=None, label_map_file="data/ic_chip_name_to_id_mapping.txt"):
        self.directory = directory
        self.label_encoder = keras.layers.CategoryEncoding(
            num_tokens=CFG.label_count,
            output_mode="one_hot"
        )
        self.labels = pd.read_csv(annotations, names=['file', 'label'])
        self.transform = transform         
        self.len = self.labels.shape[0]
        
        # load label map for easier inspection
        labels_text = open(label_map_file).readlines()
        labels_text = [text.split(':') for text in labels_text if text.strip()]
        self.label_map = {int(text[0]): text[1].strip() for text in labels_text}               
 

    def __len__(self):
        return self.len


    def __getitem__(self, idx):
        # image path: directory + file name from labels DF
        image_path = os.path.join(self.directory, self.labels.iloc[idx, 0])
        image = read_image(image_path)
        label = self.labels.iloc[idx, 1]
 
        # apply the transform if not set to None
        if self.transform:
            image = self.transform(image)
        
        # returning the RGB image tensor and one-hot encoded label
        return torch.Tensor(image), np.squeeze(np.array(self.label_encoder(label-1)))
    

    def inspect(self, idx):
        # return (non-encoded) human-interpretable version of image and label
        image_path = os.path.join(self.directory, self.labels.iloc[idx, 0])
        image = read_image(image_path)
        if self.transform:
            image = self.transform(image)
        label = self.labels.iloc[idx, 1]

        return image, self.label_map[label]
