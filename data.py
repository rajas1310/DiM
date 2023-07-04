import torch
from glob import glob
import os
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image

from torch.utils.data import Dataset

from utils import set_seed



class PACS(Dataset):
    def __init__(self, data_path, domain, mode="train", test_split=0.2, seed=42) -> None:
        super().__init__()
        set_seed(seed)
        self.path = data_path
        self.domain = domain
        self.mode = mode
        self.label2int = {
            "dog": 0,
            "elephant": 1,
            "giraffe": 2,
            "guitar": 3,
            "horse": 4,
            "house": 5,
            "person": 6
        }
        self.int2label = {v:k for k,v in self.label2int.items()}
        self.domain_to_foldername = {
            "p": "photo",
            "a": "art_painting",
            "c": "cartoon",
            "s": "sketch"
        }
        self.folder_name = self.domain_to_foldername[self.domain]
        self.test_split = test_split
        self.split_data()

    def split_data(self):
        all_images = glob(os.path.join(self.path, self.folder_name, "*", "*"))
        self.train_images, self.test_images = train_test_split(all_images, test_size=self.test_split)
        label_fn = lambda impath: self.label2int[impath.split("/")[-2]]
        self.train_labels = list(map(label_fn, self.train_images))
        self.test_labels = list(map(label_fn, self.test_images))

    def __len__(self):
        if self.mode == "train":
            return len(self.train_images)
        else:
            return len(self.test_images)

    def __getitem__(self, idx):
        self.image_list = self.train_images if self.mode == "train" else self.test_images
        self.label_list = self.train_labels if self.mode == "train" else self.test_labels
        img = Image.open(self.image_list[idx])
        label = self.label_list[idx]
        return img, label



# TODO: create a PACS dataset class to load data of a given domain and split

if __name__ == "__main__":
    pacs = PACS(data_path="/content/data/PACS", domain="p")
    print(pacs[42])
    