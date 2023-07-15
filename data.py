import torch
from glob import glob
import os
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, ConcatDataset

from utils import set_seed

class ImageDataset(Dataset):
    def __init__(self, image_list, label_list):
        super().__init__()
        self.image_list = image_list
        self.label_list = label_list
        self.transform = transforms.Compose([transforms.ToTensor(), Resize(128)])

    def __len__(self):
       return len(self.label_list)

    def __getitem__(self, idx):
        img = self.transform(Image.open(self.image_list[idx]))
        label = self.label_list[idx]
        return img, label


class PACS:
    def __init__(self, data_path, domain, test_split=0.2, seed=42) -> None:
        set_seed(seed)
        self.path = data_path
        self.domain = domain
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

    def get_datasets(self):
        return ImageDataset(self.train_images, self.train_labels), ImageDataset(self.test_images, self.test_labels)
        


def get_pacs_datasets(data_path, holdout_domain, test_split=0.2, seed=42):
    domain_to_foldername = {
        "p": "photo",
        "a": "art_painting",
        "c": "cartoon",
        "s": "sketch"
    }

    train_dss = list()
    test_dss = list()

    for domain in domain_to_foldername:
        if domain == holdout_domain:
            pass
        else:
            pacs = PACS(data_path, domain, test_split=test_split, seed=seed)
            train_ds, test_ds = pacs.get_datasets()
            train_dss.append(train_ds)
            test_dss.append(test_ds)
    
    return ConcatDataset(train_dss), ConcatDataset(test_dss)

# TODO: create a PACS dataset class to load data of a given domain and split

# if __name__ == "__main__":
#     pacs = PACS(data_path="/content/data/PACS", domain="p")
#     print(pacs[42])
    