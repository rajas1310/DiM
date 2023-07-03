import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class PACS(Dataset):
    def __init__(self, domain, mode="train") -> None:
        super().__init__()
        self.domain = domain
        self.mode = mode
        self.domain_to_foldername = {
            "p": "photo",
            "a": "art_painting",
            "c": "cartoon",
            "s": "sketch"
        }
        self.data_path = f"data/pacs/pacs_data/{self.domain_to_foldername[domain]}"
        self.labels_path = f"data/pacs/pacs_label/{self.domain_to_foldername[domain]}_{self.mode}_kfold.txt"


# TODO: create a PACS dataset class to load data of a given domain and split
    