import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class PACS(Dataset):
    def __init__(self, domain, test=False) -> None:
        super().__init__()
        