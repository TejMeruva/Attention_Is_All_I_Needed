import torch
from torch.utils.data import Dataset

class TextFile(Dataset):
    def __init__(self, file_path: str):
        self.path = file_path
