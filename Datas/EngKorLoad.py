'''
English Korean Dataset Loader
Implemented by Jiwhan Kim

Cautions:
    1. You first download datasets from Google Drive
        engtokor_train_set.json
        engtokor_test_set.json
    2. Move downloaded files to ./Datas/
'''


import torch
from torch.utils.data import DataLoader, random_split
