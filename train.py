'''
DATES Lab
Transformer
Model Check for CIFAR10
Training
Made by KimJW (SSE21)
'''

import torch
import numpy as np
from tqdm import tqdm
from os import path

import Models   as M
import Trainers as T
import Datas    as D

# Set Devices (M1/M2 mps, NVIDIA cuda:0, else cpu)
device = None
if   torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

# Global Data


if __name__ == "__main__":
    pass