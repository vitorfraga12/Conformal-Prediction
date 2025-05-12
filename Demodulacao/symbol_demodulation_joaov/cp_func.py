import numpy as np
import torch
from torch import nn
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy.stats import beta
from sklearn.model_selection import train_test_split
from tqdm import tqdm

np.random.seed(12) # for reproducibility
torch.manual_seed(12) # for reproducibility
alpha_index = 0.1 # 0.9 of confiability (1-α)


def lin2dB(x):
    return 10*np.log10(x) # Convert linear scale to dB

def dB2lin(x):
    return 10**(x/10) # Convert dB scale to linear

def get8APSK():
    # 8APSK constellation points

    # Internal and external radius
    apsk8_internal_radius = 2.0 / torch.sqrt(2.0 + (1 + torch.sqrt(torch.tensor(3.0))) ** 2)  # ~0.65
    apsk8_external_radius = torch.sqrt(2.0 - apsk8_internal_radius ** 2)  # ~1.26

    # Making the mapping
    vMapping = torch.cat(
                                [torch.tensor([+1.+1.j, -1.+1.j, +1.-1.j, -1.-1.j], dtype=torch.complex64) * apsk8_internal_radius/torch.tensor(2.0).sqrt(),
                                 torch.tensor([+1.,     +1.j,    -1,      -1.j   ], dtype=torch.complex64) * apsk8_external_radius]
                                 ) # Shifted by 45º

    


    return constellation