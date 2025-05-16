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


## 1 - 8APSK Constellation 
def get8APSK():
    """Generate 8APSK constellation parameters.
    
    Returns:
        dict: Dictionary containing:
            - constellation: 8APSK constellation points
            - internal_radius: Inner ring radius
            - external_radius: Outer ring radius
            - gray_mapping: Gray mapping
            - bit_mapping: Bit mapping
            - normalized: Normalization flag
            - description: Constellation description
    """


    # Internal and external radius
    apsk8_internal_radius = 2.0 / torch.sqrt(2.0 + (1 + torch.sqrt(torch.tensor(3.0))) ** 2)  # ~0.65
    apsk8_external_radius = torch.sqrt(2.0 - apsk8_internal_radius ** 2)  # ~1.26

    # Making the mapping
    vMapping = torch.cat(
                                [torch.tensor([+1.+1.j, -1.+1.j, +1.-1.j, -1.-1.j], dtype=torch.complex64) * apsk8_internal_radius/torch.tensor(2.0).sqrt(),
                                 torch.tensor([+1.,     +1.j,    -1,      -1.j   ], dtype=torch.complex64) * apsk8_external_radius]
                                 ) # Shifted by 45º

    gray_mapping = {
        0: 0,  # 000
        1: 1,  # 001
        2: 3,  # 011
        3: 2,  # 010
        4: 6,  # 110
        5: 7,  # 111
        6: 5,  # 101
        7: 4   # 100
    }

    bit_mapping = {
        0: [0,0,0], 1: [0,0,1], 3: [0,1,1], 2: [0,1,0],
        6: [1,1,0], 7: [1,1,1], 5: [1,0,1], 4: [1,0,0]
    }

    return {
        'constellation': vMapping,
        'internal_radius': apsk8_internal_radius,
        'external_radius': apsk8_external_radius,
        'gray_mapping': gray_mapping,
        'bit_mapping': bit_mapping,
        'normalized': True,
        'description': '8APSK constellation with Gray coding'
    }

def modulate_8apsk(bits, constellation_params=None):
    """Modulate bits to 8APSK symbols using Gray mapping.   
    
    Args:
        bits: Bit sequence for modulation (shape [N,3] or [3N])
        constellation_params: Constellation parameters (optional)
    
    Returns:
        tuple: (modulated symbols, Gray-mapped indices)
    """


    if constellation_params is None:
        constellation_params = get8APSK()
    
    # Converte bits para tensor de forma segura
    if isinstance(bits, torch.Tensor):
        bits = bits.detach().clone().float()
    else:
        bits = torch.tensor(bits, dtype=torch.float32)
    
    # Garante shape [N, 3]
    if bits.dim() == 1:
        bits = bits.view(-1, 3)
    
    # Converte bits para índices inteiros (0-7)
    bit_indices = (bits @ torch.tensor([4, 2, 1], dtype=torch.float32)).long()
    
    # Aplica mapeamento Gray - versão corrigida
    gray_indices = torch.tensor([constellation_params['gray_mapping'][idx.item()] for idx in bit_indices])
    
    # Modulação
    symbols = constellation_params['constellation'][gray_indices]
    
    return symbols, gray_indices


def generate_8apsk_samples(num_samples, pattern=0, constellation_params=None):
    """Generates 8APSK samples in different patterns.
    
    Args:
        num_samples: Number of samples to generate
        pattern: 0=cyclic sequence, 1=blocks, 2=random
        constellation_params: Constellation parameters (optional)
    
    Returns:
        tuple: (complex symbols, symbol indices)
    """
    if constellation_params is None:
        constellation_params = get8APSK()
    
    order = 8  # Ordem fixa para 8APSK
    
    # Gera os índices dos símbolos
    if pattern == 0:  # Sequência cíclica 0,1,2,...,7,0,1,...
        symbol_indices = torch.arange(0, num_samples, dtype=torch.long) % order
    elif pattern == 1:  # Blocos 0,0,...,0,1,1,...,1,...
        symbol_indices = (torch.arange(0, num_samples, dtype=torch.long) // order) % order
    else:  # Aleatório balanceado
        symbol_indices = torch.randperm(num_samples) % order
    
    # Modula os símbolos
    symbols = constellation_params['constellation'][symbol_indices]
    
    return symbols, symbol_indices

def slicer_demodulation(rx_symbols, constellation_params):
    """Performs hard decision demodulation for 8APSK constellation.
    
    Args:
        rx_symbols: Received complex symbols
        constellation_params: Dictionary containing constellation parameters
    
    Returns:
        torch.Tensor: Indices of the closest constellation points
    """


    constellation = constellation_params['constellation']
    distances = torch.abs(rx_symbols.unsqueeze(1) - constellation)
    return torch.argmin(distances, dim=1)


def plot_decision_borders(constellation_params):
    """Plots the 8APSK constellation with decision boundaries.
    
    Args:
        constellation_params: Dictionary containing constellation parameters
                            including 'constellation', 'internal_radius',
                            and 'external_radius'
    
    Returns:
        None: Displays the plot using matplotlib
    """

    # Plota os pontos da constelação
    constellation = constellation_params['constellation']
    plt.scatter(constellation.real, constellation.imag, c='b', marker='o')
    
    # Plota os círculos que separam os anéis
    r_in = constellation_params['internal_radius']
    r_out = constellation_params['external_radius']
    circle_in = plt.Circle((0,0), r_in, fill=False, linestyle='--', color='k')
    circle_out = plt.Circle((0,0), r_out, fill=False, linestyle='--', color='k')
    plt.gca().add_artist(circle_in)
    plt.gca().add_artist(circle_out)
    
    plt.grid(True)
    plt.axis('equal')
    plt.title('8APSK Constellation')
