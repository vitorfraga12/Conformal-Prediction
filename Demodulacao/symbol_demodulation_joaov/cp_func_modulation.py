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
awgn_situation = True # If True, the channel is AWGN

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



## 2 - System Model
def channel_states():
    """
    Simulates and returns a new channel state dictionary.
    
    Returns:
        dict: Dictionary containing channel state parameters:
              'tx_amp_imbalance_factor', 'tx_phase_imbalance_factor', 'ch_mult_factor'.
    """
    # Generate the channel multiplication factor
    ch_mult_factor = torch.randn(1, dtype=torch.cfloat) # Corresponds to ψ (psi) in the article 

    # Create a beta distribution for the transmitter imperfection factors
    rand_beta_dist_instance = torch.distributions.Beta(torch.tensor(5), torch.tensor(2))
    
    # If AWGN situation is forced, remove amplitude variation from fading
    if awgn_situation: 
        ch_mult_factor /= ch_mult_factor.abs() 
    
    # Sample transmitter imperfection factors using the beta distribution instance
    tx_amplitude_imbalance_factor = rand_beta_dist_instance.sample() * 0.15 # Corresponds to ϵ (epsilon) in the article 
    tx_phase_imbalance_factor = rand_beta_dist_instance.sample() * 15 * np.pi / 180 # Corresponds to δ (delta) in the article 
    
    # Return the channel state dictionary, "c" in the article
    d_channel_state = {
        'tx_amp_imbalance_factor': tx_amplitude_imbalance_factor,
        'tx_phase_imbalance_factor': tx_phase_imbalance_factor,
        'ch_mult_factor': ch_mult_factor
    }
    
    return d_channel_state

def simulate_channel_aplication(num_samples, b_enforce_pattern, b_noise_free,
                                d_setting, channel_state, mod_constellation_params):
    """
    Simulates a channel step for demodulation, applying imperfections and noise.
    
    Args:
        num_samples (int): Number of samples to generate.
        b_enforce_pattern (bool): If True, uses a fixed pattern for TX symbol generation.
        b_noise_free (bool): If True, no noise is added.
        d_setting (dict): Configuration dictionary, containing 'snr_dB'.
        channel_state (dict): Dictionary containing channel state parameters (ψ, ϵ, δ).
        mod_constellation_params (dict): Modulation constellation parameters (from get_8apsk_constellation_params).
        
    Returns:
        tuple: (rx_real_iq, tx_sym_uint)
               rx_real_iq (torch.Tensor): Received signals (real and imaginary parts separated).
               tx_sym_uint (torch.Tensor): Original transmitted symbols (integer indices).
    """
    # Define the symbol generation pattern
    pattern = 0 if b_enforce_pattern else -1

    # 1. Generate TX Symbols
    tx_iq, tx_sym_uint = generate_8apsk_samples(num_samples, pattern, mod_constellation_params)

    # 2. Get channel state parameters
    epsilon = channel_state['tx_amp_imbalance_factor'] # ϵ (epsilon) from the article
    ch_mult_factor = channel_state['ch_mult_factor']   # Complex channel factor (contains ψ)
    delta = channel_state['tx_phase_imbalance_factor'] # δ (delta) from the article

    cos_delta = torch.cos(delta) 
    sin_delta = torch.sin(delta)

    # 3. Apply Transmitter Hardware Imperfections (I/Q Imbalance and Phase Rotation, Eq. 28 from the article) [cite: 248]

    tx_distorted_real = (1 + epsilon) * (cos_delta * tx_iq.real - sin_delta * tx_iq.imag)
    tx_distorted_imag = (1 - epsilon) * (cos_delta * tx_iq.imag - sin_delta * tx_iq.real)
    
    tx_distorted = tx_distorted_real + 1j * tx_distorted_imag # This is the f_IQ function from the article [cite: 246, 248]

    # 4. Apply Channel Effect (Complex Multiplication) [cite: 246]
    tx_rayleighed = tx_distorted * ch_mult_factor

    # 5. Add Noise (AWGN) [cite: 246]
    rx_iq = tx_rayleighed # Initialize with the signal before noise addition
    if not b_noise_free: # If it's not a noise-free situation
        snr_lin = dB2lin(d_setting['snr_dB']) # Calculate linear SNR from d_setting
        noise_std_dev = np.sqrt(0.5 / snr_lin)
        noise = noise_std_dev * (torch.randn(num_samples, dtype=torch.float64) + 1j * torch.randn(num_samples, dtype=torch.float64))
        rx_iq += noise # Add noise

    # 6. Convert the received signal to real format (I and Q separated)
    # The neural network expects inputs as real tensors [num_samples, 2]
    rx_real_iq = torch.stack([rx_iq.real, rx_iq.imag], dim=1).type(torch.float64)

    return rx_real_iq, tx_sym_uint