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



## 3 - Data Set Preparation
def leave_one_out_data(x_input: torch.Tensor, y_output: torch.Tensor, index: int): #-> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns a new dataset with one sample removed (leave-one-out - LOO). Used to CV-CP 
    
    Args:
        x_input (torch.Tensor): Input features tensor.
        y_output (torch.Tensor): Output labels tensor.
        index (int): Index of the sample to be removed.
        
    Returns:
        tuple[torch.Tensor, torch.Tensor]: (X_loo, y_loo) - Tensors representing the dataset with one sample left out.
    """
    N_len = len(y_output)
    
    # Input validation (optional, but good practice)
    if not (0 <= index < N_len):
        raise IndexError(f"Index {index} is out of bounds for dataset of size {N_len}.")
        
    # Create indices for all samples except the one at 'index'
    indices_loo = np.concatenate((np.arange(0, index), np.arange(index + 1, N_len)), axis=0)
    
    # Convert numpy array of indices to a PyTorch tensor for advanced indexing
    indices_loo_tensor = torch.from_numpy(indices_loo).long()
    
    # Use the generated indices to select elements from X and y
    X_loo = x_input[indices_loo_tensor, :]
    y_loo = y_output[indices_loo_tensor]
    
    return X_loo, y_loo

def leave_fold_out_data(x_input: torch.Tensor, y_output: torch.Tensor, fold_index: int, num_folds: int): #-> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns a new dataset with one fold removed (leave-fold-out - LFO). Used to K-CV-CP
    
    Args:
        x_input (torch.Tensor): Input features tensor.
        y_output (torch.Tensor): Output labels tensor.
        fold_index (int): Index of the fold to be removed (0-indexed).
        num_folds (int): Total number of folds.
        
    Returns:
        tuple[torch.Tensor, torch.Tensor]: (X_lfo, y_lfo) - Tensors representing the dataset with one fold left out.
    """
    N_len = len(y_output)
    # Calculate the number of samples per fold. Rounding is consistent with original code.
    N_over_K = round(N_len / num_folds)
    
    if N_len % num_folds != 0:
        print(f"Warning: Dataset size ({N_len}) is not perfectly divisible by num_folds ({num_folds}).")

    # Input validation
    if not (0 <= fold_index < num_folds):
        raise IndexError(f"Fold index {fold_index} is out of bounds for {num_folds} folds.")

    # Calculate start and end indices of the fold to be excluded
    fold_start_idx = fold_index * N_over_K
    fold_end_idx = (fold_index + 1) * N_over_K
    
    # Concatenate indices for the segments before and after the excluded fold
    indices_lfo = np.concatenate((np.arange(0, fold_start_idx), np.arange(fold_end_idx, N_len)), axis=0)
    
    # Convert numpy array of indices to a PyTorch tensor for advanced indexing
    indices_lfo_tensor = torch.from_numpy(indices_lfo).long()
    
    # Select elements using the generated indices
    X_lfo = x_input[indices_lfo_tensor, :]
    y_lfo = y_output[indices_lfo_tensor]
    
    return X_lfo, y_lfo


def split_data_into_subsets(x_input: torch.Tensor, y_output: torch.Tensor, N_samples_first_subset: int, shuffle: bool = True):# -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """
    Splits the dataset into two subsets (e.g., training and validation). Used to VB-CP
    
    Args:
        x_input (torch.Tensor): Input features tensor.
        y_output (torch.Tensor): Output labels tensor.
        N_samples_first_subset (int): Number of samples for the first subset.
        shuffle (bool, optional): If True, shuffles the data before splitting. Defaults to True.
        
    Returns:
        tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]: 
            A tuple containing two (X, y) tuples: ((X0, y0), (X1, y1)),
            representing the training and validation subsets respectively.
    """
    N_len = len(y_output)

    # Handle edge cases where one subset is empty or all data goes into one subset
    if N_samples_first_subset == 0: # All data in second subset
        return ( (torch.empty(0, x_input.shape[1], dtype=x_input.dtype), torch.empty(0, dtype=y_output.dtype)), (x_input, y_output) )
    elif N_samples_first_subset == N_len: # All data in first subset
        return ( (x_input, y_output), (torch.empty(0, x_input.shape[1], dtype=x_input.dtype), torch.empty(0, dtype=y_output.dtype)) )
    
    # Generate permutation indices
    if shuffle:
        perm = torch.randperm(N_len)
    else:
        perm = torch.arange(0, N_len)
    
    # Split X and y using the generated permutation indices
    X0 = x_input[perm[:N_samples_first_subset], :]
    y0 = y_output[perm[:N_samples_first_subset]]
    
    X1 = x_input[perm[N_samples_first_subset:], :] # Corrected slicing for N1 (rest of the data)
    y1 = y_output[perm[N_samples_first_subset:]] # Corrected slicing for N1 (rest of the data)
    
    return ( (X0, y0), (X1, y1) )



## 4 - Neural Network Models
class FcReluDnn(nn.Module):
    """
    Fully-Connected ReLU Deep Neural Network.
    This class defines the neural network architecture.
    """
    def __init__(self, vLayers: list):
        """
        Constructor for the FcReluDnn model.
        
        Args:
            vLayers (list): A list of integers defining the number of neurons in each layer.
                            Example: [input_dim, hidden1_dim, hidden2_dim, ..., output_dim].
        """
        super(FcReluDnn, self).__init__()
        
        self.hidden = nn.ModuleList() # Use ModuleList to store linear layers
        
        # Create linear layers with ReLU activation for hidden layers
        # and a final linear layer for the output (no activation here, softmax applied later).
        for l_idx, (input_size, output_size) in enumerate(zip(vLayers, vLayers[1:])):
            # All layers use torch.float64 as specified in the original code for Hessian calculations
            linear_layer = nn.Linear(input_size, output_size, dtype=torch.float64)
            self.hidden.append(linear_layer)
        
    def forward(self, activation: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the neural network.
        
        Args:
            activation (torch.Tensor): Input tensor to the network.
        
        Returns:
            torch.Tensor: Output tensor (logits before softmax).
        """
        L = len(self.hidden) # Number of layers
        
        for l_idx, linear_transform in enumerate(self.hidden):
            activation = linear_transform(activation)
            # Apply ReLU activation for all hidden layers (not the last output layer)
            if l_idx < L - 1:
                activation = torch.nn.functional.relu(activation)
        return activation

    def num_parameters(self) -> int:
        """
        Calculates the total number of trainable parameters in the model.
        
        Returns:
            int: Total number of parameters.
        """
        return sum(torch.numel(w) for w in self.parameters())


class FcReluDnn_external(nn.Module):
    """
    Fully-Connected ReLU Deep Neural Network designed to operate with externally provided parameters.
    This is used when model parameters are managed and passed explicitly (e.g., for Hessian calculations).
    """
    def __init__(self):
        """
        Constructor for the FcReluDnn_external model.
        It does not initialize its own nn.Linear layers, as parameters are external.
        """
        super(FcReluDnn_external, self).__init__()
        # Note: No need to initialize nn.Linear layers here since parameters are provided externally.
        # This module will be used to apply linear transformations and activations given external weights/biases.
        
    def forward(self, net_in: torch.Tensor, net_params: list[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass using externally provided network parameters.
        
        Args:
            net_in (torch.Tensor): Input tensor to the network.
            net_params (list[torch.Tensor]): A list of tensors representing the weights and biases of all layers.
                                            Assumed to be ordered as [weight1, bias1, weight2, bias2, ...].
        
        Returns:
            torch.Tensor: Output tensor (logits before softmax).
        """
        # Calculate number of layers from the parameter list (each layer has weight and bias)
        L = len(net_params) // 2 
        
        for ll in range(L):
            curr_layer_weight = net_params[2 * ll]
            curr_layer_bias = net_params[2 * ll + 1]
            
            # Apply linear transformation using functional API
            net_in = torch.nn.functional.linear(net_in, curr_layer_weight, curr_layer_bias)
            
            # Apply ReLU activation for all hidden layers (not the last output layer)
            if ll < L - 1: 
                net_in = torch.nn.functional.relu(net_in)
        return net_in
