import numpy as np
import matplotlib.pyplot as plt
from cp_func import get8APSK, generate_8apsk_samples, slicer_demodulation, plot_decision_borders

def test_8apsk_system():
    # Configuração inicial
    num_samples = 1000
    snr_db = 20  # SNR em dB
    
    # Obtém parâmetros da constelação
    constellation_params = get8APSK()
    
    # Gera amostras em diferentes padrões
    symbols_cyclic, indices_cyclic = generate_8apsk_samples(num_samples, pattern=0, constellation_params=constellation_params)
    symbols_blocks, indices_blocks = generate_8apsk_samples(num_samples, pattern=1, constellation_params=constellation_params)
    symbols_random, indices_random = generate_8apsk_samples(num_samples, pattern=2, constellation_params=constellation_params)
    
    # Adiciona ruído gaussiano
    noise_power = 10**(-snr_db/10)
    noise = np.sqrt(noise_power/2) * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
    rx_symbols = symbols_random + noise
    
    # Demodulação
    detected_indices = slicer_demodulation(rx_symbols, constellation_params)
    
    # Calcula taxa de erro de símbolo
    ser = (detected_indices != indices_random).float().mean().item()
    
    # Plotagem dos resultados
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Constelação ideal
    plt.subplot(131)
    plot_decision_borders(constellation_params)
    plt.title('Constelação 8APSK Ideal')
    
    # Plot 2: Símbolos transmitidos
    plt.subplot(132)
    plt.scatter(symbols_random.real, symbols_random.imag, c='b', marker='o', alpha=0.5, label='Transmitidos')
    plt.grid(True)
    plt.axis('equal')
    plt.title('Símbolos Transmitidos')
    plt.legend()
    
    # Plot 3: Símbolos recebidos com ruído
    plt.subplot(133)
    plt.scatter(rx_symbols.real, rx_symbols.imag, c='r', marker='.', alpha=0.5, label='Recebidos')
    plt.scatter(symbols_random.real, symbols_random.imag, c='b', marker='o', alpha=0.5, label='Transmitidos')
    plt.grid(True)
    plt.axis('equal')
    plt.title(f'Símbolos Recebidos (SNR={snr_db}dB)\nSER={ser:.2%}')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return {
        'SER': ser,
        'Transmitted_Symbols': symbols_random,
        'Received_Symbols': rx_symbols,
        'Detected_Indices': detected_indices,
        'True_Indices': indices_random
    }

if __name__ == "__main__":
    results = test_8apsk_system()
    print(f"Taxa de Erro de Símbolo (SER): {results['SER']:.2%}")