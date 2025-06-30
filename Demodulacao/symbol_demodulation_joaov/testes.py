# ==============================================================================
# Configurações e Importações Iniciais
# ==============================================================================

import numpy as np
import torch
# Não é necessário importar nn e deepcopy separadamente, pois já estão em cp_func.py se você os colocou lá.
# from torch import nn
# from copy import deepcopy
import matplotlib.pyplot as plt # Mantido para plotagens, se houver
# Não é necessário importar scipy.stats e sklearn.model_selection separadamente, se já estiverem em cp_func.py
# from scipy.stats import beta
# from sklearn.model_selection import train_test_split
import datetime
import os
from tqdm import tqdm # Para barras de progresso

# Importar todas as funções e classes do seu arquivo cp_func.py
# Certifique-se de que o arquivo cp_func.py esteja na mesma pasta do seu notebook
from cp_func import (
    # Funções Utilitárias
    lin2dB, dB2lin,
    # Funções de Constelação
    get8APSK, modulate_8apsk, generate_8apsk_samples, slicer_demodulation, plot_decision_borders,
    # Funções do Modelo de Sistema (Canal)
    channel_states, simulate_channel_application, awgn_situation, # awgn_situation como global definida em cp_func.py
    # Funções de Preparação de Dataset
    leave_one_out_data, leave_fold_out_data, split_data_into_subsets,
    # Classes de Rede Neural
    FcReluDnn, FcReluDnn_external,
    # Funções de Treinamento
    eval_hessian, fitting_erm_ml__gd, fitting_erm_map_gd,
    # Funções de Predição e Nonconformity Score
    ensemble_predict, nonconformity_frq, nonconformity_bay,
    nonconformity_frq_giq, nonconformity_bay_giq,
    # Funções de Quantil
    quantile_from_top, quantile_from_btm,
    # Funções de Avaliação do Conformal Prediction (apenas Cobertura)
    vb__covrg, jkp_covrg, kfp_covrg
    # Outras variáveis globais de cp_func.py se você as definiu lá
    # COMPUTE_BAYES, REPARAM_COEFF
)

# --- Global Configuration Variables ---
np.random.seed(12) # For reproducibility
torch.manual_seed(12) # For reproducibility

# Parâmetros de simulação (definidos diretamente, como se viessem de argparse)
FIXED_SNR_DB = 5.0 # Signal to noise ratio per one RX antenna
FIXED_MOD_KEY = '8APSK' # Modulation key
FIXED_I_GAMMA = 8 # Index for regularization coeff gamma (from v_gamma_options below)
FIXED_GD_LR = 2e-1 # Learning rate while GDing
FIXED_GD_NUM_ITERS = 120 # Number of GD iterations (for ML and MAP)
FIXED_LMC_LR_INIT = 2e-1 # Initial Learning rate while LMC
FIXED_LMC_LR_DECAYING = 1.0 # Learning rate decaying factor over iterations
FIXED_LMC_BURN_IN = 100 # Total number of burn in first discarded model parameters while LMCing
FIXED_ENSEMBLE_SIZE = 20 # Ensemble prediction size for Bayesian methods
FIXED_NUM_SIM = 50 # Number of independent simulation runs
FIXED_COMPUTE_HESSIAN = 0 # {0: all zero hessian = faster = nonsense, 1: model-based using external network, 2: grad(grad) model agnostic}

# Parâmetros adicionais e globais do artigo
alpha_index = 0.1 # Miscoverage level (1-α is desired coverage)
# awgn_situation está definido e importado de cp_func.py
# COMPUTE_BAYES está definido e importado de cp_func.py

# Definir K para K-fold cross-validation
K_FOLDS = 4

# O tamanho do dataset (N) será fixo e não variável em um loop
# No original, v_N = np.arange(12,48+1,K). Usaremos o maior N=48 ou um múltiplo de K_FOLDS menor
N_FIXED = 48 # Fixed size of the training dataset (N)

# Número de amostras de teste (N_te original)
# LAST_LAYER_DIM virá de get8APSK().order
MOD_CONST_PARAMS_TEMP = get8APSK()
LAST_LAYER_DIM = MOD_CONST_PARAMS_TEMP['order'] # Number of classes (8 for 8APSK)
del MOD_CONST_PARAMS_TEMP # Remover variável temporária

N_TEST_SAMPLES = LAST_LAYER_DIM * 12 # Number of test points (e.g., 8 * 12 = 96)

# Miscoverage level for cross-validation-based methods
ALPHA_CVB = alpha_index

# Definir o espaço de gamma para regularização MAP
V_GAMMA_OPTIONS = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1e0, 2e0, 5e0]
GAMMA_REG_COEFF = V_GAMMA_OPTIONS[FIXED_I_GAMMA] # Selected gamma based on index

# Definir índices para algoritmos (para indexar arrays de resultados)
I_ML = 0
I_MAP = 1
I_HES = 2
I_FIM = 3
I_LMC = 4
ALG_STR_NAMES = ['frequentist ML', 'frequentist MAP', 'Bayesian using Hessian', 'Bayesian using FIM', 'Bayesian using LMC']

# Calcular o número total de iterações LMC
LMC_NUM_ITERS = FIXED_LMC_BURN_IN + FIXED_ENSEMBLE_SIZE - 1

# Definir dimensões da rede neural
FIRST_LAYER_DIM = 2 # I and Q components
V_LAYERS = [FIRST_LAYER_DIM, 10, 30, 30, LAST_LAYER_DIM] # Architecture: [Input, Hidden1, Hidden2, Hidden3, Output]

# --- Inicialização das Estruturas de Dados para Resultados ---
# A dimensão 'num_N' (variação do tamanho do dataset) será 1 agora
# Usamos `1` como dimensão para 'N' nos arrays de resultados, pois N_FIXED é um valor único para esta simulação.
M_ALLOC_2D_GLOBAL = np.empty((len(ALG_STR_NAMES), FIXED_NUM_SIM)) # Para cobertura geral (Algoritmo x Simulação)
M_ALLOC_3D_LABELS = np.empty((len(ALG_STR_NAMES), FIXED_NUM_SIM, LAST_LAYER_DIM)) # Para cobertura por label (Algoritmo x Simulação x Classe)

# Dicionários para armazenar resultados de COBERTURA (Ineficiência removida)
s_vb = {'covrg': M_ALLOC_2D_GLOBAL.copy(), 'covrg_labels': M_ALLOC_3D_LABELS.copy()}
s_vb_giq = {'covrg': M_ALLOC_2D_GLOBAL.copy(), 'covrg_labels': M_ALLOC_3D_LABELS.copy()}
s_jkp = {'covrg': M_ALLOC_2D_GLOBAL.copy(), 'covrg_labels': M_ALLOC_3D_LABELS.copy()}
s_jkp_giq = {'covrg': M_ALLOC_2D_GLOBAL.copy(), 'covrg_labels': M_ALLOC_3D_LABELS.copy()}
s_kfp = {'covrg': M_ALLOC_2D_GLOBAL.copy(), 'covrg_labels': M_ALLOC_3D_LABELS.copy()}
s_kfp_giq = {'covrg': M_ALLOC_2D_GLOBAL.copy(), 'covrg_labels': M_ALLOC_3D_LABELS.copy()}

# Históricos de perdas de treinamento/teste (ajustando dimensões para o N fixo)
# A dimensão do meio `1` é para o N_FIXED (já que não há loop sobre N, é sempre o primeiro/único N)
M_LOSS_TR_VB_ML = torch.zeros((FIXED_NUM_SIM, 1, FIXED_GD_NUM_ITERS), dtype=torch.float)
M_LOSS_TE_VB_ML = torch.zeros((FIXED_NUM_SIM, 1, FIXED_GD_NUM_ITERS), dtype=torch.float)
M_LOSS_TR_VB_MAP = torch.zeros((FIXED_NUM_SIM, 1, FIXED_GD_NUM_ITERS), dtype=torch.float)
M_LOSS_TE_VB_MAP = torch.zeros((FIXED_NUM_SIM, 1, FIXED_GD_NUM_ITERS), dtype=torch.float)

# Para JKP, a dimensão é FIXED_NUM_SIM x N_FIXED (um modelo por LOO) x GD_NUM_ITERS
M_LOSS_TR_JKP_ML = torch.zeros((FIXED_NUM_SIM, N_FIXED, FIXED_GD_NUM_ITERS), dtype=torch.float)
M_LOSS_TE_JKP_ML = torch.zeros((FIXED_NUM_SIM, N_FIXED, FIXED_GD_NUM_ITERS), dtype=torch.float)
M_LOSS_TR_JKP_MAP = torch.zeros((FIXED_NUM_SIM, N_FIXED, FIXED_GD_NUM_ITERS), dtype=torch.float)
M_LOSS_TE_JKP_MAP = torch.zeros((FIXED_NUM_SIM, N_FIXED, FIXED_GD_NUM_ITERS), dtype=torch.float)

# Para KFP, a dimensão é FIXED_NUM_SIM x K_FOLDS (um modelo por LFO) x GD_NUM_ITERS
M_LOSS_TR_KFP_ML = torch.zeros((FIXED_NUM_SIM, K_FOLDS, FIXED_GD_NUM_ITERS), dtype=torch.float)
M_LOSS_TE_KFP_ML = torch.zeros((FIXED_NUM_SIM, K_FOLDS, FIXED_GD_NUM_ITERS), dtype=torch.float)
M_LOSS_TR_KFP_MAP = torch.zeros((FIXED_NUM_SIM, K_FOLDS, FIXED_GD_NUM_ITERS), dtype=torch.float)
M_LOSS_TE_KFP_MAP = torch.zeros((FIXED_NUM_SIM, K_FOLDS, FIXED_GD_NUM_ITERS), dtype=torch.float)

# Perdas LMC (ajustando dimensões)
M_LOSS_TR_VB_LMC = torch.zeros((FIXED_NUM_SIM, 1, LMC_NUM_ITERS), dtype=torch.float)
M_LOSS_TE_VB_LMC = torch.zeros((FIXED_NUM_SIM, 1, LMC_NUM_ITERS), dtype=torch.float)
M_LOSS_TR_JKP_LMC = torch.zeros((FIXED_NUM_SIM, N_FIXED, LMC_NUM_ITERS), dtype=torch.float)
M_LOSS_TE_JKP_LMC = torch.zeros((FIXED_NUM_SIM, N_FIXED, LMC_NUM_ITERS), dtype=torch.float)
M_LOSS_TR_KFP_LMC = torch.zeros((FIXED_NUM_SIM, K_FOLDS, LMC_NUM_ITERS), dtype=torch.float)
M_LOSS_TE_KFP_LMC = torch.zeros((FIXED_NUM_SIM, K_FOLDS, LMC_NUM_ITERS), dtype=torch.float)

# --- Gerenciamento de Diretório de Saída ---
PATH_OF_RUN = 'simulation_results_fixed_N_jupyter/' # Altere o nome da pasta de resultados
try:
    os.stat(PATH_OF_RUN)
except:
    os.mkdir(PATH_OF_RUN)

# ==============================================================================
# Seção Principal da Simulação
# ==============================================================================

# Obter parâmetros da constelação (fixo para 8APSK)
# MOD_CONST_PARAMS já foi definido acima

# --- Início do Loop de Simulações Independentes ---
start_time_total = datetime.datetime.now()

# Log de tempo em arquivo (opcional, pode ser comentado para Jupyter se preferir a barra de progresso)
# itr_file_str = PATH_OF_RUN + 'iterations.txt'
# try: os.remove(itr_file_str)
# except OSError: pass

for i_s in tqdm(range(FIXED_NUM_SIM), desc="Total Simulations"):
    # Log de tempo (escrito no arquivo, descomente se usar)
    # itr_file = open(itr_file_str, 'a')
    # itr_file.write(f"iter {i_s} of {FIXED_NUM_SIM}. time {datetime.datetime.now().strftime('%H:%M:%S')}.\n")
    # itr_file.close()

    # Gerar novo estado de canal (psi, epsilon, delta) para esta rodada de simulação
    current_channel_state = channel_states()

    # Gerar dataset completo (X, y) para treinamento (N_FIXED) e teste (N_TEST_SAMPLES)
    # Estes datasets são fixos para esta rodada de simulação
    D_FULL_X, D_FULL_Y = simulate_channel_application(
        num_samples=N_FIXED,
        b_enforce_pattern=False, # Use padrão aleatório
        b_noise_free=False,
        d_setting={'snr_dB': FIXED_SNR_DB}, # SNR para a simulação
        channel_state=current_channel_state,
        mod_constellation_params=MOD_CONST_PARAMS
    )
    
    D_TEST_X, D_TEST_Y = simulate_channel_application(
        num_samples=N_TEST_SAMPLES,
        b_enforce_pattern=False, # Use padrão aleatório
        b_noise_free=False,
        d_setting={'snr_dB': FIXED_SNR_DB},
        channel_state=current_channel_state,
        mod_constellation_params=MOD_CONST_PARAMS
    )
    
    # Inicializar a rede neural (mesma inicialização para todos os métodos em cada simulação)
    net_init = FcReluDnn(V_LAYERS)
    gd_init_sd = net_init.state_dict() # Obter o estado inicial dos parâmetros para resetar modelos

    # N é fixo, correspondendo a N_FIXED
    N = N_FIXED # Para compatibilidade com nomes de variáveis na lógica

    # --- VB-CP (Validation-Based Conformal Prediction) ---
    print(f"Sim {i_s+1}/{FIXED_NUM_SIM} - VB-CP")
    # Divide D_FULL_X, D_FULL_Y em treinamento e validação para VB-CP
    N_TR_VB = int(N / 2) # Número de pontos para treinamento
    N_VAL_VB = N - N_TR_VB # Número de pontos para validação

    # D_TR_VB e D_VAL_VB são tuplas (X, y)
    (D_TR_VB_X, D_TR_VB_Y), (D_VAL_VB_X, D_VAL_VB_Y) = split_data_into_subsets(D_FULL_X, D_FULL_Y, N_TR_VB, shuffle=True)

    # Inicializar modelos para VB-CP
    model_vb_ml = deepcopy(net_init) # Modelo para Frequentist ML
    model_vb_map = deepcopy(net_init) # Modelo para Frequentist MAP
    model_vb_ens = deepcopy(net_init) # Modelo para Ensemble (Bayesiano)

    # Treinar modelo ML para VB-CP
    ml_loss_tr, ml_loss_te = fitting_erm_ml__gd(
        model=model_vb_ml, D_X=D_TR_VB_X, D_y=D_TR_VB_Y, D_te_X=D_TEST_X, D_te_y=D_TEST_Y,
        gd_init_sd=gd_init_sd, gd_lr=FIXED_GD_LR, gd_num_iters=FIXED_GD_NUM_ITERS
    )
    M_LOSS_TR_VB_ML[i_s, 0, :] = ml_loss_tr
    M_LOSS_TE_VB_ML[i_s, 0, :] = ml_loss_te

    # Treinar modelo MAP e gerar ensembles Bayesianos para VB-CP
    map_loss_tr, map_loss_te, lmc_loss_tr, lmc_loss_te, \
    ens_vb_hes, ens_vb_fim, ens_vb_lmc = fitting_erm_map_gd(
        model=model_vb_map, D_X_training=D_TR_VB_X, D_y_training=D_TR_VB_Y,
        D_test_X=D_TEST_X, D_test_y=D_TEST_Y, gd_init_sd=gd_init_sd,
        gd_lr=FIXED_GD_LR, gd_num_iters=FIXED_GD_NUM_ITERS, gamma=GAMMA_REG_COEFF,
        ensemble_size=FIXED_ENSEMBLE_SIZE, compute_hessian=FIXED_COMPUTE_HESSIAN,
        lmc_burn_in=FIXED_LMC_BURN_IN, lmc_lr_init=FIXED_LMC_LR_INIT,
        lmc_lr_decaying=FIXED_LMC_LR_DECAYING, compute_bays=COMPUTE_BAYES
    )
    M_LOSS_TR_VB_MAP[i_s, 0, :] = map_loss_tr
    M_LOSS_TE_VB_MAP[i_s, 0, :] = map_loss_te
    M_LOSS_TR_VB_LMC[i_s, 0, :] = lmc_loss_tr
    M_LOSS_TE_VB_LMC[i_s, 0, :] = lmc_loss_te

    # Calcular NC scores para VB-CP (no conjunto de VALIDAÇÃO)
    v_NC_vb_ml = nonconformity_frq(D_VAL_VB_X, D_VAL_VB_Y, model_vb_ml) # Eq. 15
    v_NC_vb_map = nonconformity_frq(D_VAL_VB_X, D_VAL_VB_Y, model_vb_map) # Eq. 15
    v_NC_vb_hes = nonconformity_bay(D_VAL_VB_X, D_VAL_VB_Y, model_vb_ens, ens_vb_hes) # Eq. 15
    v_NC_vb_fim = nonconformity_bay(D_VAL_VB_X, D_VAL_VB_Y, model_vb_ens, ens_vb_fim) # Eq. 15
    v_NC_vb_lmc = nonconformity_bay(D_VAL_VB_X, D_VAL_VB_Y, model_vb_ens, ens_vb_lmc) # Eq. 15
    
    v_NC_vb_ml_giq = nonconformity_frq_giq(D_VAL_VB_X, D_VAL_VB_Y, model_vb_ml)
    v_NC_vb_map_giq = nonconformity_frq_giq(D_VAL_VB_X, D_VAL_VB_Y, model_vb_map)
    v_NC_vb_hes_giq = nonconformity_bay_giq(D_VAL_VB_X, D_VAL_VB_Y, model_vb_ens, ens_vb_hes)
    v_NC_vb_fim_giq = nonconformity_bay_giq(D_VAL_VB_X, D_VAL_VB_Y, model_vb_ens, ens_vb_fim)
    v_NC_vb_lmc_giq = nonconformity_bay_giq(D_VAL_VB_X, D_VAL_VB_Y, model_vb_ens, ens_vb_lmc)

    # Preparar X_pairs e y_pairs para cálculo de NC scores prospectivos (para cada X_test e cada Y_prime)
    Y_PRIME = torch.tensor(range(LAST_LAYER_DIM)) # Possible output labels
    X_PAIRS = D_TEST_X.repeat_interleave(len(Y_PRIME), dim=0) # X_pairs original
    Y_PAIRS = Y_PRIME.repeat(D_TEST_X.shape[0]) # y_pairs original

    # Calcular NC scores prospectivos para VB-CP (no conjunto de TESTE)
    m_NC_vb_ml_prs = nonconformity_frq(X_PAIRS, Y_PAIRS, model_vb_ml).view(N_TEST_SAMPLES, LAST_LAYER_DIM)
    m_NC_vb_map_prs = nonconformity_frq(X_PAIRS, Y_PAIRS, model_vb_map).view(N_TEST_SAMPLES, LAST_LAYER_DIM)
    m_NC_vb_hes_prs = nonconformity_bay(X_PAIRS, Y_PAIRS, model_vb_ens, ens_vb_hes).view(N_TEST_SAMPLES, LAST_LAYER_DIM)
    m_NC_vb_fim_prs = nonconformity_bay(X_PAIRS, Y_PAIRS, model_vb_ens, ens_vb_fim).view(N_TEST_SAMPLES, LAST_LAYER_DIM)
    m_NC_vb_lmc_prs = nonconformity_bay(X_PAIRS, Y_PAIRS, model_vb_ens, ens_vb_lmc).view(N_TEST_SAMPLES, LAST_LAYER_DIM)

    m_NC_vb_ml_prs_giq = nonconformity_frq_giq(X_PAIRS, Y_PAIRS, model_vb_ml).view(N_TEST_SAMPLES, LAST_LAYER_DIM)
    m_NC_vb_map_prs_giq = nonconformity_frq_giq(X_PAIRS, Y_PAIRS, model_vb_map).view(N_TEST_SAMPLES, LAST_LAYER_DIM)
    m_NC_vb_hes_prs_giq = nonconformity_bay_giq(X_PAIRS, Y_PAIRS, model_vb_ens, ens_vb_hes).view(N_TEST_SAMPLES, LAST_LAYER_DIM)
    m_NC_vb_fim_prs_giq = nonconformity_bay_giq(X_PAIRS, Y_PAIRS, model_vb_ens, ens_vb_fim).view(N_TEST_SAMPLES, LAST_LAYER_DIM)
    m_NC_vb_lmc_prs_giq = nonconformity_bay_giq(X_PAIRS, Y_PAIRS, model_vb_ens, ens_vb_lmc).view(N_TEST_SAMPLES, LAST_LAYER_DIM)

    # Armazenar resultados de cobertura para VB-CP (sem ineficiência)
    s_vb_['covrg'][I_ML, i_s], s_vb_['covrg_labels'][I_ML, i_s, :] = vb__covrg(m_NC_vb_ml_prs, v_NC_vb_ml, D_TEST_Y, ALPHA)
    s_vb_['covrg'][I_MAP, i_s], s_vb_['covrg_labels'][I_MAP, i_s, :] = vb__covrg(m_NC_vb_map_prs, v_NC_vb_map, D_TEST_Y, ALPHA)
    s_vb_['covrg'][I_HES, i_s], s_vb_['covrg_labels'][I_HES, i_s, :] = vb__covrg(m_NC_vb_hes_prs, v_NC_vb_hes, D_TEST_Y, ALPHA)
    s_vb_['covrg'][I_FIM, i_s], s_vb_['covrg_labels'][I_FIM, i_s, :] = vb__covrg(m_NC_vb_fim_prs, v_NC_vb_fim, D_TEST_Y, ALPHA)
    s_vb_['covrg'][I_LMC, i_s], s_vb_['covrg_labels'][I_LMC, i_s, :] = vb__covrg(m_NC_vb_lmc_prs, v_NC_vb_lmc, D_TEST_Y, ALPHA)
    
    s_vb_giq['covrg'][I_ML, i_s], s_vb_giq['covrg_labels'][I_ML, i_s, :] = vb__covrg(m_NC_vb_ml_prs_giq, v_NC_vb_ml_giq, D_TEST_Y, ALPHA)
    s_vb_giq['covrg'][I_MAP, i_s], s_vb_giq['covrg_labels'][I_MAP, i_s, :] = vb__covrg(m_NC_vb_map_prs_giq, v_NC_vb_map_giq, D_TEST_Y, ALPHA)
    s_vb_giq['covrg'][I_HES, i_s], s_vb_giq['covrg_labels'][I_HES, i_s, :] = vb__covrg(m_NC_vb_hes_prs_giq, v_NC_vb_hes_giq, D_TEST_Y, ALPHA)
    s_vb_giq['covrg'][I_FIM, i_s], s_vb_giq['covrg_labels'][I_FIM, i_s, :] = vb__covrg(m_NC_vb_fim_prs_giq, v_NC_vb_fim_giq, D_TEST_Y, ALPHA)
    s_vb_giq['covrg'][I_LMC, i_s], s_vb_giq['covrg_labels'][I_LMC, i_s, :] = vb__covrg(m_NC_vb_lmc_prs_giq, v_NC_vb_lmc_giq, D_TEST_Y, ALPHA)


    # --- JKP (Jackknife+ Conformal Prediction) ---
    print(f"Sim {i_s+1}/{FIXED_NUM_SIM} - JKP")
    # Treina N modelos (Leave-One-Out) para JKP
    l_model_jkp_ml = [] # Lista de modelos ML para JKP
    l_model_jkp_map = [] # Lista de modelos MAP para JKP
    l_ens_jkp_hes = [] # Lista de ensembles Hessian para JKP
    l_ens_jkp_fim = [] # Lista de ensembles FIM para JKP
    l_ens_jkp_lmc = [] # Lista de ensembles LMC para JKP

    # Tensors para armazenar NC scores de calibração para JKP
    v_NC_jkp_ml = torch.zeros(N, dtype=torch.float)
    v_NC_jkp_map = torch.zeros(N, dtype=torch.float)
    v_NC_jkp_hes = torch.zeros(N, dtype=torch.float)
    v_NC_jkp_fim = torch.zeros(N, dtype=torch.float)
    v_NC_jkp_lmc = torch.zeros(N, dtype=torch.float)
    v_NC_jkp_ml_giq = torch.zeros(N, dtype=torch.float)
    v_NC_jkp_map_giq = torch.zeros(N, dtype=torch.float)
    v_NC_jkp_hes_giq = torch.zeros(N, dtype=torch.float)
    v_NC_jkp_fim_giq = torch.zeros(N, dtype=torch.float)
    v_NC_jkp_lmc_giq = torch.zeros(N, dtype=torch.float)

    # Tensors para armazenar NC scores prospectivos (teste vs calibração) para JKP
    # Dimensão [N_te, num_classes, N_total_calibration_samples]
    m_NC_jkp_ml_prs = torch.zeros((N_TEST_SAMPLES, LAST_LAYER_DIM, N), dtype=torch.float)
    m_NC_jkp_map_prs = torch.zeros((N_TEST_SAMPLES, LAST_LAYER_DIM, N), dtype=torch.float)
    m_NC_jkp_hes_prs = torch.zeros((N_TEST_SAMPLES, LAST_LAYER_DIM, N), dtype=torch.float)
    m_NC_jkp_fim_prs = torch.zeros((N_TEST_SAMPLES, LAST_LAYER_DIM, N), dtype=torch.float)
    m_NC_jkp_lmc_prs = torch.zeros((N_TEST_SAMPLES, LAST_LAYER_DIM, N), dtype=torch.float)
    m_NC_jkp_ml_prs_giq = torch.zeros((N_TEST_SAMPLES, LAST_LAYER_DIM, N), dtype=torch.float)
    m_NC_jkp_map_prs_giq = torch.zeros((N_TEST_SAMPLES, LAST_LAYER_DIM, N), dtype=torch.float)
    m_NC_jkp_hes_prs_giq = torch.zeros((N_TEST_SAMPLES, LAST_LAYER_DIM, N), dtype=torch.float)
    m_NC_jkp_fim_prs_giq = torch.zeros((N_TEST_SAMPLES, LAST_LAYER_DIM, N), dtype=torch.float)
    m_NC_jkp_lmc_prs_giq = torch.zeros((N_TEST_SAMPLES, LAST_LAYER_DIM, N), dtype=torch.float)

    # Loop para treinar N modelos (LOO) e coletar NC scores para JKP
    for i_loo in tqdm(range(N), desc="JKP LOO Training"): # N is D_FULL_X's length
        # Criar dataset Leave-One-Out
        D_LOO_X, D_LOO_Y = leave_one_out_data(D_FULL_X, D_FULL_Y, i_loo)

        # Inicializar modelos para esta iteração LOO
        model_jkp_ml = deepcopy(net_init)
        model_jkp_map = deepcopy(net_init)
        model_jkp_ens = deepcopy(net_init) # Modelo para ensemble, parâmetros carregados por torch.nn.utils.vector_to_parameters

        # Treinar ML model para JKP
        ml_loss_tr, ml_loss_te = fitting_erm_ml__gd(
            model=model_jkp_ml, D_X=D_LOO_X, D_y=D_LOO_Y, D_te_X=D_TEST_X, D_te_y=D_TEST_Y,
            gd_init_sd=gd_init_sd, gd_lr=FIXED_GD_LR, gd_num_iters=FIXED_GD_NUM_ITERS
        )
        M_LOSS_TR_JKP_ML[i_s, i_loo, :] = ml_loss_tr
        M_LOSS_TE_JKP_ML[i_s, i_loo, :] = ml_loss_te
        
        # Treinar MAP model e gerar ensembles para JKP
        map_loss_tr, map_loss_te, lmc_loss_tr, lmc_loss_te, \
        ens_jkp_hes, ens_jkp_fim, ens_jkp_lmc = fitting_erm_map_gd(
            model=model_jkp_map, D_X_training=D_LOO_X, D_y_training=D_LOO_Y,
            D_test_X=D_TEST_X, D_test_y=D_TEST_Y, gd_init_sd=gd_init_sd,
            gd_lr=FIXED_GD_LR, gd_num_iters=FIXED_GD_NUM_ITERS, gamma=GAMMA_REG_COEFF,
            ensemble_size=FIXED_ENSEMBLE_SIZE, compute_hessian=FIXED_COMPUTE_HESSIAN,
            lmc_burn_in=FIXED_LMC_BURN_IN, lmc_lr_init=FIXED_LMC_LR_INIT,
            lmc_lr_decaying=FIXED_LMC_LR_DECAYING, compute_bays=COMPUTE_BAYES
        )
        M_LOSS_TR_JKP_MAP[i_s, i_loo, :] = map_loss_tr
        M_LOSS_TE_JKP_MAP[i_s, i_loo, :] = map_loss_te
        M_LOSS_TR_JKP_LMC[i_s, i_loo, :] = lmc_loss_tr
        M_LOSS_TE_JKP_LMC[i_s, i_loo, :] = lmc_loss_te
        
        # Armazenar modelos e ensembles em listas para uso posterior
        l_model_jkp_ml.append(model_jkp_ml)
        l_model_jkp_map.append(model_jkp_map)
        l_ens_jkp_hes.append(ens_jkp_hes)
        l_ens_jkp_fim.append(ens_jkp_fim)
        l_ens_jkp_lmc.append(ens_jkp_lmc)

        # Calcular NC scores de calibração (do ponto "left out") para JKP
        x_i_loo = D_FULL_X[i_loo, :].view(1, -1) # Ponto removido (calibração)
        y_i_loo = D_FULL_Y[i_loo].view(-1) # Rótulo do ponto removido

        v_NC_jkp_ml[i_loo] = nonconformity_frq(x_i_loo, y_i_loo, model_jkp_ml) # Eq. 15
        v_NC_jkp_map[i_loo] = nonconformity_frq(x_i_loo, y_i_loo, model_jkp_map) # Eq. 15
        v_NC_jkp_hes[i_loo] = nonconformity_bay(x_i_loo, y_i_loo, model_jkp_ens, ens_jkp_hes) # Eq. 15
        v_NC_jkp_fim[i_loo] = nonconformity_bay(x_i_loo, y_i_loo, model_jkp_ens, ens_jkp_fim) # Eq. 15
        v_NC_jkp_lmc[i_loo] = nonconformity_bay(x_i_loo, y_i_loo, model_jkp_ens, ens_jkp_lmc) # Eq. 15
        v_NC_jkp_ml_giq[i_loo] = nonconformity_frq_giq(x_i_loo, y_i_loo, model_jkp_ml)
        v_NC_jkp_map_giq[i_loo] = nonconformity_frq_giq(x_i_loo, y_i_loo, model_jkp_map)
        v_NC_jkp_hes_giq[i_loo] = nonconformity_bay_giq(x_i_loo, y_i_loo, model_jkp_ens, ens_jkp_hes)
        v_NC_jkp_fim_giq[i_loo] = nonconformity_bay_giq(x_i_loo, y_i_loo, model_jkp_ens, ens_jkp_fim)
        v_NC_jkp_lmc_giq[i_loo] = nonconformity_bay_giq(x_i_loo, y_i_loo, model_jkp_ens, ens_jkp_lmc)

        # Calcular NC scores prospectivos (teste vs calibração) para JKP
        # m_NC_jkp_ml_prs[:,:,i_loo] armazena o NC score prospectivo calculado com o modelo
        # treinado SEM essa amostra `i_loo`.
        m_NC_jkp_ml_prs[:, :, i_loo] = nonconformity_frq(X_PAIRS, Y_PAIRS, l_model_jkp_ml[i_loo]).view(N_TEST_SAMPLES, LAST_LAYER_DIM)
        m_NC_jkp_map_prs[:, :, i_loo] = nonconformity_frq(X_PAIRS, Y_PAIRS, l_model_jkp_map[i_loo]).view(N_TEST_SAMPLES, LAST_LAYER_DIM)
        m_NC_jkp_hes_prs[:, :, i_loo] = nonconformity_bay(X_PAIRS, Y_PAIRS, model_jkp_ens, l_ens_jkp_hes[i_loo]).view(N_TEST_SAMPLES, LAST_LAYER_DIM)
        m_NC_jkp_fim_prs[:, :, i_loo] = nonconformity_bay(X_PAIRS, Y_PAIRS, model_jkp_ens, l_ens_jkp_fim[i_loo]).view(N_TEST_SAMPLES, LAST_LAYER_DIM)
        m_NC_jkp_lmc_prs[:, :, i_loo] = nonconformity_bay(X_PAIRS, Y_PAIRS, model_jkp_ens, l_ens_jkp_lmc[i_loo]).view(N_TEST_SAMPLES, LAST_LAYER_DIM)
        m_NC_jkp_ml_prs_giq[:, :, i_loo] = nonconformity_frq_giq(X_PAIRS, Y_PAIRS, l_model_jkp_ml[i_loo]).view(N_TEST_SAMPLES, LAST_LAYER_DIM)
        m_NC_jkp_map_prs_giq[:, :, i_loo] = nonconformity_frq_giq(X_PAIRS, Y_PAIRS, l_model_jkp_map[i_loo]).view(N_TEST_SAMPLES, LAST_LAYER_DIM)
        m_NC_jkp_hes_prs_giq[:, :, i_loo] = nonconformity_bay_giq(X_PAIRS, Y_PAIRS, model_jkp_ens, l_ens_jkp_hes[i_loo]).view(N_TEST_SAMPLES, LAST_LAYER_DIM)
        m_NC_jkp_fim_prs_giq[:, :, i_loo] = nonconformity_bay_giq(X_PAIRS, Y_PAIRS, model_jkp_ens, l_ens_jkp_fim[i_loo]).view(N_TEST_SAMPLES, LAST_LAYER_DIM)
        m_NC_jkp_lmc_prs_giq[:, :, i_loo] = nonconformity_bay_giq(X_PAIRS, Y_PAIRS, model_jkp_ens, l_ens_jkp_lmc[i_loo]).view(N_TEST_SAMPLES, LAST_LAYER_DIM)

    # Armazenar resultados de cobertura para JKP (sem ineficiência)
    s_jkp['covrg'][I_ML, i_s], s_jkp['covrg_labels'][I_ML, i_s, :] = jkp_covrg(m_NC_jkp_ml_prs, v_NC_jkp_ml, D_TEST_Y, ALPHA_CVB) # Eq. 18
    s_jkp['covrg'][I_MAP, i_s], s_jkp['covrg_labels'][I_MAP, i_s, :] = jkp_covrg(m_NC_jkp_map_prs, v_NC_jkp_map, D_TEST_Y, ALPHA_CVB) # Eq. 18
    s_jkp['covrg'][I_HES, i_s], s_jkp['covrg_labels'][I_HES, i_s, :] = jkp_covrg(m_NC_jkp_hes_prs, v_NC_jkp_hes, D_TEST_Y, ALPHA_CVB) # Eq. 18
    s_jkp['covrg'][I_FIM, i_s], s_jkp['covrg_labels'][I_FIM, i_s, :] = jkp_covrg(m_NC_jkp_fim_prs, v_NC_jkp_fim, D_TEST_Y, ALPHA_CVB) # Eq. 18
    s_jkp['covrg'][I_LMC, i_s], s_jkp['covrg_labels'][I_LMC, i_s, :] = jkp_covrg(m_NC_jkp_lmc_prs, v_NC_jkp_lmc, D_TEST_Y, ALPHA_CVB) # Eq. 18
    s_jkp_giq['covrg'][I_ML, i_s], s_jkp_giq['covrg_labels'][I_ML, i_s, :] = jkp_covrg(m_NC_jkp_ml_prs_giq, v_NC_jkp_ml_giq, D_TEST_Y, ALPHA_CVB) # Eq. 18
    s_jkp_giq['covrg'][I_MAP, i_s], s_jkp_giq['covrg_labels'][I_MAP, i_s, :] = jkp_covrg(m_NC_jkp_map_prs_giq, v_NC_jkp_map_giq, D_TEST_Y, ALPHA_CVB) # Eq. 18
    s_jkp_giq['covrg'][I_HES, i_s], s_jkp_giq['covrg_labels'][I_HES, i_s, :] = jkp_covrg(m_NC_jkp_hes_prs_giq, v_NC_jkp_hes_giq, D_TEST_Y, ALPHA_CVB) # Eq. 18
    s_jkp_giq['covrg'][I_FIM, i_s], s_jkp_giq['covrg_labels'][I_FIM, i_s, :] = jkp_covrg(m_NC_jkp_fim_prs_giq, v_NC_jkp_fim_giq, D_TEST_Y, ALPHA_CVB) # Eq. 18
    s_jkp_giq['covrg'][I_LMC, i_s], s_jkp_giq['covrg_labels'][I_LMC, i_s, :] = jkp_covrg(m_NC_jkp_lmc_prs_giq, v_NC_jkp_lmc_giq, D_TEST_Y, ALPHA_CVB) # Eq. 18

    # --- KFP (K-fold Conformal Prediction) ---
    print(f"Sim {i_s+1}/{FIXED_NUM_SIM} - KFP")
    # Treina K modelos (Leave-Fold-Out) para KFP
    N_OVER_K = round(N / K_FOLDS)
    assert(N_OVER_K * K_FOLDS == N) # Assegurar divisibilidade perfeita para K-folds

    l_model_kfp_ml = []
    l_model_kfp_map = []
    l_ens_kfp_hes = []
    l_ens_kfp_fim = []
    l_ens_kfp_lmc = []

    # Tensors para armazenar NC scores de calibração para KFP
    v_NC_kfp_ml = torch.zeros(N, dtype=torch.float)
    v_NC_kfp_map = torch.zeros(N, dtype=torch.float)
    v_NC_kfp_hes = torch.zeros(N, dtype=torch.float)
    v_NC_kfp_fim = torch.zeros(N, dtype=torch.float)
    v_NC_kfp_lmc = torch.zeros(N, dtype=torch.float)
    v_NC_kfp_ml_giq = torch.zeros(N, dtype=torch.float)
    v_NC_kfp_map_giq = torch.zeros(N, dtype=torch.float)
    v_NC_kfp_hes_giq = torch.zeros(N, dtype=torch.float)
    v_NC_kfp_fim_giq = torch.zeros(N, dtype=torch.float)
    v_NC_kfp_lmc_giq = torch.zeros(N, dtype=torch.float)

    # Tensors para armazenar NC scores prospectivos (teste vs calibração) para KFP
    m_NC_kfp_ml_prs = torch.zeros((N_TEST_SAMPLES, LAST_LAYER_DIM, N), dtype=torch.float)
    m_NC_kfp_map_prs = torch.zeros((N_TEST_SAMPLES, LAST_LAYER_DIM, N), dtype=torch.float)
    m_NC_kfp_hes_prs = torch.zeros((N_TEST_SAMPLES, LAST_LAYER_DIM, N), dtype=torch.float)
    m_NC_kfp_fim_prs = torch.zeros((N_TEST_SAMPLES, LAST_LAYER_DIM, N), dtype=torch.float)
    m_NC_kfp_lmc_prs = torch.zeros((N_TEST_SAMPLES, LAST_LAYER_DIM, N), dtype=torch.float)
    m_NC_kfp_ml_prs_giq = torch.zeros((N_TEST_SAMPLES, LAST_LAYER_DIM, N), dtype=torch.float)
    m_NC_kfp_map_prs_giq = torch.zeros((N_TEST_SAMPLES, LAST_LAYER_DIM, N), dtype=torch.float)
    m_NC_kfp_hes_prs_giq = torch.zeros((N_TEST_SAMPLES, LAST_LAYER_DIM, N), dtype=torch.float)
    m_NC_kfp_fim_prs_giq = torch.zeros((N_TEST_SAMPLES, LAST_LAYER_DIM, N), dtype=torch.float)
    m_NC_kfp_lmc_prs_giq = torch.zeros((N_TEST_SAMPLES, LAST_LAYER_DIM, N), dtype=torch.float)

    # Loop para treinar K modelos (LFO) e coletar NC scores para KFP
    V_K_FOLDS_MAP = torch.arange(K_FOLDS).repeat_interleave(N_OVER_K) # Mapping for index to fold
    for k_idx in tqdm(range(K_FOLDS), desc="KFP LFO Training"):
        # Criar dataset Leave-Fold-Out
        D_LFO_X, D_LFO_Y = leave_fold_out_data(D_FULL_X, D_FULL_Y, k_idx, K_FOLDS)

        # Inicializar modelos para esta iteração LFO
        model_kfp_ml = deepcopy(net_init)
        model_kfp_map = deepcopy(net_init)
        model_kfp_ens = deepcopy(net_init)

        # Treinar ML model para KFP
        ml_loss_tr, ml_loss_te = fitting_erm_ml__gd(
            model=model_kfp_ml, D_X=D_LFO_X, D_y=D_LFO_Y, D_te_X=D_TEST_X, D_te_y=D_TEST_Y,
            gd_init_sd=gd_init_sd, gd_lr=FIXED_GD_LR, gd_num_iters=FIXED_GD_NUM_ITERS
        )
        M_LOSS_TR_KFP_ML[i_s, k_idx, :] = ml_loss_tr
        M_LOSS_TE_KFP_ML[i_s, k_idx, :] = ml_loss_te
        
        # Treinar MAP model e gerar ensembles para KFP
        map_loss_tr, map_loss_te, lmc_loss_tr, lmc_loss_te, \
        ens_kfp_hes, ens_kfp_fim, ens_kfp_lmc = fitting_erm_map_gd(
            model=model_kfp_map, D_X_training=D_LFO_X, D_y_training=D_LFO_Y,
            D_test_X=D_TEST_X, D_test_y=D_TEST_Y, gd_init_sd=gd_init_sd,
            gd_lr=FIXED_GD_LR, gd_num_iters=FIXED_GD_NUM_ITERS, gamma=GAMMA_REG_COEFF,
            ensemble_size=FIXED_ENSEMBLE_SIZE, compute_hessian=FIXED_COMPUTE_HESSIAN,
            lmc_burn_in=FIXED_LMC_BURN_IN, lmc_lr_init=FIXED_LMC_LR_INIT,
            lmc_lr_decaying=FIXED_LMC_LR_DECAYING, compute_bays=COMPUTE_BAYES
        )
        M_LOSS_TR_KFP_MAP[i_s, k_idx, :] = map_loss_tr
        M_LOSS_TE_KFP_MAP[i_s, k_idx, :] = map_loss_te
        M_LOSS_TR_KFP_LMC[i_s, k_idx, :] = lmc_loss_tr
        M_LOSS_TE_KFP_LMC[i_s, k_idx, :] = lmc_loss_te
        
        # Armazenar modelos e ensembles em listas para uso posterior
        l_model_kfp_ml.append(model_kfp_ml)
        l_model_kfp_map.append(model_kfp_map)
        l_ens_kfp_hes.append(ens_kfp_hes)
        l_ens_kfp_fim.append(ens_kfp_fim)
        l_ens_kfp_lmc.append(ens_kfp_lmc)

    # Calcular NC scores de calibração (do ponto "left out") para KFP
    for i_sample in tqdm(range(N), desc="KFP NC Score Calculation"): # Loop sobre todos os N pontos do D_FULL
        k_idx_for_sample = V_K_FOLDS_MAP[i_sample] # Determina a qual fold este sample pertence
        x_i_lfo = D_FULL_X[i_sample, :].view(1, -1) # O ponto de calibração
        y_i_lfo = D_FULL_Y[i_sample].view(-1) # O rótulo do ponto de calibração

        # Calcula o NC score para o ponto `x_i_lfo, y_i_lfo` usando o modelo *que não o viu* (i.e., do fold `k_idx_for_sample`)
        v_NC_kfp_ml[i_sample] = nonconformity_frq(x_i_lfo, y_i_lfo, l_model_kfp_ml[k_idx_for_sample]) # Eq. 15
        v_NC_kfp_map[i_sample] = nonconformity_frq(x_i_lfo, y_i_lfo, l_model_kfp_map[k_idx_for_sample]) # Eq. 15
        v_NC_kfp_hes[i_sample] = nonconformity_bay(x_i_lfo, y_i_lfo, model_kfp_ens, l_ens_kfp_hes[k_idx_for_sample]) # Eq. 15
        v_NC_kfp_fim[i_sample] = nonconformity_bay(x_i_lfo, y_i_lfo, model_kfp_ens, l_ens_kfp_fim[k_idx_for_sample]) # Eq. 15
        v_NC_kfp_lmc[i_sample] = nonconformity_bay(x_i_lfo, y_i_lfo, model_kfp_ens, l_ens_kfp_lmc[k_idx_for_sample]) # Eq. 15
        v_NC_kfp_ml_giq[i_sample] = nonconformity_frq_giq(x_i_lfo, y_i_lfo, l_model_kfp_ml[k_idx_for_sample])
        v_NC_kfp_map_giq[i_sample] = nonconformity_frq_giq(x_i_lfo, y_i_lfo, l_model_kfp_map[k_idx_for_sample])
        v_NC_kfp_hes_giq[i_sample] = nonconformity_bay_giq(x_i_lfo, y_i_lfo, model_kfp_ens, l_ens_kfp_hes[k_idx_for_sample])
        v_NC_kfp_fim_giq[i_sample] = nonconformity_bay_giq(x_i_lfo, y_i_lfo, model_kfp_ens, l_ens_kfp_fim[k_idx_for_sample])
        v_NC_kfp_lmc_giq[i_sample] = nonconformity_bay_giq(x_i_lfo, y_i_lfo, model_kfp_ens, l_ens_kfp_lmc[k_idx_for_sample])

        # Calcular NC scores prospectivos (teste vs calibração) para KFP
        m_NC_kfp_ml_prs[:, :, i_sample] = nonconformity_frq(X_PAIRS, Y_PAIRS, l_model_kfp_ml[k_idx_for_sample]).view(N_TEST_SAMPLES, LAST_LAYER_DIM)
        m_NC_kfp_map_prs[:, :, i_sample] = nonconformity_frq(X_PAIRS, Y_PAIRS, l_model_kfp_map[k_idx_for_sample]).view(N_TEST_SAMPLES, LAST_LAYER_DIM)
        m_NC_kfp_hes_prs[:, :, i_sample] = nonconformity_bay(X_PAIRS, Y_PAIRS, model_kfp_ens, l_ens_kfp_hes[k_idx_for_sample]).view(N_TEST_SAMPLES, LAST_LAYER_DIM)
        m_NC_kfp_fim_prs[:, :, i_sample] = nonconformity_bay(X_PAIRS, Y_PAIRS, model_kfp_ens, l_ens_kfp_fim[k_idx_for_sample]).view(N_TEST_SAMPLES, LAST_LAYER_DIM)
        m_NC_kfp_lmc_prs[:, :, i_sample] = nonconformity_bay(X_PAIRS, Y_PAIRS, model_kfp_ens, l_ens_kfp_lmc[k_idx_for_sample]).view(N_TEST_SAMPLES, LAST_LAYER_DIM)
        m_NC_kfp_ml_prs_giq[:, :, i_sample] = nonconformity_frq_giq(X_PAIRS, Y_PAIRS, l_model_kfp_ml[k_idx_for_sample]).view(N_TEST_SAMPLES, LAST_LAYER_DIM)
        m_NC_kfp_map_prs_giq[:, :, i_sample] = nonconformity_frq_giq(X_PAIRS, Y_PAIRS, l_model_kfp_map[k_idx_for_sample]).view(N_TEST_SAMPLES, LAST_LAYER_DIM)
        m_NC_kfp_hes_prs_giq[:, :, i_sample] = nonconformity_bay_giq(X_PAIRS, Y_PAIRS, model_kfp_ens, l_ens_kfp_hes[k_idx_for_sample]).view(N_TEST_SAMPLES, LAST_LAYER_DIM)
        m_NC_kfp_fim_prs_giq[:, :, i_sample] = nonconformity_bay_giq(X_PAIRS, Y_PAIRS, model_kfp_ens, l_ens_kfp_fim[k_idx_for_sample]).view(N_TEST_SAMPLES, LAST_LAYER_DIM)
        m_NC_kfp_lmc_prs_giq[:, :, i_sample] = nonconformity_bay_giq(X_PAIRS, Y_PAIRS, model_kfp_ens, l_ens_kfp_lmc[k_idx_for_sample]).view(N_TEST_SAMPLES, LAST_LAYER_DIM)

    # Armazenar resultados de cobertura para KFP (sem ineficiência)
    s_kfp['covrg'][I_ML, i_s], s_kfp['covrg_labels'][I_ML, i_s, :] = kfp_covrg(m_NC_kfp_ml_prs, v_NC_kfp_ml, D_TEST_Y, ALPHA_CVB) # Eq. 18
    s_kfp['covrg'][I_MAP, i_s], s_kfp['covrg_labels'][I_MAP, i_s, :] = kfp_covrg(m_NC_kfp_map_prs, v_NC_kfp_map, D_TEST_Y, ALPHA_CVB) # Eq. 18
    s_kfp['covrg'][I_HES, i_s], s_kfp['covrg_labels'][I_HES, i_s, :] = kfp_covrg(m_NC_kfp_hes_prs, v_NC_kfp_hes, D_TEST_Y, ALPHA_CVB) # Eq. 18
    s_kfp['covrg'][I_FIM, i_s], s_kfp['covrg_labels'][I_FIM, i_s, :] = kfp_covrg(m_NC_kfp_fim_prs, v_NC_kfp_fim, D_TEST_Y, ALPHA_CVB) # Eq. 18
    s_kfp['covrg'][I_LMC, i_s], s_kfp['covrg_labels'][I_LMC, i_s, :] = kfp_covrg(m_NC_kfp_lmc_prs, v_NC_kfp_lmc, D_TEST_Y, ALPHA_CVB) # Eq. 18
    s_kfp_giq['covrg'][I_ML, i_s], s_kfp_giq['covrg_labels'][I_ML, i_s, :] = kfp_covrg(m_NC_kfp_ml_prs_giq, v_NC_kfp_ml_giq, D_TEST_Y, ALPHA_CVB) # Eq. 18
    s_kfp_giq['covrg'][I_MAP, i_s], s_kfp_giq['covrg_labels'][I_MAP, i_s, :] = kfp_covrg(m_NC_kfp_map_prs_giq, v_NC_kfp_map_giq, D_TEST_Y, ALPHA_CVB) # Eq. 18
    s_kfp_giq['covrg'][I_HES, i_s], s_kfp_giq['covrg_labels'][I_HES, i_s, :] = kfp_covrg(m_NC_kfp_hes_prs_giq, v_NC_kfp_hes_giq, D_TEST_Y, ALPHA_CVB) # Eq. 18
    s_kfp_giq['covrg'][I_FIM, i_s], s_kfp_giq['covrg_labels'][I_FIM, i_s, :] = kfp_covrg(m_NC_kfp_fim_prs_giq, v_NC_kfp_fim_giq, D_TEST_Y, ALPHA_CVB) # Eq. 18
    s_kfp_giq['covrg'][I_LMC, i_s], s_kfp_giq['covrg_labels'][I_LMC, i_s, :] = kfp_covrg(m_NC_kfp_lmc_prs_giq, v_NC_kfp_lmc_giq, D_TEST_Y, ALPHA_CVB) # Eq. 18

# --- Fim do Loop de Simulações Independentes ---

# --- Salvar Resultados ---
import scipy.io # Importar scipy.io para savemat
file_str = 'simulation_results_fixed_N.mat' # Nome do arquivo de saída
scipy.io.savemat(PATH_OF_RUN + file_str, {
    "dSetting": {'snr_dB': FIXED_SNR_DB, 'modKey': FIXED_MOD_KEY},
    "N_fixed": N_FIXED, # Agora é um valor fixo
    "gamma": GAMMA_REG_COEFF,
    "num_sim": FIXED_NUM_SIM,
    "alg_str_names": ALG_STR_NAMES,
    "i_ml": I_ML, "i_map": I_MAP, "i_hes": I_HES, "i_fim": I_FIM, "i_lmc": I_LMC,
    "alpha": ALPHA, "alpha_cvb": ALPHA_CVB, "K_folds": K_FOLDS,
    "gd_num_iters": FIXED_GD_NUM_ITERS, "gd_lr": FIXED_GD_LR,
    "lmc_burn_in": FIXED_LMC_BURN_IN, "lmc_lr_init": FIXED_LMC_LR_INIT, "lmc_lr_decaying": FIXED_LMC_LR_DECAYING,
    "lmc_num_iters": LMC_NUM_ITERS,
    "ensemble_size": FIXED_ENSEMBLE_SIZE, "compute_hessian": FIXED_COMPUTE_HESSIAN,
    # Estruturas de resultados de cobertura (remova 'ineff' nas definições aqui se existirem)
    "s_vb": s_vb,
    "s_jkp": s_jkp,
    "s_kfp": s_kfp,
    "s_vb_giq": s_vb_giq,
    "s_jkp_giq": s_jkp_giq,
    "s_kfp_giq": s_kfp_giq,
    # Históricos de perda
    "m_loss_tr_vb_ml": M_LOSS_TR_VB_ML.numpy(), "m_loss_te_vb_ml": M_LOSS_TE_VB_ML.numpy(),
    "m_loss_tr_vb_map": M_LOSS_TR_VB_MAP.numpy(), "m_loss_te_vb_map": M_LOSS_TE_VB_MAP.numpy(),
    "m_loss_tr_jkp_ml": M_LOSS_TR_JKP_ML.numpy(), "m_loss_te_jkp_ml": M_LOSS_TE_JKP_ML.numpy(),
    "m_loss_tr_jkp_map": M_LOSS_TR_JKP_MAP.numpy(), "m_loss_te_jkp_map": M_LOSS_TE_JKP_MAP.numpy(),
    "m_loss_tr_kfp_ml": M_LOSS_TR_KFP_ML.numpy(), "m_loss_te_kfp_ml": M_LOSS_TE_KFP_ML.numpy(),
    "m_loss_tr_kfp_map": M_LOSS_TR_KFP_MAP.numpy(), "m_loss_te_kfp_map": M_LOSS_TE_KFP_MAP.numpy(),
    "m_loss_tr_vb_lmc": M_LOSS_TR_VB_LMC.numpy(), "m_loss_te_vb_lmc": M_LOSS_TE_VB_LMC.numpy(),
    "m_loss_tr_jkp_lmc": M_LOSS_TR_JKP_LMC.numpy(), "m_loss_te_jkp_lmc": M_LOSS_TE_JKP_LMC.numpy(),
    "m_loss_tr_kfp_lmc": M_LOSS_TR_KFP_LMC.numpy(), "m_loss_te_kfp_lmc": M_LOSS_TE_KFP_LMC.numpy(),
})
print(f'Results saved to: {PATH_OF_RUN + file_str}')

# --- Finalização ---
print('Simulation done. Total time:')
print(datetime.datetime.now() - start_time_total)

# Opcional: Liberação de memória (mais relevante para GPUs)
# if torch.cuda.is_available():
#     torch.cuda.empty_cache()
#     torch.cuda.ipc_collect()
# import gc
# gc.collect()