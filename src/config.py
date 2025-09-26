#src/config.py
import logging
import os
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#DEVICE = torch.device("cpu")
# Tham số chung
GLOBAL_SEED = 42
NUM_CLIENTS = 50
NUM_ROUNDS = 100
#NUM_CLIENTS_PER_ROUND = 5
LOCAL_EPOCHS = 2

# Tham số dữ liệu
DATA_DIR = './data'
ALPHA = 0.3

# Tham số mạng
SERVER_PORT = 9999
BUFFER_SIZE = 4096

# Tham số mô hình
LEARNING_RATE = 0.01
BATCH_SIZE = 32

# Tham số FedProx
MU = 0.001

# Tham số FedSparse
K_PERCENT = 0.2

# Tham số AdaptiveTopK
ADAPTOPK_GAMMA = 0.5  # Scaling factor γ
ADAPTOPK_BASE_K = 0.1  # Base sparsity level k
ADAPTOPK_T_HAT = None  # Will be calculated based on rounds

# Tham số FedZip
FEDZIP_SPARSITY = 0.2  # Top-z sparsity (10% of weights kept)
FEDZIP_NUM_CLUSTERS = 3  # Number of clusters for k-means quantization
FEDZIP_ENCODING_METHOD = 'difference'  # Options: 'huffman', 'position', 'difference'

# Tham số logging
LOG_DIR = './logs'
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

def setup_logger(name, log_file):
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    if not logger.handlers:
        handler = logging.FileHandler(os.path.join(LOG_DIR, log_file))
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
    return logger