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

# Tham số FedLiteKb
INITIAL_RETENTION = 0.8  # Không dùng nếu cố định
FINAL_RETENTION = 0.8    # Không dùng nếu cố định
GROWTH_RATE = 0.0       # Không dùng nếu cố định
#retention_factor = initial_retention + (final_retention - initial_retention) * (1 - np.exp(-growth_rate * round_num))
#knowledge_bank[name] = retention_factor * knowledge_bank[name] + (1 - retention_factor) * sparse_delta_avg[name]
K_PERCENT = 0.2

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