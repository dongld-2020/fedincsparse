# client.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import socket
import io
import pickle
from src.model import LeNet5, ResNet18Fashion, ResNet32
from .config import GLOBAL_SEED, LEARNING_RATE, BATCH_SIZE, LOCAL_EPOCHS, SERVER_PORT, MU, BUFFER_SIZE, setup_logger
from .config import K_PERCENT, ADAPTOPK_GAMMA, ADAPTOPK_BASE_K, ADAPTOPK_T_HAT
from .config import FEDZIP_SPARSITY, FEDZIP_NUM_CLUSTERS, FEDZIP_ENCODING_METHOD
from src.config import DEVICE

def calculate_adaptive_topk_sparsity(round_num, total_rounds, client_id):
    """Calculate adaptive Top-K sparsity level based on AdapTop-K algorithm"""
    if ADAPTOPK_T_HAT is None:
        # Calculate t_hat based on total rounds if not provided
        t_hat = total_rounds // 2
    else:
        t_hat = ADAPTOPK_T_HAT
    
    base_k = ADAPTOPK_BASE_K
    gamma = ADAPTOPK_GAMMA
    
    # Apply AdapTop-K strategy
    if (round_num < t_hat // 2) or (round_num >= (t_hat + total_rounds) // 2):
        # Higher sparsity in early and late stages
        k_percent = base_k * (1 + gamma)
    else:
        # Lower sparsity in middle stages
        k_percent = base_k * (1 - gamma)
    
    # Ensure k_percent is within valid range [0, 1]
    k_percent = max(0.01, min(0.99, k_percent))
    
    return k_percent

def fedzip_compress(delta_weights, sparsity=FEDZIP_SPARSITY, num_clusters=FEDZIP_NUM_CLUSTERS, 
                   encoding_method=FEDZIP_ENCODING_METHOD):
    """
    FedZip compression: Top-z sparsification + k-means quantization + efficient encoding
    """
    from sklearn.cluster import KMeans
    import heapq
    
    compressed_data = {}
    
    for name, delta in delta_weights.items():
        # Step 1: Top-z sparsification
        flat_delta = delta.flatten().cpu().numpy()
        num_params = len(flat_delta)
        k = max(1, int(sparsity * num_params))
        
        # Get top-k values by absolute magnitude
        if k < num_params:
            # Get indices of top-k values by absolute magnitude
            topk_indices = np.argpartition(np.abs(flat_delta), -k)[-k:]
            # Create sparse vector
            sparse_values = flat_delta[topk_indices]
        else:
            topk_indices = np.arange(num_params)
            sparse_values = flat_delta
        
        # Step 2: K-means quantization
        if len(sparse_values) > num_clusters:
            kmeans = KMeans(n_clusters=num_clusters, random_state=GLOBAL_SEED, n_init=10)
            kmeans.fit(sparse_values.reshape(-1, 1))
            quantized_values = kmeans.cluster_centers_[kmeans.labels_].flatten()
            centroids = kmeans.cluster_centers_.flatten()
        else:
            quantized_values = sparse_values
            centroids = np.unique(sparse_values)
        
        # Step 3: Encoding
        if encoding_method == 'huffman':
            # Simple frequency-based encoding (simplified Huffman)
            unique_vals, counts = np.unique(quantized_values, return_counts=True)
            encoding_map = {val: i for i, val in enumerate(unique_vals)}
            encoded_values = [encoding_map[val] for val in quantized_values]
            
            compressed_data[name] = {
                'method': 'huffman',
                'indices': topk_indices,
                'encoded_values': encoded_values,
                'centroids': centroids,
                'encoding_map': encoding_map,
                'shape': delta.shape
            }
            
        elif encoding_method == 'position':
            # Position encoding
            compressed_data[name] = {
                'method': 'position',
                'indices': topk_indices,
                'values': quantized_values,
                'centroids': centroids,
                'shape': delta.shape
            }
            
        elif encoding_method == 'difference':
            # Difference of address positions (most efficient)
            if len(topk_indices) > 1:
                diff_indices = np.diff(topk_indices)
                compressed_data[name] = {
                    'method': 'difference',
                    'first_index': topk_indices[0],
                    'diff_indices': diff_indices,
                    'values': quantized_values,
                    'centroids': centroids,
                    'shape': delta.shape
                }
            else:
                compressed_data[name] = {
                    'method': 'difference',
                    'first_index': topk_indices[0] if len(topk_indices) > 0 else 0,
                    'diff_indices': np.array([]),
                    'values': quantized_values,
                    'centroids': centroids,
                    'shape': delta.shape
                }
    
    return compressed_data

def fedzip_decompress(compressed_data, original_shape=None):
    """Decompress FedZip compressed data"""
    decompressed_weights = {}
    
    for name, data in compressed_data.items():
        shape = data['shape']
        total_params = np.prod(shape)
        flat_array = np.zeros(total_params)
        
        if data['method'] == 'huffman':
            indices = data['indices']
            encoded_values = data['encoded_values']
            centroids = data['centroids']
            encoding_map = data['encoding_map']
            
            # Reverse encoding map
            decoding_map = {v: k for k, v in encoding_map.items()}
            values = [decoding_map[code] for code in encoded_values]
            values = [centroids[val] if val < len(centroids) else 0 for val in values]
            
            flat_array[indices] = values
            
        elif data['method'] == 'position':
            indices = data['indices']
            values = data['values']
            flat_array[indices] = values
            
        elif data['method'] == 'difference':
            first_index = data['first_index']
            diff_indices = data['diff_indices']
            values = data['values']
            
            if len(diff_indices) > 0:
                indices = np.cumsum(np.concatenate([[first_index], diff_indices]))
            else:
                indices = np.array([first_index]) if first_index < total_params else np.array([])
            
            if len(indices) > 0:
                flat_array[indices] = values
        
        decompressed_weights[name] = torch.from_numpy(flat_array.reshape(shape)).to(DEVICE)
    
    return decompressed_weights

def start_client(client_id, seed, dataset, global_model=None, algorithm='fedavg', global_control=None, model_name='lenet5', current_round=0, total_rounds=100):
    logger = setup_logger(f'client_{client_id}', f'client_{client_id}.log')
    logger.info(f"Client {client_id} started with algorithm: {algorithm}, model: {model_name}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('localhost', SERVER_PORT))

    size_data = client.recv(8)
    expected_size = int.from_bytes(size_data, 'big')
    data = b""
    while len(data) < expected_size:
        packet = client.recv(BUFFER_SIZE)
        if not packet:
            break
        data += packet

    if len(data) != expected_size:
        logger.error(f"Data incomplete: expected {expected_size}, got {len(data)}")
        client.close()
        return

    received_data = pickle.load(io.BytesIO(data))
    global_model_state = received_data['global_model']
    global_control = received_data['global_control']
    model_name = received_data.get('model_name', 'lenet5')
    current_round = received_data.get('current_round', 0)
    total_rounds = received_data.get('total_rounds', 100)

    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize the local model based on model_name
    if model_name == 'lenet5':
        local_model = LeNet5()
    elif model_name == 'resnet18fashion':        
        local_model = ResNet18Fashion()
    elif model_name == 'resnet32':        
        local_model = ResNet32()
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    local_model.load_state_dict(global_model_state)
    # Use Adam for DermaMNISTNet due to no BatchNorm; SGD for others
    if model_name == 'dermamnistnet':
        optimizer = optim.Adam(local_model.parameters(), lr=0.0005)
    else:
        optimizer = optim.SGD(local_model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-3)
    local_model = local_model.to(DEVICE)
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    data_size = len(dataset)
    global_params = {name: param.clone().detach() for name, param in local_model.named_parameters()}
    initial_params = {name: param.clone().detach() for name, param in local_model.named_parameters()}

    if algorithm.lower() == 'scaffold':
        if global_control is None:
            global_control = {name: torch.zeros_like(param) for name, param in local_model.named_parameters()}
        local_control = {name: param.clone().detach() for name, param in global_control.items()}
    else:
        local_control = None

    local_model.train()
    if algorithm.lower() == 'scaffold':
        num_steps = 0

    prev_loss = float('inf')
    for epoch in range(LOCAL_EPOCHS):
        epoch_loss = 0.0
        num_batches = 0

        for data, target in data_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            if data.size(0) == 0 or target.size(0) == 0:
                logger.warning(f"Empty batch encountered in epoch {epoch + 1}, skipping...")
                continue

            optimizer.zero_grad()
            output = local_model(data)
            if target.dim() > 1:
                target = target.squeeze()
            if target.dim() == 0:
                target = target.unsqueeze(0)
            loss = criterion(output, target.long())

            if algorithm.lower() == 'fedprox':
                proximal_term = 0
                for name, param in local_model.named_parameters():
                    proximal_term += (MU / 2) * torch.norm(param - global_params[name])**2
                loss += proximal_term

            elif algorithm.lower() == 'scaffold' and local_control is not None:
                for name, param in local_model.named_parameters():
                    if name in global_control and name in local_control:
                        correction = global_control[name] - local_control[name]
                        param.grad = (param.grad if param.grad is not None else torch.zeros_like(param)) + correction

            loss.backward()
            optimizer.step()

            if algorithm.lower() == 'scaffold':
                num_steps += 1

            epoch_loss += loss.item()
            num_batches += 1

        if num_batches == 0:
            logger.warning(f"No valid batches in epoch {epoch + 1}, skipping epoch...")
            continue

        epoch_loss /= num_batches
        logger.info(f"Epoch {epoch + 1}/{LOCAL_EPOCHS} - Loss: {epoch_loss:.4f}")

        if epoch > 0 and epoch_loss > prev_loss:
            logger.info(f"Early stopping triggered at epoch {epoch + 1}: Loss increased from {prev_loss:.4f} to {epoch_loss:.4f}")
            break
        prev_loss = epoch_loss

    delta_c = None
    if algorithm.lower() == 'scaffold' and num_steps > 0:
        delta_c = {}
        final_params = {name: param.clone().detach() for name, param in local_model.named_parameters()}
        for name in local_control:
            delta_x = (final_params[name] - initial_params[name]) / (num_steps * LEARNING_RATE)
            local_control[name] = local_control[name] - global_control[name] + delta_x
            delta_c[name] = local_control[name] - global_control[name]

    # Compression methods
    delta_weights = {}
    for name, param in local_model.named_parameters():
        delta_weights[name] = param.data - global_params[name]
    
    compressed_weights = None
    
    if algorithm.lower() == 'fedzip':
        # Apply FedZip compression
        compressed_weights = fedzip_compress(delta_weights)
        logger.info(f"Client {client_id} using FedZip compression with sparsity={FEDZIP_SPARSITY}, clusters={FEDZIP_NUM_CLUSTERS}, encoding={FEDZIP_ENCODING_METHOD}")
    
    elif algorithm.lower() in ['fedsparse', 'adaptopk']:
        # Adaptive Top-K sparsification
        delta_weights = {}
        for name, param in local_model.named_parameters():
            delta_weights[name] = param.data - global_params[name]
        
        sparse_delta = {}
        
        if algorithm.lower() == 'adaptopk':
            # Use adaptive Top-K sparsity
            k_percent = calculate_adaptive_topk_sparsity(current_round, total_rounds, client_id)
            logger.info(f"Client {client_id} using AdaptiveTop-K with k_percent={k_percent:.4f} in round {current_round}")
        else:
            # Use fixed FedSparse sparsity
            k_percent = K_PERCENT
        
        for name, delta in delta_weights.items():
            flat_delta = delta.flatten()
            num_params = flat_delta.numel()
            k = int(k_percent * num_params)
            if k > 0:
                values, indices = torch.topk(torch.abs(flat_delta), k)
                sparse_delta[name] = {
                    'values': flat_delta[indices],
                    'indices': indices,
                    'shape': delta.shape
                }
            else:
                sparse_delta[name] = {
                    'values': torch.tensor([], dtype=torch.float32, device=DEVICE),
                    'indices': torch.tensor([], dtype=torch.long, device=DEVICE),
                    'shape': delta.shape
                }
        compressed_weights = sparse_delta
        logger.info(f"Client {client_id} sending sparse_delta with k_percent={k_percent}")

    buffer = io.BytesIO()
    data_to_send = {
        'client_id': client_id,
        'weights': local_model.state_dict() if algorithm.lower() not in ['fedsparse', 'adaptopk', 'fedzip'] else compressed_weights,
        'data_size': algorithm.lower() in ['fedavg', 'fedzip'],
        'delta_c': delta_c if algorithm.lower() == 'scaffold' else None,
        'is_sparse': algorithm.lower() in ['fedsparse', 'adaptopk', 'fedzip'],
        'algorithm': algorithm.lower(),
        'compression_method': 'fedzip' if algorithm.lower() == 'fedzip' else None
    }
    pickle.dump(data_to_send, buffer)
    buffer.seek(0)
    data = buffer.read()
    
    data_size_bytes = len(data)
    client.send(data_size_bytes.to_bytes(8, 'big'))
    client.sendall(data)
    
    weights_size = len(pickle.dumps(data_to_send['weights']))
    logger.info(f"Sent {data_size_bytes} bytes to server (weights size: {weights_size} bytes, client data size: {data_size}, sparse: {algorithm.lower() in ['fedsparse', 'adaptopk', 'fedzip']})")

    response = client.recv(4)
    if response == b"ACK":
        logger.info("Data transmission successful")
    else:
        logger.error("Data transmission failed")
    
    client.close()