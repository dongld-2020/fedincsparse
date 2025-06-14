import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import socket
import io
import pickle
from src.model import LeNet5, ResNet18Fashion
from .config import GLOBAL_SEED, LEARNING_RATE, BATCH_SIZE, LOCAL_EPOCHS, SERVER_PORT, MU, BUFFER_SIZE, setup_logger
from .config import K_PERCENT
from src.config import DEVICE

def start_client(client_id, seed, dataset, global_model=None, algorithm='fedavg', global_control=None, model_name='lenet5'):
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

    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize the local model based on model_name
    if model_name == 'lenet5':
        local_model = LeNet5()
    elif model_name == 'resnet18fashion':        
        local_model = ResNet18Fashion()
        
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

    if algorithm.lower() == 'fedsparse':
        delta_weights = {}
        for name, param in local_model.named_parameters():
            delta_weights[name] = param.data - global_params[name]
        
        sparse_delta = {}
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
        logger.info(f"Client {client_id} sending sparse_delta with k_percent={k_percent}")

    buffer = io.BytesIO()
    data_to_send = {
        'client_id': client_id,
        'weights': local_model.state_dict() if algorithm.lower() != 'fedsparse' else sparse_delta,
        'data_size': data_size if algorithm.lower() == 'fedavg' else None,
        'delta_c': delta_c if algorithm.lower() == 'scaffold' else None,
        'is_sparse': algorithm.lower() == 'fedsparse'
    }
    pickle.dump(data_to_send, buffer)
    buffer.seek(0)
    data = buffer.read()
    
    data_size_bytes = len(data)
    client.send(data_size_bytes.to_bytes(8, 'big'))
    client.sendall(data)
    
    weights_size = len(pickle.dumps(data_to_send['weights']))
    logger.info(f"Sent {data_size_bytes} bytes to server (weights size: {weights_size} bytes, client data size: {data_size}, sparse: {algorithm.lower() == 'fedsparse'})")

    response = client.recv(4)
    if response == b"ACK":
        logger.info("Data transmission successful")
    else:
        logger.error("Data transmission failed")
    
    client.close()