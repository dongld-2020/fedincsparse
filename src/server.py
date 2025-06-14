#scr/server.py
import torch
import socket
import threading
import io
import numpy as np
import time
import pickle
import pandas as pd
from .utils import evaluate_global_model
from .config import GLOBAL_SEED, NUM_ROUNDS, SERVER_PORT, BUFFER_SIZE, NUM_CLIENTS, LOCAL_EPOCHS, ALPHA, LEARNING_RATE, setup_logger, DEVICE
from .config import INITIAL_RETENTION, FINAL_RETENTION, GROWTH_RATE

def aggregate_with_fedsparse(client_weights, knowledge_bank, retention_factor):
    sparse_delta_avg = {}
    for name in client_weights[0].keys():
        weighted_sum = None
        for w in client_weights:
            sparse_data = w[name]
            flat_delta = torch.zeros(np.prod(sparse_data['shape']), device=DEVICE)  # Đặt trên DEVICE
            indices = sparse_data['indices'].long().to(DEVICE)  # Đảm bảo indices trên DEVICE
            flat_delta[indices] = sparse_data['values'].to(DEVICE)  # Đảm bảo values trên DEVICE
            delta = flat_delta.view(sparse_data['shape'])
            if weighted_sum is None:
                weighted_sum = torch.zeros_like(delta, device=DEVICE)  # Đặt trên DEVICE
            weighted_sum += delta
        sparse_delta_avg[name] = weighted_sum / len(client_weights)

    if knowledge_bank is None:
        knowledge_bank = {name: torch.zeros_like(delta, device=DEVICE) for name, delta in sparse_delta_avg.items()}
    
    for name in sparse_delta_avg.keys():
        if knowledge_bank[name].shape != sparse_delta_avg[name].shape:
            raise ValueError(f"Shape mismatch for {name}: knowledge_bank {knowledge_bank[name].shape} vs sparse_delta_avg {sparse_delta_avg[name].shape}")
        knowledge_bank[name] = retention_factor * knowledge_bank[name] + (1 - retention_factor) * sparse_delta_avg[name]
    
    return knowledge_bank

def start_server(global_model, selected_clients_list, algorithm='fedavg', proportions=None, num_rounds=NUM_ROUNDS, test_loader=None, global_seed=GLOBAL_SEED, global_control=None, model_name='lenet5'):
    logger = setup_logger('server', 'server.log')
    logger.info(f"Server started with algorithm: {algorithm}, model: {model_name}")

    np.random.seed(global_seed)
    torch.manual_seed(global_seed)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('localhost', SERVER_PORT))
    server.listen(15)
    print("Server started...")
    
    global_model = global_model.to(DEVICE)  # Đảm bảo global_model trên DEVICE

    if algorithm.lower() == 'scaffold':
        if global_control is None:
            global_control = {name: torch.zeros_like(param, device=DEVICE) for name, param in global_model.named_parameters()}
    else:
        global_control = None

    if algorithm.lower() == 'fedsparse':
        knowledge_bank = None
        initial_retention = INITIAL_RETENTION
        final_retention = FINAL_RETENTION
        growth_rate = GROWTH_RATE

    all_metrics = []
    total_communication_cost = 0

    for round_num in range(num_rounds):
        print(f"\n--- Round {round_num + 1}/{num_rounds} ---")
        logger.info(f"Starting round {round_num + 1}")
        client_weights = []
        client_data_sizes = []
        client_delta_cs = []
        lock = threading.Lock()

        selected_clients = selected_clients_list[round_num]
        num_selected = len(selected_clients)
        print(f"Round {round_num + 1}: Selected {num_selected} clients: {selected_clients.tolist()}")
        logger.info(f"Round {round_num + 1}: Selected {num_selected} clients: {selected_clients.tolist()}")

        data_to_send = {
            'global_model': global_model.state_dict(),
            'global_control': global_control if algorithm.lower() == 'scaffold' else None,
            'model_name': model_name
        }
        buffer = io.BytesIO()
        pickle.dump(data_to_send, buffer)
        buffer.seek(0)
        data = buffer.read()
        data_size_to_clients = len(data)

        server_to_client_cost = data_size_to_clients * len(selected_clients)
        total_communication_cost += server_to_client_cost
        logger.info(f"Round {round_num + 1}: Server to clients communication cost: {server_to_client_cost} bytes")

        client_sockets = []
        for _ in range(len(selected_clients)):
            client_socket, addr = server.accept()
            client_sockets.append(client_socket)
            threading.Thread(target=lambda sock, data: (sock.send(len(data).to_bytes(8, 'big')), sock.sendall(data)), args=(client_socket, data)).start()

        def handle_client(client_socket):
            try:
                size_data = client_socket.recv(8)
                if not size_data:
                    logger.error(f"Client {client_socket.getpeername()} disconnected early")
                    client_socket.close()
                    return
                expected_size = int.from_bytes(size_data, 'big')
                
                data = b""
                while len(data) < expected_size:
                    packet = client_socket.recv(BUFFER_SIZE)
                    if not packet:
                        break
                    data += packet
                
                if len(data) != expected_size:
                    logger.error(f"Data incomplete from {client_socket.getpeername()}: expected {expected_size}, got {len(data)}")
                    client_socket.send(b"FAIL")
                    client_socket.close()
                    return
                
                nonlocal total_communication_cost
                client_to_server_cost = len(data)
                with lock:
                    total_communication_cost += client_to_server_cost

                received_data = pickle.load(io.BytesIO(data))
                client_id = received_data['client_id']
                weights = received_data['weights']
                data_size = received_data['data_size']
                delta_c = received_data.get('delta_c', None)
                is_sparse = received_data.get('is_sparse', False)

                with lock:
                    client_weights.append(weights)
                    client_data_sizes.append(data_size)
                    if delta_c is not None:
                        client_delta_cs.append(delta_c)

                weights_size = len(pickle.dumps(weights))
                print(f"Connection from ('127.0.0.1', {client_socket.getpeername()[1]}) - Client ID: {client_id}")
                logger.info(f"Received data from Client {client_id}, size: {client_to_server_cost} bytes, weights size: {weights_size} bytes, client data size: {data_size}, is_sparse: {is_sparse}")
                
                client_socket.send(b"ACK")
            except Exception as e:
                logger.error(f"Error handling client {client_socket.getpeername()}: {str(e)}")
                client_socket.send(b"FAIL")
            finally:
                client_socket.close()

        for client_socket in client_sockets:
            threading.Thread(target=handle_client, args=(client_socket,)).start()

        while len(client_weights) < len(selected_clients):
            time.sleep(0.1)

        averaged_weights = {}
        num_selected_clients = len(client_weights)

        try:
            if algorithm.lower() == 'fedavg':
                total_data_size = sum(client_data_sizes)
                weights_per_client = [data_size / total_data_size for data_size in client_data_sizes]
                selected_proportions = [proportions[i] for i in selected_clients]
                logger.info(f"Aggregating with weights: {weights_per_client} (from client data sizes)")
                
                averaged_weights = {}
                for name in client_weights[0].keys():
                    weighted_sum = torch.zeros_like(client_weights[0][name], device=DEVICE)  # Đặt trên DEVICE
                    for w, weight in zip(client_weights, weights_per_client):
                        weighted_sum += w[name].to(DEVICE) * weight  # Đảm bảo trên DEVICE
                    averaged_weights[name] = weighted_sum
                
                current_global = global_model.state_dict()
                current_global.update(averaged_weights)

            elif algorithm.lower() == 'fedprox':
                averaged_weights = {}
                for name in client_weights[0].keys():
                    stacked_weights = torch.stack([w[name].float().to(DEVICE) for w in client_weights], dim=0)  # Đặt trên DEVICE
                    averaged_weights[name] = stacked_weights.mean(dim=0)
                current_global = global_model.state_dict()
                for name in current_global.keys():
                    if name in averaged_weights:
                        current_global[name] = averaged_weights[name].to(DEVICE)  # Đảm bảo trên DEVICE

            elif algorithm.lower() == 'scaffold':
                current_global = global_model.state_dict()
                global_state = current_global.copy()
                delta_x = {}
                
                for name in client_weights[0].keys():
                    delta_y_sum = torch.zeros_like(global_state[name], device=DEVICE)  # Đặt trên DEVICE
                    for w in client_weights:
                        delta_y = w[name].to(DEVICE) - global_state[name].to(DEVICE)  # Đảm bảo trên DEVICE
                        delta_y_sum += delta_y
                    delta_x[name] = delta_y_sum / num_selected_clients
                    global_state[name] = global_state[name].to(DEVICE) + delta_x[name]  # Đảm bảo trên DEVICE
                
                current_global = global_state
                averaged_weights = global_state
                if client_delta_cs:
                    for name in global_control:
                        delta_c_avg = sum(dc[name].to(DEVICE) for dc in client_delta_cs) / num_selected_clients  # Đảm bảo trên DEVICE
                        global_control[name] = (global_control[name].to(DEVICE) + (num_selected_clients / NUM_CLIENTS) * delta_c_avg).to(DEVICE)  # Đảm bảo trên DEVICE

            elif algorithm.lower() == 'fedsparse':
                retention_factor = initial_retention + (final_retention - initial_retention) * (1 - np.exp(-growth_rate * round_num))
                logger.info(f"Round {round_num + 1} - Retention Factor: {retention_factor:.4f}")
                knowledge_bank = aggregate_with_fedsparse(client_weights, knowledge_bank, retention_factor)
                current_global = global_model.state_dict()
                for name in current_global.keys():
                    if name in knowledge_bank:
                        if current_global[name].shape != knowledge_bank[name].shape:
                            raise ValueError(f"Shape mismatch: {name} - global {current_global[name].shape} vs knowledge {knowledge_bank[name].shape}")
                        current_global[name] = (current_global[name].to(DEVICE) + knowledge_bank[name].to(DEVICE)).to(DEVICE)  # Đảm bảo trên DEVICE
                
                print("Loading updated state_dict into global_model...")
                global_model.load_state_dict(current_global)
                print("Evaluating global model...")
                metrics = evaluate_global_model(global_model, test_loader)

            global_model.load_state_dict(current_global)
            
            metrics = evaluate_global_model(global_model, test_loader)
            metrics['round'] = round_num + 1
            metrics['communication_cost'] = total_communication_cost

            print(f"Round {round_num + 1} completed. Global model updated.")
            print(f"Global model - Accuracy: {metrics['accuracy']:.2f}%, Loss: {metrics['loss']:.4f}")
            print(f"Precision (macro): {metrics['precision']:.4f}, Recall (macro): {metrics['recall']:.4f}, F1-Score (macro): {metrics['f1_score']:.4f}")
            print(f"Total Communication Cost: {total_communication_cost} bytes")
            print("Per-class Accuracy (%):")
            for i, acc in enumerate(metrics['per_class_accuracy']):
                print(f"  Class {i}: {acc:.2f}")
            print("Confusion Matrix:")
            print(metrics['confusion_matrix'])
            logger.info(f"Round {round_num + 1} completed - Accuracy: {metrics['accuracy']:.2f}%, Loss: {metrics['loss']:.4f}, Communication Cost: {total_communication_cost} bytes")

            row = {
                'Round': metrics['round'],
                'Accuracy': metrics['accuracy'],
                'Loss': metrics['loss'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'Communication_Cost': metrics['communication_cost'],
            }
            for i, acc in enumerate(metrics['per_class_accuracy']):
                row[f'Class_{i}_Accuracy'] = acc            
            row['Confusion_Matrix'] = str(metrics['confusion_matrix'].tolist())
            all_metrics.append(row)

        except Exception as e:
            logger.error(f"Error in round {round_num + 1}: {str(e)}")
            print(f"Error in round {round_num + 1}: {str(e)}")
            continue

    filename = f"results_{algorithm}_clients{NUM_CLIENTS}_rounds{NUM_ROUNDS}_epochs{LOCAL_EPOCHS}_alpha{ALPHA}_lr{LEARNING_RATE}_seed{GLOBAL_SEED}_{model_name}.csv"
    df = pd.DataFrame(all_metrics)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")
    logger.info(f"Results saved to {filename}")

    server.close()
    print("Server stopped.")
    logger.info("Server stopped")

    return global_control if algorithm.lower() == 'scaffold' else None