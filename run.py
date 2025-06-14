import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import threading
import time
from src.model import LeNet5, ResNet18Fashion
from src.server import start_server
from src.client import start_client
from src.utils import non_iid_partition_dirichlet
from src.config import GLOBAL_SEED, NUM_ROUNDS, NUM_CLIENTS, DATA_DIR, BATCH_SIZE
from src.config import DEVICE

def load_dataset(dataset_name):
    if dataset_name.lower() == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = torchvision.datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
        
    elif dataset_name.lower() == 'fashion':
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        train_dataset = torchvision.datasets.FashionMNIST(root=DATA_DIR, train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.FashionMNIST(root=DATA_DIR, train=False, download=True, transform=transform_test)
        
    else:
        raise ValueError("Dataset must be 'mnist' or 'fashion'")

    return train_dataset, test_dataset


def run_server(global_model, selected_clients_list, algorithm, proportions, test_loader, global_control, model_name):
    print("Starting server...")
    global_control = start_server(global_model, selected_clients_list, algorithm=algorithm, proportions=proportions, test_loader=test_loader, global_control=global_control, model_name=model_name)
    print("Server finished.")
    return global_control

def run_clients(global_model, selected_clients, algorithm, client_datasets, global_control, model_name):
    client_threads = []
    for client_id in selected_clients:
        print(f"Starting client {client_id}...")
        seed = GLOBAL_SEED + int(client_id)
        t = threading.Thread(target=start_client, args=(client_id, seed, client_datasets[client_id], global_model, algorithm, global_control, model_name))
        client_threads.append(t)
        t.start()
    
    for t in client_threads:
        t.join()
    print("All clients for this round finished.")

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    
    while True:
        algorithm = input("Enter the federated learning algorithm (fedavg, fedprox, fedsparse): ").strip().lower()
        if algorithm in ['fedavg', 'fedprox', 'fedsparse']:
            break
        print("Invalid input! Please enter 'fedavg', 'fedprox', or 'fedsparse'.")

    while True:
        dataset_name = input("Enter the dataset (mnist, fashion): ").strip().lower()
        if dataset_name in ['mnist', 'fashion']:
            break
        print("Invalid input! Please enter 'mnist', or 'fashion'.")

    # Automatically select the model based on the dataset
    if dataset_name.lower() == 'mnist':
        model_name = 'lenet5'
    elif dataset_name.lower() == 'fashion':
        model_name = 'resnet18fashion'

    print(f"Running with algorithm: {algorithm}, dataset: {dataset_name}, model: {model_name}")

    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)

    train_dataset, test_dataset = load_dataset(dataset_name)
    client_datasets, proportions = non_iid_partition_dirichlet(train_dataset, NUM_CLIENTS, partition="hetero")
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model initialization
    if model_name == 'lenet5':
        global_model = LeNet5().to(DEVICE)
    elif model_name == 'resnet18fashion':
        global_model = ResNet18Fashion().to(DEVICE)        
    
    global_control = None

    selected_clients_list = []
    for round_num in range(NUM_ROUNDS):
        np.random.seed(GLOBAL_SEED + round_num)
        selected_clients = np.random.choice(NUM_CLIENTS, np.random.randint(5, 11), replace=False)
        selected_clients_list.append(selected_clients)

    server_thread = threading.Thread(target=run_server, args=(global_model, selected_clients_list, algorithm, proportions, test_loader, global_control, model_name))
    server_thread.daemon = True
    server_thread.start()

    time.sleep(2)

    for round_num in range(NUM_ROUNDS):
        print(f"\nStarting clients for round {round_num+1}")
        run_clients(global_model, selected_clients_list[round_num], algorithm, client_datasets, global_control, model_name)
        time.sleep(2)

    server_thread.join()
    print("Training completed.")