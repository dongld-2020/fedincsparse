import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from .config import GLOBAL_SEED, ALPHA, BATCH_SIZE, DEVICE

def non_iid_partition_dirichlet(dataset, num_clients, partition="hetero", alpha=ALPHA, seed=GLOBAL_SEED):
    np.random.seed(seed)
    if hasattr(dataset, 'targets'):
        y_train = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        y_train = np.array(dataset.labels).flatten()
    else:
        raise ValueError("Dataset must have either 'targets' or 'labels' attribute")
    
    N = y_train.shape[0]
    if partition == "homo":
        idxs = np.random.permutation(N)
        batch_idxs = np.array_split(idxs, num_clients)
        clients_data = [Subset(dataset, batch_idxs[i]) for i in range(num_clients)]
        proportions = [1.0 / num_clients] * num_clients
    elif partition == "hetero":
        K = len(np.unique(y_train))
        min_size = 0
        attempts = 0
        max_attempts = 100
        while min_size < 10 and attempts < max_attempts:
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)]).clip(min=0)
                proportions = proportions / proportions.sum()
                split_points = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, split_points))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
            attempts += 1
        if min_size < 10:
            raise ValueError(f"Cannot partition data with alpha={alpha} and {num_clients} clients: minimum size {min_size} < 100")
        clients_data = [Subset(dataset, idx_batch[i]) for i in range(num_clients)]
        proportions = [len(client_data) / N for client_data in idx_batch]
    return clients_data, proportions

def evaluate_global_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)  # Move to correct device
            #print(f"Eval - Data shape: {data.shape}, Target shape: {target.shape}")
            output = model(data)
            #print(f"Eval - Output shape: {output.shape}")
            target = target.squeeze()
            if target.dim() == 0:
                target = target.unsqueeze(0)
            #print(f"Eval - Adjusted Target shape: {target.shape}, Unique classes: {torch.unique(target)}")
            loss += criterion(output, target.long()).item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target.long()).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.long().cpu().numpy())

    accuracy = 100 * correct / total
    avg_loss = loss / len(test_loader)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1) * 100

    metrics = {
        'accuracy': accuracy,
        'loss': avg_loss,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'per_class_accuracy': per_class_accuracy,
        'confusion_matrix': cm
    }
    return metrics