import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
from torch.utils.data import ConcatDataset
from medmnist import OrganAMNIST, BloodMNIST
import random

# Constants
GLOBAL_SEED = 42
ALPHA_MNIST = 0.05
ALPHA_ORGAN = 0.1
ALPHA_BLOOD = 0.2
ALPHA_CIFAR10 = 0.2
NUM_CLIENTS = 50

# Set seeds and limit thread usage
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
torch.backends.cudnn.deterministic = True

# === Keep this function unchanged ===
def non_iid_partition_dirichlet(dataset, num_clients, partition="hetero", alpha=ALPHA_MNIST, seed=GLOBAL_SEED):
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

# Load MNIST
def load_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    return dataset

# Load BloodMNIST
def load_bloodmnist_combined():
    transform = transforms.Compose([
        transforms.RandomCrop(28, padding=1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    train_split = BloodMNIST(split='train', root='./data', download=True, transform=transform)
    test_split = BloodMNIST(split='test', root='./data', download=True, transform=transform)
    
    combined_dataset = ConcatDataset([train_split, test_split])
    
    combined_labels = np.concatenate([train_split.labels, test_split.labels], axis=0).flatten()
    combined_dataset.targets = combined_labels
    return combined_dataset

# Load OrganAMNIST
def load_organamnist_combined():
    transform = transforms.Compose([
        transforms.RandomCrop(28, padding=1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    train_split = OrganAMNIST(split='train', root='./data', download=True, transform=transform)
    test_split = OrganAMNIST(split='test', root='./data', download=True, transform=transform)
    
    combined_dataset = ConcatDataset([train_split, test_split])
    
    combined_labels = np.concatenate([train_split.labels, test_split.labels], axis=0).flatten()
    combined_dataset.targets = combined_labels
    return combined_dataset

# Load CIFAR-10
def load_cifar10():
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    return dataset

# Count class distribution per client
def get_class_counts(dataset, client_data):
    num_classes = len(np.unique(np.array(dataset.targets)))
    return [np.bincount(np.array(dataset.targets)[subset.indices], minlength=num_classes) for subset in client_data]

# Prepare data
mnist_data = load_mnist()
blood_data = load_bloodmnist_combined()
organ_data = load_organamnist_combined()
cifar10_data = load_cifar10()

mnist_clients, _ = non_iid_partition_dirichlet(mnist_data, NUM_CLIENTS, "hetero", alpha=ALPHA_MNIST)
blood_clients, _ = non_iid_partition_dirichlet(blood_data, NUM_CLIENTS, "hetero", alpha=ALPHA_BLOOD)
organ_clients, _ = non_iid_partition_dirichlet(organ_data, NUM_CLIENTS, "hetero", alpha=ALPHA_ORGAN)
cifar10_clients, _ = non_iid_partition_dirichlet(cifar10_data, NUM_CLIENTS, "hetero", alpha=ALPHA_CIFAR10)

mnist_class_counts = get_class_counts(mnist_data, mnist_clients)
blood_class_counts = get_class_counts(blood_data, blood_clients)
organ_class_counts = get_class_counts(organ_data, organ_clients)
cifar10_class_counts = get_class_counts(cifar10_data, cifar10_clients)

# Plot MNIST
plt.figure(figsize=(20, 6))
x = np.arange(NUM_CLIENTS)
mnist_colors = plt.cm.tab10(np.linspace(0, 1, 10))
for i in range(NUM_CLIENTS):
    bottom = 0
    for class_idx, count in enumerate(mnist_class_counts[i]):
        if count > 0:
            plt.bar(x[i], count, bottom=bottom, color=mnist_colors[class_idx], width=0.8)
            bottom += count
mnist_legend = [plt.Line2D([0], [0], color=mnist_colors[i], lw=5, label=f'Class {i}') for i in range(10)]
plt.legend(handles=mnist_legend, bbox_to_anchor=(0.9, 1), loc='upper left', fontsize=15)
plt.title("Client-wise Class Distribution - MNIST", fontsize=30)
plt.xlabel("Client ID", fontsize=25)
plt.ylabel("Number of Samples", fontsize=25)
plt.xticks(np.arange(1, NUM_CLIENTS, step=2), fontsize=25)
plt.yticks(fontsize=25)
plt.tight_layout()
plt.show()

# Plot BloodMNIST
plt.figure(figsize=(20, 6))
x = np.arange(NUM_CLIENTS)
blood_colors = plt.cm.tab10(np.linspace(0, 1, 9))
for i in range(NUM_CLIENTS):
    bottom = 0
    for class_idx, count in enumerate(blood_class_counts[i]):
        if count > 0:
            plt.bar(x[i], count, bottom=bottom, color=blood_colors[class_idx], width=0.8)
            bottom += count
blood_legend = [plt.Line2D([0], [0], color=blood_colors[i], lw=5, label=f'Class {i}') for i in range(9)]
plt.legend(handles=blood_legend, bbox_to_anchor=(0.9, 1), loc='upper left', fontsize=15)
plt.title("Client-wise Class Distribution - BloodMNIST", fontsize=30)
plt.xlabel("Client ID", fontsize=25)
plt.ylabel("Number of Samples", fontsize=25)
plt.xticks(np.arange(1, NUM_CLIENTS+1, step=2), fontsize=25)
plt.yticks(fontsize=25)
plt.tight_layout()
plt.show()

# Plot OrganAMNIST
plt.figure(figsize=(20, 6))
x = np.arange(NUM_CLIENTS)
organ_colors = plt.cm.tab10(np.linspace(0, 1, 11))
for i in range(NUM_CLIENTS):
    bottom = 0
    for class_idx, count in enumerate(organ_class_counts[i]):
        if count > 0:
            plt.bar(x[i], count, bottom=bottom, color=organ_colors[class_idx], width=0.8)
            bottom += count
organ_legend = [plt.Line2D([0], [0], color=organ_colors[i], lw=5, label=f'Class {i}') for i in range(11)]
plt.legend(handles=organ_legend, bbox_to_anchor=(0.9, 1), loc='upper left', fontsize=15)
plt.title("Client-wise Class Distribution - OrganAMNIST", fontsize=30)
plt.xlabel("Client ID", fontsize=25)
plt.ylabel("Number of Samples", fontsize=25)
plt.xticks(np.arange(1, NUM_CLIENTS+1, step=2), fontsize=25)
plt.yticks(fontsize=25)
plt.tight_layout()
plt.show()

# Plot CIFAR-10
plt.figure(figsize=(20, 6))
x = np.arange(NUM_CLIENTS)
cifar10_colors = plt.cm.tab10(np.linspace(0, 1, 10))
for i in range(NUM_CLIENTS):
    bottom = 0
    for class_idx, count in enumerate(cifar10_class_counts[i]):
        if count > 0:
            plt.bar(x[i], count, bottom=bottom, color=cifar10_colors[class_idx], width=0.8)
            bottom += count
cifar10_legend = [plt.Line2D([0], [0], color=cifar10_colors[i], lw=5, label=f'Class {i}') for i in range(10)]
plt.legend(handles=cifar10_legend, bbox_to_anchor=(0.9, 1), loc='upper left', fontsize=15)
plt.title("Client-wise Class Distribution - CIFAR-10", fontsize=30)
plt.xlabel("Client ID", fontsize=25)
plt.ylabel("Number of Samples", fontsize=25)
plt.xticks(np.arange(1, NUM_CLIENTS+1, step=2), fontsize=25)
plt.yticks(fontsize=25)
plt.tight_layout()
plt.show()

# === Statistics for MNIST, BloodMNIST, OrganAMNIST, and CIFAR-10 ===

# Number of classes per client
client_class_counts_mnist = [sum(1 for count in counts if count > 0) for counts in mnist_class_counts]
client_class_counts_blood = [sum(1 for count in counts if count > 0) for counts in blood_class_counts]
client_class_counts_organ = [sum(1 for count in counts if count > 0) for counts in organ_class_counts]
client_class_counts_cifar10 = [sum(1 for count in counts if count > 0) for counts in cifar10_class_counts]

# Number of samples per client
client_sample_counts_mnist = [len(subset) for subset in mnist_clients]
client_sample_counts_blood = [len(subset) for subset in blood_clients]
client_sample_counts_organ = [len(subset) for subset in organ_clients]
client_sample_counts_cifar10 = [len(subset) for subset in cifar10_clients]

# Find clients with min/max samples and classes
min_samples_client_mnist = np.argmin(client_sample_counts_mnist)
max_samples_client_mnist = np.argmax(client_sample_counts_mnist)
min_samples_client_blood = np.argmin(client_sample_counts_blood)
max_samples_client_blood = np.argmax(client_sample_counts_blood)
min_samples_client_organ = np.argmin(client_sample_counts_organ)
max_samples_client_organ = np.argmax(client_sample_counts_organ)
min_samples_client_cifar10 = np.argmin(client_sample_counts_cifar10)
max_samples_client_cifar10 = np.argmax(client_sample_counts_cifar10)

min_classes_client_mnist = np.argmin(client_class_counts_mnist)
max_classes_client_mnist = np.argmax(client_class_counts_mnist)
min_classes_client_blood = np.argmin(client_class_counts_blood)
max_classes_client_blood = np.argmax(client_class_counts_blood)
min_classes_client_organ = np.argmin(client_class_counts_organ)
max_classes_client_organ = np.argmax(client_class_counts_organ)
min_classes_client_cifar10 = np.argmin(client_class_counts_cifar10)
max_classes_client_cifar10 = np.argmax(client_class_counts_cifar10)

print("\nMNIST Statistics:")
print(f"Clients with the most classes ({client_class_counts_mnist[max_classes_client_mnist]}): Client {max_classes_client_mnist}")
print(f"Clients with the fewest classes ({client_class_counts_mnist[min_classes_client_mnist]}): Client {min_classes_client_mnist}")
print(f"Client with most samples ({client_sample_counts_mnist[max_samples_client_mnist]}): Client {max_samples_client_mnist}")
print(f"Client with fewest samples ({client_sample_counts_mnist[min_samples_client_mnist]}): Client {min_samples_client_mnist}")

print("\nBloodMNIST Statistics:")
print(f"Clients with the most classes ({client_class_counts_blood[max_classes_client_blood]}): Client {max_classes_client_blood}")
print(f"Clients with the fewest classes ({client_class_counts_blood[min_classes_client_blood]}): Client {min_classes_client_blood}")
print(f"Client with most samples ({client_sample_counts_blood[max_samples_client_blood]}): Client {max_samples_client_blood}")
print(f"Client with fewest samples ({client_sample_counts_blood[min_samples_client_blood]}): Client {min_samples_client_blood}")

print("\nOrganAMNIST Statistics:")
print(f"Clients with the most classes ({client_class_counts_organ[max_classes_client_organ]}): Client {max_classes_client_organ}")
print(f"Clients with the fewest classes ({client_class_counts_organ[min_classes_client_organ]}): Client {min_classes_client_organ}")
print(f"Client with most samples ({client_sample_counts_organ[max_samples_client_organ]}): Client {max_samples_client_organ}")
print(f"Client with fewest samples ({client_sample_counts_organ[min_samples_client_organ]}): Client {min_samples_client_organ}")

print("\nCIFAR-10 Statistics:")
print(f"Clients with the most classes ({client_class_counts_cifar10[max_classes_client_cifar10]}): Client {max_classes_client_cifar10}")
print(f"Clients with the fewest classes ({client_class_counts_cifar10[min_classes_client_cifar10]}): Client {min_classes_client_cifar10}")
print(f"Client with most samples ({client_sample_counts_cifar10[max_samples_client_cifar10]}): Client {max_samples_client_cifar10}")
print(f"Client with fewest samples ({client_sample_counts_cifar10[min_samples_client_cifar10]}): Client {min_samples_client_cifar10}")

# Total samples
print(f"\nTotal samples in MNIST: {len(mnist_data)}")
print(f"Total samples in BloodMNIST: {len(blood_data)}")
print(f"Total samples in OrganAMNIST: {len(organ_data)}")
print(f"Total samples in CIFAR-10: {len(cifar10_data)}")

# Overall class distribution
print("\nClass distribution in MNIST:", np.bincount(np.array(mnist_data.targets)))
print("Class distribution in BloodMNIST:", np.bincount(np.array(blood_data.targets)))
print("Class distribution in OrganAMNIST:", np.bincount(np.array(organ_data.targets)))
print("Class distribution in CIFAR-10:", np.bincount(np.array(cifar10_data.targets)))

# Average/max/min statistics
print(f"\nMNIST - Avg samples/client: {np.mean(client_sample_counts_mnist):.2f}")
print(f"MNIST - Max samples: {np.max(client_sample_counts_mnist)}")
print(f"MNIST - Min samples: {np.min(client_sample_counts_mnist)}")
print(f"MNIST - Avg classes/client: {np.mean(client_class_counts_mnist):.2f}")
print(f"MNIST - Max classes: {np.max(client_class_counts_mnist)}")
print(f"MNIST - Min classes: {np.min(client_class_counts_mnist)}")

print(f"\nBloodMNIST - Avg samples/client: {np.mean(client_sample_counts_blood):.2f}")
print(f"BloodMNIST - Max samples: {np.max(client_sample_counts_blood)}")
print(f"BloodMNIST - Min samples: {np.min(client_sample_counts_blood)}")
print(f"BloodMNIST - Avg classes/client: {np.mean(client_class_counts_blood):.2f}")
print(f"BloodMNIST - Max classes: {np.max(client_class_counts_blood)}")
print(f"BloodMNIST - Min classes: {np.min(client_class_counts_blood)}")

print(f"\nOrganAMNIST - Avg samples/client: {np.mean(client_sample_counts_organ):.2f}")
print(f"OrganAMNIST - Max samples: {np.max(client_sample_counts_organ)}")
print(f"OrganAMNIST - Min samples: {np.min(client_sample_counts_organ)}")
print(f"OrganAMNIST - Avg classes/client: {np.mean(client_class_counts_organ):.2f}")
print(f"OrganAMNIST - Max classes: {np.max(client_class_counts_organ)}")
print(f"OrganAMNIST - Min classes: {np.min(client_class_counts_organ)}")

print(f"\nCIFAR-10 - Avg samples/client: {np.mean(client_sample_counts_cifar10):.2f}")
print(f"CIFAR-10 - Max samples: {np.max(client_sample_counts_cifar10)}")
print(f"CIFAR-10 - Min samples: {np.min(client_sample_counts_cifar10)}")
print(f"CIFAR-10 - Avg classes/client: {np.mean(client_class_counts_cifar10):.2f}")
print(f"CIFAR-10 - Max classes: {np.max(client_class_counts_cifar10)}")
print(f"CIFAR-10 - Min classes: {np.min(client_class_counts_cifar10)}")

# Gini coefficient (Non-IID)
def gini_coefficient(counts):
    counts = np.array(counts)
    sorted_counts = np.sort(counts)
    n = len(sorted_counts)
    cumsum = np.cumsum(sorted_counts)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

print(f"\nGini coefficient for MNIST class distribution: {gini_coefficient(np.bincount(np.array(mnist_data.targets))):.4f}")
print(f"Gini coefficient for BloodMNIST class distribution: {gini_coefficient(np.bincount(np.array(blood_data.targets))):.4f}")
print(f"Gini coefficient for OrganAMNIST class distribution: {gini_coefficient(np.bincount(np.array(organ_data.targets))):.4f}")
print(f"Gini coefficient for CIFAR-10 class distribution: {gini_coefficient(np.bincount(np.array(cifar10_data.targets))):.4f}")