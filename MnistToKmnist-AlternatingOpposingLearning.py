# -*- coding: utf-8 -*-
"""
transfer_learning_experiment.py

A script to compare different pre-training strategies for transfer learning.

This script conducts a comparative experiment on the effects of pre-training
methods on a simple neural network. It compares three strategies for pre-training
on the MNIST dataset:
1.  Standard Training: Uses a conventional positive learning rate.
2.  Alternating Training: Alternates the learning rate between positive and
    negative at every step.
3.  Negative Training: Uses a consistently negative learning rate, effectively
    maximizing the loss.

After pre-training, each model is fine-tuned on the KMNIST dataset, and their
performance (training loss and validation accuracy) is plotted for comparison.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# --- 1. Define the Network Architecture ---
class MinimalNet(nn.Module):
    """A minimal fully-connected neural network for MNIST/KMNIST classification."""
    def __init__(self):
        super(MinimalNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Flatten the image
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# --- 2. Data Loading Function ---
def load_datasets(batch_size=64):
    """
    Downloads and prepares the MNIST and KMNIST datasets.

    Args:
        batch_size (int): The batch size for the data loaders.

    Returns:
        tuple: A tuple containing dataloaders for MNIST training, KMNIST training,
               and KMNIST testing.
    """
    print("Loading datasets...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # MNIST for pre-training
    mnist_trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    mnist_trainloader = DataLoader(
        mnist_trainset, batch_size=batch_size, shuffle=True
    )

    # KMNIST for fine-tuning and validation
    kmnist_trainset = torchvision.datasets.KMNIST(
        root='./data', train=True, download=True, transform=transform
    )
    kmnist_trainloader = DataLoader(
        kmnist_trainset, batch_size=batch_size, shuffle=True
    )

    kmnist_testset = torchvision.datasets.KMNIST(
        root='./data', train=False, download=True, transform=transform
    )
    kmnist_testloader = DataLoader(
        kmnist_testset, batch_size=1000, shuffle=False
    )
    print("Datasets loaded successfully.")
    return mnist_trainloader, kmnist_trainloader, kmnist_testloader


# --- 3. Core Experiment Function ---
def run_experiment(pretrain_method, loaders, epochs_config, lr_config):
    """
    Runs a full pre-training and fine-tuning experiment for a given method.

    Args:
        pretrain_method (str): The pre-training strategy ('standard', 'alternating', 'negative').
        loaders (tuple): Dataloaders from the load_datasets function.
        epochs_config (dict): Dictionary with 'pretrain' and 'finetune' epoch counts.
        lr_config (dict): Dictionary with the initial learning rate.

    Returns:
        tuple: A tuple containing the list of fine-tuning losses and accuracies.
    """
    mnist_loader, kmnist_loader, kmnist_test_loader = loaders
    pretrain_epochs = epochs_config['pretrain']
    finetune_epochs = epochs_config['finetune']
    initial_lr = lr_config['initial']

    # Each experiment gets a fresh model and optimizer
    net = MinimalNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=initial_lr)

    # --- Phase 1: Pre-training on MNIST ---
    print(f"\n--- Starting MNIST Pre-training (Method: {pretrain_method}) ---")
    global_step = 0
    net.train() # Set model to training mode
    for epoch in range(pretrain_epochs):
        for i, data in enumerate(mnist_loader, 0):
            lr_sign = 1
            if pretrain_method == 'alternating':
                lr_sign = -1 if global_step % 2 == 1 else 1
            elif pretrain_method == 'negative':
                lr_sign = -1

            # Set learning rate for the current step
            for param_group in optimizer.param_groups:
                param_group['lr'] = initial_lr * lr_sign

            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            global_step += 1
        print(f"Pre-training Epoch {epoch + 1}/{pretrain_epochs} complete.")

    # --- Phase 2: Fine-tuning on KMNIST ---
    print(f"--- Starting KMNIST Fine-tuning (from {pretrain_method} pre-train) ---")
    finetune_losses = []
    finetune_accuracies = []

    # CRITICAL: Ensure learning rate is positive for the fine-tuning phase
    for param_group in optimizer.param_groups:
        param_group['lr'] = initial_lr

    for epoch in range(finetune_epochs):
        net.train() # Set model to training mode
        running_loss = 0.0
        for i, data in enumerate(kmnist_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(kmnist_loader)
        finetune_losses.append(epoch_loss)

        # Validation on KMNIST test set
        net.eval() # Set model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():
            for data in kmnist_test_loader:
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        finetune_accuracies.append(accuracy)
        print(
            f"Fine-tuning Epoch {epoch + 1}/{finetune_epochs}, "
            f"Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%"
        )

    return finetune_losses, finetune_accuracies

# --- 4. Plotting Function ---
def plot_results(results, finetune_epochs):
    """
    Generates and displays comparative plots for the experiment results.

    Args:
        results (dict): A dictionary containing the loss and accuracy lists
                        for each pre-training method.
        finetune_epochs (int): The number of epochs used for fine-tuning.
    """
    print("\n--- Generating Plots ---")
    plt.figure(figsize=(16, 7))
    finetune_epochs_range = range(1, finetune_epochs + 1)

    # Color and style mapping for plots
    styles = {
        'standard': {'color': 'g', 'style': 'o-', 'label': 'Pre-trained w/ Standard LR'},
        'alternating': {'color': 'b', 'style': 's-', 'label': 'Pre-trained w/ Alternating LR'},
        'negative': {'color': 'y', 'style': '^-', 'label': 'Pre-trained w/ Negative LR'}
    }

    # Plot Training Loss Comparison
    plt.subplot(1, 2, 1)
    for method, data in results.items():
        plt.plot(finetune_epochs_range, data['losses'], styles[method]['style'],
                 color=styles[method]['color'], label=styles[method]['label'])
    plt.title('KMNIST Fine-tuning Loss Comparison', fontsize=16)
    plt.xlabel('Fine-tuning Epochs', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.legend()
    plt.grid(True)

    # Plot Validation Accuracy Comparison
    plt.subplot(1, 2, 2)
    for method, data in results.items():
        plt.plot(finetune_epochs_range, data['accuracies'], styles[method]['style'],
                 color=styles[method]['color'], label=styles[method]['label'])
    plt.title('KMNIST Validation Accuracy Comparison', fontsize=16)
    plt.xlabel('Fine-tuning Epochs', fontsize=12)
    plt.ylabel('Validation Accuracy (%)', fontsize=12)
    plt.legend()
    plt.grid(True)

    plt.suptitle('Comparison of Pre-training Strategies for Transfer Learning', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("transfer_learning_comparison.png")
    plt.show()
    print("Plot saved as 'transfer_learning_comparison.png'")

# --- 5. Main Execution Block ---
def main():
    """Main function to run the entire experimental pipeline."""
    # --- Configuration ---
    EPOCHS_CONFIG = {'pretrain': 27, 'finetune': 200} 
    LR_CONFIG = {'initial': 0.01}
    PRETRAIN_METHODS = ['standard', 'alternating', 'negative']

    # --- Pipeline ---
    loaders = load_datasets()
    all_results = {}

    for method in PRETRAIN_METHODS:
        losses, accuracies = run_experiment(method, loaders, EPOCHS_CONFIG, LR_CONFIG)
        all_results[method] = {'losses': losses, 'accuracies': accuracies}

    plot_results(all_results, EPOCHS_CONFIG['finetune'])

if __name__ == "__main__":
    main()
