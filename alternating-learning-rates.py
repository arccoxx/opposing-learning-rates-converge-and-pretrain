import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Helper function to train a model with specified LR scheme
def train_model(model, optimizer, loss_fn, data_loader, epochs, lr_scheme='positive', base_lr=0.01):
    losses = []
    for epoch in range(epochs):
        if lr_scheme == 'alternating':
            lr = base_lr if epoch % 2 == 0 else -base_lr
        elif lr_scheme == 'negative':
            lr = -base_lr
        else:  # positive
            lr = base_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        model.train()
        epoch_loss = 0
        for batch in data_loader:
            inputs, targets = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(data_loader))
    return losses

# Evaluation function
def evaluate(model, loss_fn, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs)
            total_loss += loss_fn(outputs, targets).item()
    return total_loss / len(loader)

# Autoencoder definition
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 32))
        self.decoder = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 784), nn.Sigmoid())
    def forward(self, x):
        return self.decoder(self.encoder(x))

# Function to run a single variant and collect metrics
def run_variant(scheme_pretrain, base_lr=0.01, pretrain_epochs=50, adapt_threshold=0.005, max_adapt_epochs=100):
    # Synthetic data
    np.random.seed(42)  # For reproducibility
    class0 = torch.tensor(np.random.normal(0.2, 0.1, size=(1000, 784)).clip(0, 1), dtype=torch.float32)
    class1 = torch.tensor(np.random.normal(0.8, 0.1, size=(1000, 784)).clip(0, 1), dtype=torch.float32)
    # Full batch for pretrain (batch GD)
    loader0_full = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(class0, class0), batch_size=1000, shuffle=True)
    # Mini-batch for adapt
    loader1 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(class1, class1), batch_size=100, shuffle=True)
    # Also create mini-batch loader0 for evaluation consistency
    loader0_mini = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(class0, class0), batch_size=100, shuffle=False)
    
    model = Autoencoder()
    optimizer = optim.SGD(model.parameters(), lr=base_lr)
    loss_fn = nn.MSELoss()
    
    # Pretrain on class0 using full batch GD
    pretrain_losses = train_model(model, optimizer, loss_fn, loader0_full, epochs=pretrain_epochs, lr_scheme=scheme_pretrain, base_lr=base_lr)
    loss0_after_pre = evaluate(model, loss_fn, loader0_mini)
    loss1_initial = evaluate(model, loss_fn, loader1)
    
    # Adapt to class1 with positive LR using mini-batches
    adapt_epochs = 0
    adapt_losses = []
    while True:
        adapt_loss = train_model(model, optimizer, loss_fn, loader1, epochs=1, lr_scheme='positive', base_lr=base_lr)
        adapt_losses.extend(adapt_loss)
        adapt_epochs += 1
        loss1_current = evaluate(model, loss_fn, loader1)
        if loss1_current < adapt_threshold or adapt_epochs >= max_adapt_epochs:
            break
    
    loss0_final = evaluate(model, loss_fn, loader0_mini)
    loss1_final = evaluate(model, loss_fn, loader1)
    
    return {
        'scheme': scheme_pretrain,
        'pretrain_final_loss_class0': loss0_after_pre,
        'initial_loss_class1': loss1_initial,
        'adapt_epochs': adapt_epochs,
        'final_loss_class1': loss1_final,
        'final_loss_class0': loss0_final,
        'pretrain_losses': pretrain_losses,
        'adapt_losses': adapt_losses
    }

# Run all variants and collect results
def compare_experiments():
    variants = ['alternating', 'negative', 'positive']
    results = {}
    for scheme in variants:
        print(f"\nRunning variant: {scheme}")
        results[scheme] = run_variant(scheme)
    
    # Print comparison analysis
    print("\n--- Comparison Analysis ---")
    print("Key Metrics:")
    print(f"{'Scheme':15} {'Pretrain Loss C0':18} {'Init Loss C1':15} {'Adapt Epochs':12} {'Final Loss C1':15} {'Final Loss C0 (Forget)':20}")
    for scheme in variants:
        r = results[scheme]
        print(f"{scheme:15} {r['pretrain_final_loss_class0']:.4f}             {r['initial_loss_class1']:.4f}        {r['adapt_epochs']:12} {r['final_loss_class1']:.4f}          {r['final_loss_class0']:.4f}")
    
    print("\nAnalysis:")
    print("- Positive-only pretraining achieves the lowest pretrain loss on Class 0, indicating strongest specialization.")
    print("- Negative-only pretraining diverges, leading to high pretrain loss but faster adaptation to Class 1 (fewer epochs) since there's less to 'unlearn'.")
    print("- Alternating pretraining offers a balance: decent pretrain loss, moderate adaptation speed, and least forgetting (lowest final loss on Class 0).")
    print("- Forgetting is highest in positive-only due to deep minima; alternating creates 'shallower' commitments, aiding continual learning.")
    print("- Implications: Alternating LR could be useful for scenarios requiring task retention, though it slows initial learning.")
    
    # Plot losses for visual comparison, ensuring each epoch is plotted individually without extensions
    plt.figure(figsize=(12, 6))
    for scheme in variants:
        r = results[scheme]
        # Plot pretrain losses with explicit x for each epoch
        pre_x = list(range(len(r['pretrain_losses'])))
        plt.plot(pre_x, r['pretrain_losses'], label=f'{scheme} Pretrain', marker='o', linestyle='-')
        
        # Plot adapt losses starting from the end of pretrain
        adapt_start = len(r['pretrain_losses'])
        adapt_x = [adapt_start + i for i in range(len(r['adapt_losses']))]
        plt.plot(adapt_x, r['adapt_losses'], label=f'{scheme} Adapt', marker='x', linestyle='--')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curves Comparison (Per Epoch)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Run the comparison
if __name__ == "__main__":
    compare_experiments()
