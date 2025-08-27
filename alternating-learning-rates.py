import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

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
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {losses[-1]:.4f}')
    return losses

# Experiment 1: Linear Regression
def exp1_linear():
    print("\n--- Experiment 1: Linear Regression with Alternating LR ---")
    x = torch.tensor([[1.], [2.], [3.], [4.]])
    y = torch.tensor([[2.], [4.], [6.], [8.]])
    data_loader = [(x, y)]  # Single batch
    
    model = nn.Linear(1, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    
    losses = train_model(model, optimizer, loss_fn, data_loader, epochs=1000, lr_scheme='alternating')
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Final params: w={model.weight.item():.3f}, b={model.bias.item():.3f}")

# Experiment 2: MLP on XOR
def exp2_xor():
    print("\n--- Experiment 2: MLP on XOR with Large Alternating LR ---")
    inputs = torch.tensor([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float32)
    targets = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
    data_loader = [(inputs, targets)]
    
    model = nn.Sequential(
        nn.Linear(2, 4), nn.ReLU(),
        nn.Linear(4, 1), nn.Sigmoid()
    )
    optimizer = optim.SGD(model.parameters(), lr=1.0)
    loss_fn = nn.BCELoss()
    
    losses = train_model(model, optimizer, loss_fn, data_loader, epochs=10000, lr_scheme='alternating', base_lr=1.0)
    print(f"Final loss: {losses[-1]:.4f}")
    with torch.no_grad():
        preds = model(inputs)
        print(f"Predictions: {preds.flatten().tolist()}")

# Experiment 3: Continual Learning with Autoencoder
def exp3_continual():
    print("\n--- Experiment 3: Continual Learning with Autoencoder ---")
    # Synthetic data
    class0 = torch.tensor(np.random.normal(0.2, 0.1, size=(1000, 784)).clip(0, 1), dtype=torch.float32)
    class1 = torch.tensor(np.random.normal(0.8, 0.1, size=(1000, 784)).clip(0, 1), dtype=torch.float32)
    loader0 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(class0, class0), batch_size=100, shuffle=True)
    loader1 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(class1, class1), batch_size=100, shuffle=True)
    
    class Autoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 32))
            self.decoder = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 784), nn.Sigmoid())
        def forward(self, x):
            return self.decoder(self.encoder(x))
    
    def run_variant(scheme_pretrain):
        model = Autoencoder()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()
        
        print(f"\nVariant: Pretrain with {scheme_pretrain}, Adapt with positive")
        # Pretrain on class0
        train_model(model, optimizer, loss_fn, loader0, epochs=50, lr_scheme=scheme_pretrain)
        loss0_after_pre = evaluate(model, loss_fn, loader0)
        loss1_initial = evaluate(model, loss_fn, loader1)
        print(f"After pretrain - Loss on Class0: {loss0_after_pre:.4f}, Initial Loss on Class1: {loss1_initial:.4f}")
        
        # Adapt to class1 with positive LR
        adapt_epochs = 0
        while True:
            train_model(model, optimizer, loss_fn, loader1, epochs=1, lr_scheme='positive')
            adapt_epochs += 1
            loss1 = evaluate(model, loss_fn, loader1)
            if loss1 < 0.005 or adapt_epochs > 100:
                break
        loss0_final = evaluate(model, loss_fn, loader0)
        loss1_final = evaluate(model, loss_fn, loader1)
        print(f"Adapted in {adapt_epochs} epochs - Final Loss on Class1: {loss1_final:.4f}, Final Loss on Class0: {loss0_final:.4f}")
    
    def evaluate(model, loss_fn, loader):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in loader:
                outputs = model(inputs)
                total_loss += loss_fn(outputs, targets).item()
        return total_loss / len(loader)
    
    run_variant('alternating')
    run_variant('negative')
    run_variant('positive')

# Run all experiments
if __name__ == "__main__":
    exp1_linear()
    exp2_xor()
    exp3_continual()
