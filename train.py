import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import time

# Função para o loop de treinamento
def train_step(model, dataloader, loss_fn, optimizer, device):
    model.train()  # Configura o modelo para modo de treinamento
    train_loss = 0
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Forward pass
        y_pred = model(X)
        
        # Calcula o loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(dataloader)
    return train_loss

# Função para imprimir o tempo de treinamento
def print_train_time(start: float, end: float, device: torch.device = None):
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time