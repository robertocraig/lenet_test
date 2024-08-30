import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchmetrics.functional import accuracy
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

def test_step(model:torch.nn.Module, loss_fn:torch.nn.Module, dataloader:torch.utils.data.DataLoader, device:torch.device):
    test_loss, test_acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            # 1. Forward pass
            test_pred = model(X)

            # 2. Loss
            test_loss += loss_fn(test_pred, y) # accumulatively add up the loss per epoch

            # 3. Computa a acuracia
            test_acc += accuracy(target=y,
                                 preds=torch.softmax(test_pred,dim=1),
                                 task='multiclass',
                                 num_classes=10)

        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
        return test_loss, test_acc


# Função para imprimir o tempo de treinamento
def print_train_time(start: float, end: float, device: torch.device = None):
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time