import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

def load_data(batch_size=64):
    """Carrega os dados de treino e teste para MNIST e retorna dataloaders.

    Args:
        batch_size (int, optional): Tamanho do batch para os dataloaders. Padrão é 64.

    Returns:
        Tuple[DataLoader, DataLoader]: train_dataloader, test_dataloader
    """
    # Carrega o dataset MNIST
    train_data = datasets.MNIST(
        root="data",
        train=True,
        transform=ToTensor(),
        download=True
    )

    test_data = datasets.MNIST(
        root="data",
        train=False,
        transform=ToTensor(),
        download=True
    )

    # Cria os dataloaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False
    )

    return train_dataloader, test_dataloader