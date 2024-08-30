import argparse
import torch
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
import time

from train import train_step, test_step, print_train_time  # Importa funções do train.py
from model import LeNet5  # Importa o modelo LeNet-5
from data_prep import load_data  # Importa a função para carregar os dados

def main():

  # Configuração do ArgumentParser
  parser = argparse.ArgumentParser(description='Treinamento do modelo LeNet-5 no dataset MNIST')

  # Definindo argumentos de linha de comando
  parser.add_argument('--epochs', type=int, default=5, help='Número de épocas para treinar (default: 6)')
  parser.add_argument('--batch_size', type=int, default=64, help='Tamanho do batch para o DataLoader (default: 64)')
  parser.add_argument('--learning_rate', type=float, default=0.1, help='Taxa de aprendizado para o otimizador (default: 0.1)')
  parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Dispositivo para treinamento (default: cuda se disponível)')

  args = parser.parse_args()

  # Acessando os argumentos passados
  EPOCHS = args.epochs
  BATCH_SIZE = args.batch_size
  LEARNING_RATE = args.learning_rate
  device = args.device

  print(f"Usando o dispositivo: {device}")
  print(f"Treinando por {EPOCHS} épocas com batch size {BATCH_SIZE} e learning rate {LEARNING_RATE}")

  # Carregando os dataloaders de treino e teste
  train_dataloader, test_dataloader = load_data(batch_size=BATCH_SIZE)

  # Inicializando o modelo LeNet-5
  model = LeNet5().to(device)

  # Definindo a função de perda e o otimizador
  loss_fn = CrossEntropyLoss()
  optimizer = SGD(model.parameters(), lr=LEARNING_RATE)

  # Loop de treinamento e validação
  train_time_start = time.time()
  for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")

    # Treinamento
    train_loss = train_step(model, train_dataloader, loss_fn, optimizer, device)
    
    # Validação
    test_loss, test_acc = test_step(model, loss_fn, test_dataloader, device)

    print(f"Train loss: {train_loss:.5f} | Test loss: {test_loss:.5f} | Test acc: {test_acc*100:.2f}%")

  train_time_end = time.time()
  total_train_time = print_train_time(train_time_start, train_time_end, device)

  # Salva o modelo treinado
  torch.save(model.state_dict(), "lenet5_mnist.pth")
  print("Modelo treinado salvo como lenet5_mnist.pth")

# Verificação para execução direta
if __name__ == "__main__":
    main()