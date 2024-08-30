import argparse
import torch
from torch.nn import CrossEntropyLoss
from model import LeNet5  # Certifique-se de importar a definição do modelo
from train import test_step  # Função para avaliação
from data_prep import load_data  # Função para carregar dados

def main(model_path):
    # 1. Defina a Arquitetura do Modelo
    model = LeNet5()

    # Verifica o dispositivo disponível (CPU ou GPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # 2. Carregue os Pesos Salvos no Modelo
    model.load_state_dict(torch.load(model_path, map_location=device))

    # 3. Coloque o Modelo em Modo de Avaliação
    model.eval()

    # 4. Carregar o Dataloader de Teste
    _, test_dataloader = load_data(batch_size=64)  # Carrega apenas o test_dataloader

    # 5. Use o Modelo no `test_step` para Avaliação
    test_loss, test_acc = test_step(model, CrossEntropyLoss(), test_dataloader, device)

    # Exibe os resultados
    print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc * 100:.2f}%")

if __name__ == "__main__":
    # Configuração do ArgumentParser
    parser = argparse.ArgumentParser(description='Avaliação do modelo LeNet-5 no dataset MNIST')
    parser.add_argument('--model_path', type=str, default="lenet5_mnist.pth", help='Caminho para o arquivo do modelo salvo (default: lenet5_mnist.pth)')

    args = parser.parse_args()

    # Chama a função principal com o caminho do modelo passado
    main(args.model_path)