import torch
import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # Definindo as camadas convolucionais
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2) # Entrada: 1 canal (imagem em escala de cinza), Saída: 6 filtros, Kernel 5x5
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5) # Entrada: 6 canais, Saída: 16 filtros, Kernel 5x5
        
        # Definindo as camadas totalmente conectadas (fully connected)
        self.fc1 = nn.Linear(16*5*5, 120) # Entrada: 16*5*5 (flatten da última camada convolucional), Saída: 120 neurônios
        self.fc2 = nn.Linear(120, 84)     # Entrada: 120 neurônios, Saída: 84 neurônios
        self.fc3 = nn.Linear(84, 10)      # Entrada: 84 neurônios, Saída: 10 neurônios (um para cada classe do MNIST)

    def forward(self, x):
        # Definindo a passagem para frente (forward pass)
        x = torch.max_pool2d(torch.relu(self.conv1(x)), 2)  # Aplicação de ReLU após conv1 e max pooling
        x = torch.max_pool2d(torch.relu(self.conv2(x)), 2)  # Aplicação de ReLU após conv2 e max pooling
        x = torch.flatten(x, 1)                            # Flatten das saídas antes de passar pelas camadas fully connected
        x = torch.relu(self.fc1(x))                        # Aplicação de ReLU após fc1
        x = torch.relu(self.fc2(x))                        # Aplicação de ReLU após fc2
        x = self.fc3(x)                                    # Saída final após fc3 (sem função de ativação, pois usaremos CrossEntropyLoss)
        return x

# Testando o modelo para verificar se não há erros de sintaxe
if __name__ == "__main__":
    torch.manual_seed(42)
    model = LeNet5()
    print(model)