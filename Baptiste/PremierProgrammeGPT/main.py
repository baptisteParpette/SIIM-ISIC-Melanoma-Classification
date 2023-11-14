import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

# Paramètres
batch_size = 64
epochs = 10
lr = 0.001

# Transformations pour les images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Chargement des données CIFAR-10
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Définition du modèle CNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = Net()
if torch.cuda.is_available():
    model = model.cuda()

# Fonction de perte et optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Pour enregistrer la perte et la précision
train_losses = []
accuracies = []

# Créer un écrivain TensorBoard
writer = SummaryWriter()

# Ajouter le modèle à TensorBoard
data_iter = iter(train_loader)
images, _ = next(data_iter)  # Prenez un batch d'images
if torch.cuda.is_available():
    images = images.cuda()
writer.add_graph(model, images)

# Entraînement
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Enregistrer la perte dans TensorBoard
    writer.add_scalar('Loss/train', total_loss / len(train_loader), epoch)

# Évaluation pour la précision et visualisation des prédictions
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_loader):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        outputs = model(data)
        _, predicted = outputs.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        # Ajouter quelques images et prédictions à TensorBoard
        if batch_idx == 0:  # Juste pour le premier lot
            for i in range(10):  # Ajoutons 10 images et leurs prédictions
                img = data[i] / 2 + 0.5  # un-normalize
                writer.add_image(f"Image {i}", img, epoch)
                writer.add_text(f"Prediction {i}", f"Predicted: {predicted[i]}, True: {target[i]}", epoch)

    # Enregistrer la précision dans TensorBoard
    accuracy = 100. * correct / total
    writer.add_scalar('Accuracy/test', accuracy, epoch)

writer.close()