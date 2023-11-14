import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from struct import unpack

from torch.utils.tensorboard import SummaryWriter
# Reconstruction du modèle en PyTorch basé sur l'architecture du code MATLAB

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # Convolution layer
        self.conv = nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=0)
        
        # Batch Normalization layer
        self.batchnorm = nn.BatchNorm2d(20)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Fully Connected layer
        self.fc = nn.Linear(20 * 12 * 12, 10)  # après le pooling, la taille de l'image est 12x12

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, 20 * 12 * 12)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
    
class MNISTDataset(torch.utils.data.Dataset):
    """Custom dataset for MNIST to read from IDX format and normalize on the fly."""
    
    def __init__(self, image_file, label_file):
        self.image_file = image_file
        self.label_file = label_file
        
        with open(label_file, 'rb') as f:
            _, self.num_items = unpack('>II', f.read(8))
        
    def __len__(self):
        return self.num_items
    
    def __getitem__(self, index):
        with open(self.image_file, 'rb') as f:
            f.seek(16 + index * 28 * 28)
            image_data = np.frombuffer(f.read(28 * 28), dtype=np.uint8).reshape(1, 28, 28)
        
        with open(self.label_file, 'rb') as f:
            f.seek(8 + index)
            label_data = np.frombuffer(f.read(1), dtype=np.uint8)[0]
        
        image_data = (torch.tensor(image_data, dtype=torch.float32) / 255.0) * 2 - 1
        return image_data, label_data

# Create custom dataset instances
train_custom_dataset = MNISTDataset("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte")
test_custom_dataset = MNISTDataset("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte")

# Create DataLoader instances   
batch_size=64

train_loader = DataLoader(train_custom_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_custom_dataset, batch_size=batch_size, shuffle=False)

# Check the size of the datasets
print(len(train_loader.dataset), len(test_loader.dataset))


# Créer l'instance du modèle
cnn_model = CNN()
if torch.cuda.is_available():
    cnn_model = cnn_model.cuda()

print(cnn_model)



# Paramètres
batch_size = 64
epochs = 10
lr = 0.001

# Définir la fonction de perte et l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=lr)

# Pour enregistrer la perte et la précision
train_losses = []
accuracies = []

# Créer un écrivain TensorBoard
writer = SummaryWriter()

# Ajouter le modèle à TensorBoard
data_iter = iter(train_loader)
images, _ = next(data_iter)
if torch.cuda.is_available():
    images = images.cuda()
writer.add_graph(cnn_model, images)

# Entraîner le modèle
for epoch in range(epochs):
    cnn_model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        outputs = cnn_model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Enregistrer la perte dans TensorBoard
    writer.add_scalar('Loss/train', total_loss / len(train_loader), epoch)

    # Évaluation de la précision et visualisation des prédictions
    cnn_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            outputs = cnn_model(data)
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
    accuracies.append(accuracy)
    writer.add_scalar('Accuracy/test', accuracy, epoch)

writer.close()

# Sauvegarder le modèle
torch.save(cnn_model.state_dict(), '/data/mnist_cnn_model.pth')

print(accuracies)
