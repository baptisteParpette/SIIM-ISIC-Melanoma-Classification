import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F


# Architecture du CNN
class MelanomaCNN(nn.Module):
    def __init__(self):
        super(MelanomaCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.batchnorm1(self.conv1(x))))
        x = self.pool2(F.relu(self.batchnorm2(self.conv2(x))))
        x = self.pool3(F.relu(self.batchnorm3(self.conv3(x))))
        x = x.view(-1, 128 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Ensemble de données personnalisé
class MelanomaDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, limit=100):
        self.labels_df = pd.read_csv(csv_file).head(limit)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.labels_df.iloc[idx, 0] + ".jpg")
        image = Image.open(img_name)
        label = torch.tensor(self.labels_df.iloc[idx, 1], dtype=torch.float32)  # Convertir en float32
        if self.transform:
            image = self.transform(image)
        return image, label


# Transformations pour les images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Charger l'ensemble de données
train_dataset = MelanomaDataset(csv_file='../../data/train-labels.csv',  # Ajustez le chemin
                                img_dir='../../data/train-resized/train-resized',      # Ajustez le chemin
                                transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialisation du modèle, fonction de perte et optimiseur
model = MelanomaCNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entraînement du modèle
epochs = 10
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.float(), target.float()  # Convertir en float32
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target.unsqueeze(1))
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# Sauvegarder le modèle
torch.save(model.state_dict(), 'melanoma_model.pth')  # Ajustez le chemin
