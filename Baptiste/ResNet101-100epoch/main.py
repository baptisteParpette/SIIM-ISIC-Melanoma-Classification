import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, models
import torch.nn.functional as F

# Vérifier si CUDA est disponible, sinon utiliser le CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Architecture simplifiée du CNN
class MelanomaCNN(nn.Module):
    def __init__(self):
        super(MelanomaCNN, self).__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.fc = nn.Linear(16 * 112 * 112, 1)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 16 * 112 * 112)
        x = torch.sigmoid(self.fc(x))
        return x


# Ensemble de données personnalisé
class MelanomaDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, limit=None):
        self.labels_df = pd.read_csv(csv_file).head(limit)  # Limite à 100 images
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.labels_df.iloc[idx, 0] + ".jpg")
        image = Image.open(img_name)
        label = torch.tensor(self.labels_df.iloc[idx, 1], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, label

# Transformations pour les images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Charger l'ensemble de données
train_dataset = MelanomaDataset(csv_file='/kaggle/input/melanome/data/train-labels.csv',  # Ajustez le chemin
                                img_dir='/kaggle/input/melanome/data/train-resized/train-resized',      # Ajustez le chemin
                                transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Charger ResNet50 pré-entraîné
model = models.resnet101(pretrained=True)

# Geler tous les paramètres du modèle
for param in model.parameters():
    param.requires_grad = False

# Remplacer la dernière couche fc pour notre classification binaire
model.fc = nn.Sequential(
    nn.Linear(2048, 1),
    nn.Sigmoid()
)

# Si une GPU est disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Fonction de perte et optimiseur
criterion = nn.BCELoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)  # Optimiser seulement les paramètres de model.fc

# Entraîner le modèle
epochs = 100
print("debut training")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        outputs = model(data).squeeze()
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader)}")

# Sauvegarder le modèle
torch.save(model.state_dict(), 'melanoma_model.pth')  # Ajustez le chemin
