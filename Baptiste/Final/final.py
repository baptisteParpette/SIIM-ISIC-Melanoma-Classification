# Modification du code de l'utilisateur pour répondre aux objectifs identifiés
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, models
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import precision_score, recall_score

# Créer un écrivain SummaryWriter
writer = SummaryWriter('runs/melanoma_experiment_22')  # Ajustez le chemin

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
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Charger l'ensemble de données
full_dataset = MelanomaDataset(csv_file='../../data/train-labels.csv',  # Ajustez le chemin
                               img_dir='../../data/train-resized/train-resized',     # Ajustez le chemin
                               transform=transform)

# Séparer l'ensemble de données en ensembles d'entraînement et de validation
train_size = int(0.8 * len(full_dataset))  # 80% pour l'entraînement
val_size = len(full_dataset) - train_size  # 20% pour la validation
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

def count_class_frequencies(dataset):
    class_counts = {}
    
    for _, label in dataset:
        label = int(label.item())
        if label in class_counts:
            class_counts[label] += 1
        else:
            class_counts[label] = 1

    return class_counts


# Calculer les poids pour chaque échantillon dans le dataset
class_counts = count_class_frequencies(full_dataset)
weights = [1.0 / class_counts[int(label.item())] for _, label in full_dataset]
train_sampler = WeightedRandomSampler(weights, num_samples=len(train_dataset), replacement=True)

# Utiliser le sampler correct pour chaque DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler, pin_memory=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=0)

# Charger le modèle pré-entraîné ResNet50 et dégeler tous les paramètres
model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = True  # Entraîner tout le modèle

# Remplacer la dernière couche fc pour la classification binaire
model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 1), nn.Sigmoid())
model = model.to(device)

# Fonction de perte et optimiseur
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimiser tous les paramètres

# Fonction pour calculer la précision
def calculate_accuracy(y_true, y_pred):
    predicted = y_pred.ge(.5).view(-1)
    return (y_true == predicted).sum().float() / len(y_true)

# Supposons que vous avez 2 classes (0 et 1)
num_classes = 2

# Entraîner le modèle
epochs = 10
for epoch in range(epochs):
    # Initialisation des compteurs par classe pour l'entraînement
    correct_pred_train = {classname: 0 for classname in range(num_classes)}
    total_pred_train = {classname: 0 for classname in range(num_classes)}
    
    model.train()
    train_loss, train_accuracy, train_precision, train_recall = 0, 0, 0, 0

    # Boucle d'entraînement
    for batch_idx, (data, targets) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training Epoch {epoch+1}/{epochs}", leave=False):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data).squeeze()
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_accuracy += calculate_accuracy(target, outputs).item()
        predictions = outputs.ge(.5).float()
        train_precision += precision_score(target.cpu(), predictions.cpu(), zero_division=0)
        train_recall += recall_score(target.cpu(), predictions.cpu(), zero_division=0)

        # Mettre à jour les compteurs par classe
        predictions = outputs.ge(.5).float()
        for label, prediction in zip(targets, predictions):
            if label == prediction:
                correct_pred_train[int(label.item())] += 1
            total_pred_train[int(label.item())] += 1

    for classname in range(num_classes):
        precision = 100 * float(correct_pred_train[classname]) / total_pred_train[classname]
        recall = 100 * float(correct_pred_train[classname]) / total_pred_train[classname]
        writer.add_scalar(f'Train/Precision_class_{classname}', precision, epoch)
        writer.add_scalar(f'Train/Recall_class_{classname}', recall, epoch)

    # Réinitialisation des compteurs pour la validation
    correct_pred_val = {classname: 0 for classname in range(num_classes)}
    total_pred_val = {classname: 0 for classname in range(num_classes)}

    train_loss /= len(train_loader)
    train_accuracy /= len(train_loader)
    train_precision /= len(train_loader)
    train_recall /= len(train_loader)
    
    # Enregistrement des métriques d'entraînement dans TensorBoard
    writer.add_scalar('Loss/Train', train_loss, epoch)
    writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
    writer.add_scalar('Precision/Train', train_precision, epoch)
    writer.add_scalar('Recall/Train', train_recall, epoch)

    # Validation
    model.eval()
    val_loss, val_accuracy, val_precision, val_recall = 0, 0, 0, 0
    with torch.no_grad():
        for data, targets in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{epochs}", leave=False):
            data, target = data.to(device), target.to(device)
            outputs = model(data).squeeze()
            loss = criterion(outputs, target)

            val_loss += loss.item()
            val_accuracy += calculate_accuracy(target, outputs).item()

            predictions = outputs.ge(.5).float()
            val_precision += precision_score(target.cpu(), predictions.cpu(), zero_division=0)
            val_recall += recall_score(target.cpu(), predictions.cpu(), zero_division=0)

            # Mettre à jour les compteurs par classe
            predictions = outputs.ge(.5).float()
            for label, prediction in zip(targets, predictions):
                if label == prediction:
                    correct_pred_val[int(label.item())] += 1
                total_pred_val[int(label.item())] += 1
        
        # Calculer la précision et le rappel par classe pour la validation
    for classname in range(num_classes):
        precision = 100 * float(correct_pred_val[classname]) / total_pred_val[classname]
        recall = 100 * float(correct_pred_val[classname]) / total_pred_val[classname]
        writer.add_scalar(f'Validation/Precision_class_{classname}', precision, epoch)
        writer.add_scalar(f'Validation/Recall_class_{classname}', recall, epoch)

    val_loss /= len(val_loader)
    val_accuracy /= len(val_loader)
    val_precision /= len(val_loader)
    val_recall /= len(val_loader)

    # Enregistrement des métriques de validation dans TensorBoard
    writer.add_scalar('Loss/Validation', val_loss, epoch)
    writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)
    writer.add_scalar('Precision/Validation', val_precision, epoch)
    writer.add_scalar('Recall/Validation', val_recall, epoch)


    # Affichage des statistiques pour chaque époque
    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}')
    print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}')

# Sauvegarder le modèle
torch.save(model.state_dict(), 'melanoma_model2.pth')

# Fermer le SummaryWriter
writer.close()