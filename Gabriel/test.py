import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from torch.multiprocessing import freeze_support

# Define the neural network architecture

    
class MelanomaCNN(nn.Module):
    def __init__(self):
        super(MelanomaCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # Adjust input channels to 3 for RGB images
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)  # Adjust the input size based on your image size
        self.fc2 = nn.Linear(128, 1)  # Single output neuron for binary classification

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 64 * 56 * 56)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class MelanomaDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, limit=None):
        self.labels_df = pd.read_csv(csv_file).head(limit)  
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.labels_df.iloc[idx, 0] + ".jpg")
        image = Image.open(img_name).convert('RGB')  # Ensure the image has 3 channels (RGB)
        label = torch.tensor(self.labels_df.iloc[idx, 1], dtype=torch.long)  # Change dtype to torch.long
        if self.transform:
            image = self.transform(image)
        return image, label


if __name__ == '__main__':
    freeze_support() 
    # Adjust the paths based on your actual file locations
    train_dataset = MelanomaDataset(csv_file=r'E:\TC\4TC\TIP\Projet CVML\isic-2020-resized\train-labels.csv',
                                    img_dir=r'E:\TC\4TC\TIP\Projet CVML\isic-2020-resized\train-resized\train-resized',
                                    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]))


    test_dataset = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=5)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=5)

    # Initialize the model, loss function, and optimizer
    model = MelanomaCNN()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    # Test the model
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
