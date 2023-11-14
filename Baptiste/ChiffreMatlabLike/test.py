import tkinter as tk
from tkinter import Canvas
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from struct import unpack

# Votre architecture de modèle CNN (doit être identique à celle utilisée pour l'entraînement)
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

# Charger le modèle
model = CNN()
model.load_state_dict(torch.load('mnist_cnn_model.pth'))
model.eval()

def predict_digit(img):
    """Prédire le chiffre à partir d'une image dessinée."""
    img = img.resize((28, 28)).convert('L')  # Redimensionner et convertir en niveaux de gris
    img = np.array(img)
    img = (img / 255.0) * 2 - 1  # Normaliser
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        output = model(img)
        prediction = output.argmax(dim=1).item()
    return [output,prediction]

def on_submit():
    global last_x, last_y, draw
    prediction = predict_digit(image)
    label.config(text=f"Prédiction : {prediction}")
    canvas.delete("all")
    draw.rectangle([0, 0, 256, 256], fill=(255, 255, 255))
    last_x, last_y = None, None

def on_draw(event):
    global last_x, last_y
    x, y = event.x, event.y
    if last_x and last_y:
        canvas.create_line((last_x, last_y, x, y), width=8, fill='black', capstyle=tk.ROUND, smooth=tk.TRUE)
        draw.line((last_x, last_y, x, y), fill='black', width=8)
    last_x, last_y = x, y

def on_release(event):
    global last_x, last_y
    last_x, last_y = None, None

root = tk.Tk()
root.title("Prédiction de chiffre")

canvas = Canvas(root, bg="white", width=256, height=256)
canvas.pack(pady=20)

label = tk.Label(root, text="Dessinez un chiffre!")
label.pack(pady=20)

submit_button = tk.Button(root, text="Prédire", command=on_submit)
submit_button.pack(pady=20)

canvas.bind("<B1-Motion>", on_draw)
canvas.bind("<ButtonRelease-1>", on_release)

last_x, last_y = None, None

# Création d'une image PIL et d'un objet de dessin
image = Image.new("RGB", (256, 256), (255, 255, 255))
draw = ImageDraw.Draw(image)

root.mainloop()