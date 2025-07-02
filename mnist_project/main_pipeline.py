import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from src.model.convnet import ConvNet
from src.model.mlp import MLP

# -------- Configuration --------
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
MODEL_TYPE = "cnn"
N_EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MODEL_PATH = f"model/mnist_{MODEL_TYPE}.pt"

# -------- Préparation des données --------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -------- Initialisation du modèle --------
if MODEL_TYPE == "cnn":
    model = ConvNet(input_size=28*28, n_kernels=10, output_size=10)
elif MODEL_TYPE == "mlp":
    model = MLP(input_size=28*28, n_hidden=256, output_size=10)
else:
    raise ValueError("MODEL_TYPE must be 'cnn' or 'mlp'")

model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# -------- Entraînement --------
print("[INFO] Début de l'entraînement...")
model.train()
for epoch in range(N_EPOCHS):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        if MODEL_TYPE == "mlp":
            inputs = inputs.view(inputs.size(0), -1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{N_EPOCHS} - Loss: {running_loss/len(train_loader):.4f}")

# -------- Évaluation --------
print("[INFO] Évaluation sur le jeu de test...")
model.eval()
preds = []
true_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        if MODEL_TYPE == "mlp":
            inputs = inputs.view(inputs.size(0), -1)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        preds.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

acc = accuracy_score(true_labels, preds)
print(f"[RESULT] Test Accuracy: {acc:.4f}")

# -------- Confusion Matrix --------
cm = confusion_matrix(true_labels, preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"Confusion Matrix ({MODEL_TYPE.upper()})")
plt.xlabel("Predicted")
plt.ylabel("True")
os.makedirs("outputs", exist_ok=True)
plt.savefig(f"outputs/confusion_matrix_{MODEL_TYPE}.png")
print("[INFO] Matrice de confusion sauvegardée dans outputs/")

# -------- Sauvegarde du modèle --------
os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)
print(f"[INFO] Modèle sauvegardé dans {MODEL_PATH}")