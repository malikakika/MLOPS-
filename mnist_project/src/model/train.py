import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset
from src.model.convnet import ConvNet
import os
device = "mps" if torch.backends.mps.is_available() else "cpu"

def train(model, train_loader, device, perm=None, n_epoch=30):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters())
    for epoch in range(n_epoch):
        for step, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            if perm is not None:
                data = data.view(-1, 28*28)[:, perm].view(-1, 1, 28, 28)
            optimizer.zero_grad()
            logits = model(data)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                print(f"epoch={epoch}, step={step}: train loss={loss.item():.4f}")

def test(model, test_loader, device, perm=None):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if perm is not None:
                data = data.view(-1, 28*28)[:, perm].view(-1, 1, 28, 28)
            logits = model(data)
            test_loss += F.cross_entropy(logits, target, reduction='sum').item()
            pred = torch.argmax(logits, dim=1)
            correct += pred.eq(target).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    print(f"test loss={test_loss:.4f}, accuracy={accuracy:.4f}")

def main():
    input_size = 28 * 28
    output_size = 10
    n_kernels = 16

    # 1. Création du modèle
    perm = torch.arange(0, 784).long()
    model = ConvNet(input_size, n_kernels, output_size).to(device)
    print(f"📦 Parameters = {sum(p.numel() for p in model.parameters()) / 1e3:.3f}K")

    # 2. Chargement des données MNIST
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000)

    # 3. Entraînement et test
    train(model, train_loader, device, perm=perm, n_epoch=80)
    test(model, test_loader, device, perm=perm)

    # 4. Sauvegarde du modèle
    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), "model/mnist-0.0.1.pt")
    print("Modèle sauvegardé sous : model/mnist-0.0.1.pt")

if __name__ == "__main__":
    main()