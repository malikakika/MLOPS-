import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset
from src.model.convnet import ConvNet
import os

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("evice utilisé :", device)

tf = transforms.Compose([
    transforms.RandomAffine(degrees=25, translate=(0.2, 0.2), shear=15),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

mnist_train = datasets.MNIST("../data/raw", train=True, download=True, transform=tf)
custom_path = "streamlit_custom"
if os.path.exists(custom_path):
    print("Dataset Streamlit personnalisé trouvé.")
    streamlit_dataset = ImageFolder(custom_path, transform=tf)
    full_dataset = ConcatDataset([mnist_train, streamlit_dataset])
else:
    print("Dataset personnalisé introuvable. Entraînement uniquement sur MNIST.")
    full_dataset = mnist_train

train_loader = DataLoader(full_dataset, batch_size=64, shuffle=True)

test_loader = DataLoader(
    datasets.MNIST("../data/raw", train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=64, shuffle=False
)

def train(model, perm=None, n_epoch=30):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters())
    if perm is None:
        perm = torch.arange(0, 784).long()

    for epoch in range(n_epoch):
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(-1, 28 * 28)[:, perm].view(-1, 1, 28, 28)

            optimizer.zero_grad()
            logits = model(data)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f" epoch {epoch+1}/{n_epoch}: avg train loss = {total_loss / len(train_loader):.4f}")

def test(model, perm=None):
    model.eval()
    test_loss = 0
    correct = 0
    if perm is None:
        perm = torch.arange(0, 784).long()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(-1, 28 * 28)[:, perm].view(-1, 1, 28, 28)
            logits = model(data)
            test_loss += F.cross_entropy(logits, target, reduction="sum").item()
            pred = torch.argmax(logits, dim=1)
            correct += pred.eq(target).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    print(f"test loss={test_loss:.4f}, accuracy={accuracy:.4f}")

def main():
    input_size = 28 * 28
    output_size = 10
    n_kernels = 6

    perm = torch.arange(0, 784).long()
    model = ConvNet(input_size, n_kernels, output_size).to(device)

    print(f"📦 Parameters = {sum(p.numel() for p in model.parameters()) / 1e3:.3f}K")

    train(model, perm=perm, n_epoch=30)
    test(model, perm=perm)

    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), "model/mnist-0.0.1.pt")
    print("Modèle sauvegardé sous : model/mnist-0.0.1.pt")

if __name__ == "__main__":
    main()
