import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

print("Torch version:", torch.__version__)
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print("Using device:", device)
tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST("../data/raw", train=True, download=True, transform=tf),
    batch_size=64, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST("../data/raw", train=False, download=True, transform=tf),
    batch_size=64, shuffle=True
)

batch = next(iter(train_loader))
images = batch[0][:5]
labels = batch[1][:5]

for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(images[i][0], cmap="gray")
    plt.title(f"Label: {labels[i].item()}")
    plt.axis("off")
plt.show()
