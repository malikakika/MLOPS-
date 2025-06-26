import torch
from src.model.mlp import MLP

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = MLP(input_size=28*28, n_hidden=8, output_size=10).to(device)

print(f"Parameters={sum(p.numel() for p in model.parameters())/1e3:.3f}K")

dummy_input = torch.rand(1, 28*28).to(device)
output = model(dummy_input)
print("Sortie du modèle : ", output.shape)
