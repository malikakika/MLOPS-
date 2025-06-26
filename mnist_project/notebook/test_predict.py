import torch
from PIL import Image
import numpy as np
from src.model.convnet import ConvNet

# Choisir le device
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("🚀 Device utilisé :", device)

# Charger le modèle entraîné
input_size = 28 * 28
output_size = 10
n_kernels = 6

model = ConvNet(input_size, n_kernels, output_size).to(device)
model.load_state_dict(torch.load("model/mnist-0.0.1.pt", map_location=device))
model.eval()

def transform_image(filepath):
    image = Image.open(filepath).convert("L").resize((28, 28))
    image_array = np.array(image, dtype=np.float32) / 255.0
    tensor = torch.tensor(image_array).unsqueeze(0).unsqueeze(0)  
    tensor = (tensor - 0.1307) / 0.3081
    return tensor.to(device)

image_path = "test_images/debug_20250626005401555503.png"  
input_tensor = transform_image(image_path)

with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.argmax(output, dim=1).item()

print(f" Prédiction du modèle : {prediction}")
