from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
from src.model.convnet import ConvNet
from PIL import Image
import io
import numpy as np

app = FastAPI()

input_size = 28 * 28
n_kernels = 16
output_size = 10

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = ConvNet(input_size, n_kernels, output_size).to(device)
model.load_state_dict(torch.load("model/mnist-0.0.1.pt", map_location=device))
model.eval()

def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    image = image.resize((28, 28))

    image_array = np.array(image, dtype=np.float32) / 255.0  
    tensor = torch.tensor(image_array).unsqueeze(0).unsqueeze(0)  

    tensor = (tensor - 0.1307) / 0.3081
    return tensor.to(device)

@app.post("/api/v1/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    tensor = transform_image(image_bytes)

    with torch.no_grad():
        logits = model(tensor)
        pred = torch.argmax(logits, dim=1).item()

    return JSONResponse(content={"prediction": pred})
