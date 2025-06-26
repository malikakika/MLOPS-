import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import requests
import os
import cv2
from datetime import datetime

st.set_page_config(page_title="MNIST Live", layout="centered")
st.title("Dessine un chiffre (0 à 9)")
st.write("Clique sur ' Prédire' pour envoyer à l'API FastAPI")

canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=10,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

def preprocess_canvas_image(canvas_img):
    gray = 255 - canvas_img[:, :, 0:3].mean(axis=2).astype(np.uint8)

    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return Image.new("L", (28, 28), 0)

    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    digit = binary[y:y+h, x:x+w]

    digit = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - 20) // 2
    y_offset = (28 - 20) // 2
    canvas[y_offset:y_offset+20, x_offset:x_offset+20] = digit

    return Image.fromarray(canvas)

if st.button("Prédire"):
    if canvas_result.image_data is not None:
        image = preprocess_canvas_image(canvas_result.image_data)

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        filename = f"debug_{timestamp}.png"
        image.save(filename)

        st.image(image, caption="Image envoyée à l’API", width=100)

        with open(filename, "rb") as f:
            files = {"file": (filename, f, "image/png")}
            try:
                response = requests.post("http://mnist_api:8000/api/v1/predict", files=files)
                prediction = response.json().get("prediction", "Erreur")
                st.success(f"Prédiction : {prediction}")
            except Exception as e:
                st.error(f"Erreur de requête : {e}")
    else:
        st.warning("✏️ Dessine un chiffre avant de prédire.")
