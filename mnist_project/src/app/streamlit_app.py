import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import requests
import io

st.set_page_config(page_title="MNIST Digit Recognition", layout="centered")
st.title("MNIST Digit Recognition")

st.subheader("Dessine un chiffre (0–9)")

canvas_result = st_canvas(
    fill_color="black",  
    stroke_width=10,
    stroke_color="white", 
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype(np.uint8)).resize((28, 28)).convert("L")
    
    st.subheader("Aperçu 28x28")
    st.image(img.resize((100, 100)), width=100)

    if st.button("Envoyer au modèle"):
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        buffered.seek(0)

        try:
            response = requests.post(
                "http://mnist_api:8000/api/v1/predict",
                files={"file": ("image.png", buffered, "image/png")},
            )
            pred = response.json().get("prediction", "?")
            st.success(f" Le modèle prédit : *{pred}*")
        except Exception as e:
            st.error(f"Erreur de requête : {e}")