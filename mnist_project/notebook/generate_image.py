import torch
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
import os

save_dir = "test_images"
os.makedirs(save_dir, exist_ok=True)

mnist = datasets.MNIST(root="../data/raw", train=False, download=True)

used_labels = set()

for idx, (image, label) in enumerate(mnist):
    if label not in used_labels:
        image_array = np.array(image)
        save_path = os.path.join(save_dir, f"{label}.png")
        plt.imsave(save_path, image_array, cmap="gray", format="png")
        print(f"Image {label} sauvegardée sous : {save_path}")
        used_labels.add(label)
    
    if len(used_labels) == 10:
        break
