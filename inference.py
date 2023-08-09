import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import lightning as L
import pytorch_lightning as pl
from model.generator import RRDBNet
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import cv2

checkpoint="/home/fatih/esrgan/best_model.ckpt"

model = RRDBNet(3, 3, 64, 1, gc=32)
generator_params = model.state_dict()
checkpoint = torch.load(checkpoint)
for param_name in generator_params:
    if param_name in checkpoint["state_dict"]:
        model.load_state_dict({param_name: checkpoint["state_dict"][param_name]}, strict=False)

# disable randomness, dropout, etc...
model.eval()


data_transform = transforms.Compose([
            transforms.Resize((512, 512)),   # Gerekirse boyutu değiştirin
            transforms.ToTensor(),           # Tensor formatına çevirin
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalizasyon
        ])

# predict with the model

test_image_path="/home/fatih/esrgan/dataset/train/DIV2K_train_HR/0001.png"
label_image = Image.open(test_image_path).convert('RGB')
image=data_transform(label_image)
y_hat = model(image)
numpy_array = y_hat.detach().numpy()
# NumPy arrayini görselleştirin
plt.imshow(numpy_array[0], cmap="gray")  # İndeksi 0 olan kanalı görselleştirin (örneğin, siyah beyaz bir görüntü)
plt.show()
