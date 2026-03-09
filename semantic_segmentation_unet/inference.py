import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import UNET
import matplotlib.pyplot as plt

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
MODEL_PATH = "my_checkpoint.pth.tar"
INPUT_IMAGE_PATH = r"C:\Users\prave\PycharmProjects\CNN PROJECT M1\semantic_segmentation_unet ( carvana )\img_6.png"  # path to the input image

# --- TRANSFORMS ---
transform = A.Compose([
    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
    A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
    ToTensorV2(),
])

# --- LOAD MODEL ---
model = UNET(in_channels=3, out_channels=1).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# --- LOAD AND PREPROCESS IMAGE ---
image = cv2.imread(INPUT_IMAGE_PATH)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

augmented = transform(image=image_rgb)
input_tensor = augmented["image"].unsqueeze(0).to(DEVICE)

# --- INFERENCE ---
with torch.no_grad():
    pred = model(input_tensor)
    pred = torch.sigmoid(pred)
    pred_mask = (pred > 0.5).float()

# --- POSTPROCESS FOR DISPLAY ---
pred_mask_np = pred_mask.squeeze().cpu().numpy()
pred_mask_resized = cv2.resize(pred_mask_np, (image.shape[1], image.shape[0]))

# --- DISPLAY USING MATPLOTLIB ---
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(image_rgb)
axs[0].set_title("Input Image")
axs[0].axis("off")

axs[1].imshow(pred_mask_resized, cmap="gray")
axs[1].set_title("Predicted Mask")
axs[1].axis("off")

plt.show()
