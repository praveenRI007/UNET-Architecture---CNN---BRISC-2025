import numpy as np
import pandas as pd
import os

import cv2
import matplotlib.pyplot as plt

image_dir = 'data/train_images'
mask_dir = 'data/train_masks'

image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]

image_basenames = {os.path.splitext(f)[0]: f for f in image_files}
mask_basenames = {os.path.splitext(f)[0]: f for f in mask_files}

common_keys = set(image_basenames.keys()) & set(mask_basenames.keys())

data = []
for key in sorted(common_keys):
    image_path = os.path.join(image_dir, image_basenames[key])
    mask_path = os.path.join(mask_dir, mask_basenames[key])
    data.append({'image_path': image_path, 'mask_path': mask_path})


df = pd.DataFrame(data)


import os

image_dir = 'data/train_images'
mask_dir = 'data/train_masks'

print("Sample image files:", os.listdir(image_dir)[:5])
print("Sample mask files:", os.listdir(mask_dir)[:5])


plt.figure(figsize=(20, 8))

for i in range(5):
    image = cv2.imread(df.iloc[i]['image_path'])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(df.iloc[i]['mask_path'], cv2.IMREAD_GRAYSCALE)

    plt.subplot(2, 5, i + 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Image {i+1}")

    plt.subplot(2, 5, i + 6)
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    plt.title(f"Mask {i+1}")

plt.tight_layout()
plt.show()



import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.model_selection import train_test_split



class SegmentationDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = cv2.imread(self.df.loc[idx, 'image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.df.loc[idx, 'mask_path'], cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256))

        image = image.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))

        return torch.tensor(image), torch.tensor(mask).unsqueeze(0)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU()
            )

        self.enc1 = conv_block(3, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = conv_block(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)

        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return torch.sigmoid(self.final(d1))



def dice_score(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    smooth = 1e-5
    intersection = (preds * targets).sum(dim=(1,2,3))
    union = preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean().item()

def iou_score(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum(dim=(1,2,3))
    union = (preds + targets - preds * targets).sum(dim=(1,2,3))
    return ((intersection + 1e-5) / (union + 1e-5)).mean().item()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

train_ds = SegmentationDataset(train_df)
val_ds = SegmentationDataset(val_df)
train_dl = DataLoader(train_ds, batch_size=4, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=4)

model = UNet().to(device)

print(next(model.parameters()).device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCELoss()

for epoch in range(10):
    model.train()
    train_loss = 0
    for imgs, masks in train_dl:
        imgs, masks = imgs.to(device), masks.to(device)

        preds = model(imgs)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    val_dice, val_iou = 0, 0
    with torch.no_grad():
        for imgs, masks in val_dl:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)

            val_dice += dice_score(preds, masks)
            val_iou += iou_score(preds, masks)

    print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_dl):.4f} | "
          f"Val Dice: {val_dice/len(val_dl):.4f} | Val IoU: {val_iou/len(val_dl):.4f}")


model.eval()
plt.figure(figsize=(20, 8))

for i in range(5):
    image, mask = val_ds[i]
    pred = model(image.unsqueeze(0).to(device)).squeeze().detach().cpu().numpy()

    plt.subplot(3, 5, i + 1)
    plt.imshow(np.transpose(image.numpy(), (1, 2, 0)))
    plt.title("Image")
    plt.axis('off')

    plt.subplot(3, 5, i + 6)
    plt.imshow(mask.squeeze().numpy(), cmap='gray')
    plt.title("Ground Truth")
    plt.axis('off')

    plt.subplot(3, 5, i + 11)
    plt.imshow(pred > 0.5, cmap='gray')
    plt.title("Prediction")
    plt.axis('off')

plt.tight_layout()
plt.show()