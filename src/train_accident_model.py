import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models.video import r3d_18, R3D_18_Weights
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import random
import numpy as np

# ----------------------------
# CONFIG
# ----------------------------
DATASET_PATH = "dataset/train"
VAL_PATH = "dataset/val"
CLIP_LEN = 16
BATCH_SIZE = 2
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# VIDEO DATASET
# ----------------------------
class VideoDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((112, 112)),
            transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                                 std=[0.22803, 0.22145, 0.216989])
        ])

        classes = ["accident", "normal"]
        for label, cls in enumerate(classes):
            cls_path = os.path.join(root_dir, cls)
            for file in os.listdir(cls_path):
                if file.endswith(".mp4"):
                    self.samples.append((os.path.join(cls_path, file), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        cap = cv2.VideoCapture(video_path)

        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start_frame = 0
        if total_frames > CLIP_LEN:
            start_frame = random.randint(0, total_frames - CLIP_LEN)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for _ in range(CLIP_LEN):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.transform(frame)
            frames.append(frame)

        cap.release()

        if len(frames) < CLIP_LEN:
            frames += [frames[-1]] * (CLIP_LEN - len(frames))

        clip = torch.stack(frames)          # [16, 3, 112, 112]
        clip = clip.permute(1, 0, 2, 3)     # [3, 16, 112, 112]

        return clip, torch.tensor(label, dtype=torch.long)

# ----------------------------
# LOAD MODEL
# ----------------------------
weights = R3D_18_Weights.DEFAULT
model = r3d_18(weights=weights)

# Freeze backbone (important)
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(DEVICE)

# ----------------------------
# TRAIN SETUP
# ----------------------------
train_dataset = VideoDataset(DATASET_PATH)
val_dataset = VideoDataset(VAL_PATH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# ----------------------------
# TRAIN LOOP
# ----------------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for clips, labels in train_loader:
        clips, labels = clips.to(DEVICE), labels.to(DEVICE)

        outputs = model(clips)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

print("Training Complete")

torch.save(model.state_dict(),"accident_model.pth")
print("Model saved as accident_model.pth")