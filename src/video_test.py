import torch
import torchvision.transforms as transforms
import torchvision.models.video as models
import cv2
import numpy as np

# Load pretrained 3D model
from torchvision.models.video import r3d_18, R3D_18_Weights

weights = R3D_18_Weights.DEFAULT
model = r3d_18(weights=weights)
model.eval()

# Transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((112, 112)),
    transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                         std=[0.22803, 0.22145, 0.216989])
])

# Open video
cap = cv2.VideoCapture("videoplayback.mp4")

frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = transform(frame)
    frames.append(frame)

    # Process 16-frame clip
    if len(frames) == 16:
        clip = torch.stack(frames)        # [16, 3, 112, 112]
        clip = clip.permute(1, 0, 2, 3)   # [3, 16, 112, 112]
        clip = clip.unsqueeze(0)          # [1, 3, 16, 112, 112]

        with torch.no_grad():
            outputs = model(clip)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        print("Top prediction index:", torch.argmax(probabilities).item())

        frames = []

cap.release()