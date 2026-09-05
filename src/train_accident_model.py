import argparse
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
CLIP_LEN = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class order used throughout the project: index 0 = accident, index 1 = normal.
# detect_accident_video.py must read the same index for "accident probability".
CLASSES = ["accident", "normal"]


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune r3d_18 on trimmed accident/normal clips")
    parser.add_argument("--train-dir", default="dataset/train", help="Dir with accident/ and normal/ subfolders")
    parser.add_argument("--val-dir", default="dataset/val", help="Dir with accident/ and normal/ subfolders")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--resume-from", default=None, help="Path to an existing accident_model.pth to continue training from")
    parser.add_argument("--output", default="accident_model.pth", help="Where to save the trained model")
    return parser.parse_args()


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

        for label, cls in enumerate(CLASSES):
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

        # Clips are pre-trimmed to the 2-5s accident/normal window by
        # trim_accident_clips.py, so this samples a 16-frame sub-window from
        # within that window rather than from arbitrary footage.
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


def build_model(resume_from=None):
    model = r3d_18(weights=None if resume_from else R3D_18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))

    if resume_from:
        model.load_state_dict(torch.load(resume_from, map_location=DEVICE))
        print(f"Resumed weights from {resume_from}")

    return model.to(DEVICE)


def evaluate(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for clips, labels in val_loader:
            clips, labels = clips.to(DEVICE), labels.to(DEVICE)
            outputs = model(clips)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total else 0.0


def main():
    args = parse_args()

    model = build_model(args.resume_from)

    train_dataset = VideoDataset(args.train_dir)
    val_dataset = VideoDataset(args.val_dir)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)

    best_val_acc = 0.0
    for epoch in range(args.epochs):
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

        val_acc = evaluate(model, val_loader)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.output)
            print(f"Saved new best model (val acc {val_acc:.4f}) to {args.output}")

    print(f"Training complete. Best val acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
