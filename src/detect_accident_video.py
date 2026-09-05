import torch
import torch.nn as nn
import cv2
import torchvision.transforms as transforms
from torchvision.models.video import r3d_18

# ----------------------------
# LOAD MODEL
# ----------------------------
model = r3d_18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(torch.load("accident_model.pth", map_location="cpu"))
model.eval()

# ----------------------------
# TRANSFORM
# ----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((112,112)),
    transforms.Normalize(
        mean=[0.43216,0.394666,0.37645],
        std=[0.22803,0.22145,0.216989]
    )
])

# ----------------------------
# VIDEO INPUT
# ----------------------------
video_path = "test_videos/test_video.mp4"
cap = cv2.VideoCapture(video_path)

frames = []

cv2.namedWindow("Accident Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Accident Detection",1000,700)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = transform(rgb)

    frames.append(tensor)

    if len(frames) == 16:

        clip = torch.stack(frames)
        clip = clip.permute(1,0,2,3)
        clip = clip.unsqueeze(0)

        with torch.no_grad():
            output = model(clip)
            probs = torch.softmax(output, dim=1)

        accident_prob = probs[0][1].item()
        
        print("Accident probability:", accident_prob)

        if accident_prob > 0.35:
            cv2.putText(
                frame,
                "ACCIDENT DETECTED",
                (50,80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0,0,255),
                4
            )

        frames = []

    cv2.imshow("Accident Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()