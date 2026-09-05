from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

results = model("bus.jpg")

#Get annotated image
annotated_frame = results[0].plot()

#Show image using OpenCV
cv2.imshow("Detection", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()