from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("traffic1.mp4")

fallen_counter = 0
TRIGGER_FRAMES = 10

cv2.namedWindow("Traffic Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Traffic Detection", 1000,700)

while cap.isOpened():
	ret, frame = cap.read()
	if not ret:
		break

	results = model(frame)

	height = frame.shape[0]	#frame height
	accident_flag = False	

	for box in results[0].boxes:
		class_id = int(box.cls[0])
		class_name = model.names[class_id]
		confidence = float(box.conf[0])

		x1, y1, x2, y2 = box.xyxy[0]
		
		# Convert tensor to normal numbers
		x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

		box_width = x2 - x1
		box_height = y2 - y1
	
		if class_name == "person" and confidence > 0.6:

			near_bottom = y2 > height * 0.75

			horizontal_shape = box_width > box_height
			
			large_enough = box_width > 80 and box_height > 30

			if near_bottom and horizontal_shape and large_enough:
				fallen_counter +=1
			else:
				fallen_counter = 0

			if fallen_counter >= TRIGGER_FRAMES:
				accident_flag = True
				print("⚠ Possible fallen person detected")
		
	annotated_frame = results[0].plot()
	#Show alert text on screen if rule triggered
	if accident_flag:
		cv2.putText(annotated_frame,"POSSIBLE ACCIDENT DETECTED",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

	cv2.imshow("Traffic Detection",annotated_frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()