from ultralytics import YOLO
import cv2
import math

# Use more accurate model (important)
model = YOLO("yolov8s.pt")

cap = cv2.VideoCapture("videoplayback.mp4")

cv2.namedWindow("Traffic Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Traffic Detection", 1000, 700)

previous_positions = {}
previous_speeds = {}

HIGH_SPEED_THRESHOLD = 6
SUDDEN_DROP_THRESHOLD = 4
PROXIMITY_THRESHOLD = 120

ACCIDENT_PERSIST_FRAMES = 2
accident_counter = 0
accident_confirmed = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Use tracking
    results = model.track(frame, persist=True, verbose=False)

    detected_this_frame = False
    vehicle_centers = []

    if results[0].boxes.id is not None:

        for box, track_id in zip(results[0].boxes, results[0].boxes.id):

            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            # Debug: print detected classes (comment later)
            print("Detected:", class_name, round(confidence, 2))

            # IMPORTANT: include bicycle + lower confidence
            if class_name in ["car", "bus", "truck", "motorcycle", "bicycle"] and confidence > 0.3:

                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                vehicle_centers.append((track_id.item(), center_x, center_y))

                # Motion calculation
                if track_id.item() in previous_positions:

                    prev_x, prev_y = previous_positions[track_id.item()]
                    current_speed = math.sqrt(
                        (center_x - prev_x) ** 2 +
                        (center_y - prev_y) ** 2
                    )

                    prev_speed = previous_speeds.get(track_id.item(), 0)
                    speed_drop = prev_speed - current_speed

                    sudden_deceleration = (
                        prev_speed > HIGH_SPEED_THRESHOLD and
                        speed_drop > SUDDEN_DROP_THRESHOLD
                    )

                    if sudden_deceleration:
                        for other_id, ox, oy in vehicle_centers:
                            if other_id != track_id.item():
                                dist_between = math.sqrt(
                                    (center_x - ox) ** 2 +
                                    (center_y - oy) ** 2
                                )
                                if dist_between < PROXIMITY_THRESHOLD:
                                    detected_this_frame = True

                    previous_speeds[track_id.item()] = current_speed

                previous_positions[track_id.item()] = (center_x, center_y)

                # Draw ID
                cv2.putText(
                    frame,
                    f"ID {track_id.item()}",
                    (center_x, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)

    # Persistence logic
    if detected_this_frame:
        accident_counter += 1
    else:
        accident_counter = 0

    accident_flag = False

    if accident_counter >= ACCIDENT_PERSIST_FRAMES:
        accident_flag = True
        if not accident_confirmed:
            print("🚨 ACCIDENT DETECTED")
            accident_confirmed = True

    if accident_flag:
        cv2.putText(
            frame,
            "POSSIBLE ACCIDENT DETECTED",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    cv2.imshow("Traffic Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()