import time
import cv2
import numpy as np
import pyautogui
from ultralytics import YOLO
import mss

# Initialize screen capture
sct = mss.mss()
monitor = sct.monitors[1]  # Use primary screen
screen_width = monitor['width']
screen_height = monitor['height']

# Load the YOLO model (use a lighter version like yolov8n or yolov8s)
model = YOLO("yolov8n.pt")  # You can choose yolov8s.pt, yolov8m.pt, etc.

# Initialize the screen capture region
scan_region = {
    "top": 0,
    "left": 0,
    "width": screen_width,
    "height": screen_height
}

prev_time = time.time()

while True:
    # Grab a screenshot of the entire screen
    sct_img = sct.grab(scan_region)
    frame = np.array(sct_img)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # Process the frame with YOLO to detect persons
    results = model(frame)[0]

    for box in results.boxes:
        if int(box.cls[0]) == 0:  # Class 0 = person
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get the coordinates of the bounding box
            confidence = box.conf[0].item()  # Get the confidence score

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Add the label with confidence percentage
            label = f"Person: {confidence*100:.2f}%"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Calculate and display FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Show the frame with bounding boxes
    cv2.imshow("Person Detector", frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
