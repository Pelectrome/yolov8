import cv2
import time
from ultralytics import YOLO

# Initialize the USB camera (camera 0 is usually the default)
cap = cv2.VideoCapture(0)

# Set camera resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

# Load the YOLOv8 model
model = YOLO("yolov8n_openvino_model")

# Initialize a variable to track time
prev_time = 0

while True:
    # Capture frame-by-frame from USB camera
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Display the FPS on the frame
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("USB Camera", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
