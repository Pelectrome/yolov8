import cv2
import time
import os
import numpy as np
from ultralytics import YOLO

# Create output directory if it doesn't exist
output_dir = 'output-mp4'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize the video capture from an MP4 file
cap = cv2.VideoCapture('mp4/street.mp4')

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get the video frame width, height, and frames per second (FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object to save the output video
out = cv2.VideoWriter('output-mp4/street.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (frame_width, frame_height))

# Load the YOLOv8 model
model = YOLO("yolov8n_openvino_model")

# Initialize a variable to track time for FPS calculation
prev_time = 0

while True:
    # Capture frame-by-frame from the video file
    ret, frame = cap.read()
    
    if not ret:
        print("End of video or failed to grab frame")
        break

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Calculate FPS
    current_time = time.time()
    fps_display = 1 / (current_time - prev_time)
    prev_time = current_time

    # Display the FPS on the frame
    cv2.putText(annotated_frame, f"FPS: {fps_display:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Write the annotated frame to the output video file
    if annotated_frame is not None and annotated_frame.shape[0] == frame_height and annotated_frame.shape[1] == frame_width:
        out.write(annotated_frame.astype(np.uint8))  # Ensure correct datatype

    # Display the resulting frame (optional)
    cv2.imshow("Video", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Release the video capture and writer objects, and close all windows
cap.release()
out.release()
cv2.destroyAllWindows()
