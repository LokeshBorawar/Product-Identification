import cv2
from ultralytics import YOLO
import numpy as np

device="cpu"

def inpaint_black_areas(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to create a mask where black lines are
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)

    # Inpaint the black lines using the mask
    inpainted_image = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    return inpainted_image

def detect_person_in_video(video_path, confidence_level=0.8, class_ids=[0]):
    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')  # Ensure you have the correct model path

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the width and height of the video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a black image with the same dimensions
    background = np.zeros((height, width, 3), dtype=np.uint8)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection
        results = model.predict(frame, conf=confidence_level, classes=class_ids, device=device, verbose=False)

        # Get the box coordinates and confidence
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
            
        for box,conf in zip(boxes,confs):
            
            x1, y1, x2, y2 = map(int, box)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)  # Fill with black
            
        # Create a mask where the black pixels are located (assuming black is [0, 0, 0])
        mask = np.sum(frame, axis=-1)>0 # mask = np.all(background == [0, 0, 0], axis=-1)
        if np.sum(mask)==0:
            break
        # Fill the black pixels in the output image with values from the second image
        background[mask] = frame[mask]

        # Display the current frame
        cv2.imshow('Detected Persons', frame)
        # Display the background
        cv2.imshow('background', background)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    final_image=background.copy()
    for _ in range(50):
        final_image=inpaint_black_areas(final_image)
    cv2.imwrite("inputs/bgr.png",final_image)
    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Path to your video file and confidence level
video_path = 'inputs/rsrc.mp4'  # Update with your video path
detect_person_in_video(video_path, confidence_level=0.1, class_ids=[0])