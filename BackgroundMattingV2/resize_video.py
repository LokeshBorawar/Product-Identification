import cv2

def resize_video(input_video_path, output_video_path):
    # Load the video
    cap = cv2.VideoCapture(input_video_path)
    
    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get original video properties
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate new dimensions
    new_width = original_width - (original_width % 4)
    new_height = original_height - (original_height % 4)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change this codec if needed
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (new_width, new_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if no frame is returned

        # Resize the frame to the new dimensions
        resized_frame = cv2.resize(frame, (new_width, new_height))

        # Write the resized frame to the output video
        out.write(resized_frame)

    # Release everything if the job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video processed and saved as:", output_video_path)

# Example usage
resize_video('BackgroundMattingV2/inputs/src.mp4', 'BackgroundMattingV2/inputs/rsrc.mp4')
