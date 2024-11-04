import cv2
import torch
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image
import numpy as np
from tqdm import tqdm
import time
from model import MattingBase, MattingRefine
import torch_directml

torchscript_model=False

# Device configuration
device = torch_directml.device()  # Change to "cuda" if using a GPU

# Load the model
if torchscript_model==True:
    model = torch.jit.load('weights/torchscript_resnet50_fp32.pth').to(device).eval()
else:
    model_type='mattingrefine'
    model_backbone='resnet101'
    model_backbone_scale=0.25
    model_refine_mode='sampling'
    model_refine_sample_pixels=80000
    model_checkpoint="weights/pytorch_resnet101.pth"
    if model_type == 'mattingbase':
        model = MattingBase(model_backbone)
    if model_type == 'mattingrefine':
        model = MattingRefine(
            model_backbone,
            model_backbone_scale,
            model_refine_mode,
            model_refine_sample_pixels
        )
    model = model.to(device).eval()
    model.load_state_dict(torch.load(model_checkpoint, map_location=device), strict=False)

def remove_contours(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask to keep the main person and remove small contours
    mask = np.ones_like(image) * 255  # Start with a white mask

    # Set a threshold for the minimum contour area
    min_contour_area = 25000  # Adjust this value based on your image

    # Fill contours based on area
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_contour_area:  # If the contour is small, fill it with black
            cv2.drawContours(mask, [contour], -1, (0, 0, 0), thickness=cv2.FILLED)

    # Create a final image by combining the original image with the mask
    result = cv2.bitwise_and(image, mask)

    return result

# Input and output video paths
input_video_path = 'inputs/rsrc.mp4'  # Input video file
output_video_path = 'outputs/processed_video.mp4'  # Output video file

# Initialize video capture
cap = cv2.VideoCapture(input_video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for .mp4 format
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Progress bar
pbar = tqdm(total=frame_count, desc="Processing video", unit="frame")

# Track time for estimating the remaining time
start_time = time.time()

with torch.no_grad():
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break  # Exit if there are no more frames

        # Convert the frame to PIL Image for processing
        src = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Load the BGR image (assuming it is the same for all frames)
        bgr = Image.open('inputs/bgr.png').convert("RGB")
        
        # Convert images to tensor and add a batch dimension
        src_tensor = to_tensor(src).to(device).unsqueeze(0)
        bgr_tensor = to_tensor(bgr).to(device).unsqueeze(0)

        if torchscript_model==True:
            # Set model parameters based on input size
            if src_tensor.size(2) <= 2048 and src_tensor.size(3) <= 2048:
                model.backbone_scale = 1 / 4
                model.refine_sample_pixels = 80_000
            else:
                model.backbone_scale = 1 / 8
                model.refine_sample_pixels = 320_000

        # Get the output from the model
        if torchscript_model==True:
            pha, fgr = model(src_tensor, bgr_tensor)[:2]
        else:
            if model_type == 'mattingrefine':
                pha, fgr, pha_sm, fgr_sm, err_sm, ref_sm = model(src_tensor, bgr_tensor)
            if model_type == 'mattingbase':
                pha, fgr, err, hid = model(src_tensor, bgr_tensor)

        # Compute the composite image
        # * torch.tensor([0/255, 0/255, 0/255], device=device).view(1, 3, 1, 1) add black
        # background
        com = pha * fgr + (1 - pha) * torch.tensor([0/255, 0/255, 0/255], device=device).view(1, 3, 1, 1)

        # Convert tensor back to PIL image
        com_image = to_pil_image(com[0].cpu())

        # Convert PIL image back to OpenCV format
        output_frame = cv2.cvtColor(np.array(com_image), cv2.COLOR_RGB2BGR)
        output_frame = remove_contours(output_frame)

        # Write the processed frame to the output video
        out.write(output_frame)

        # Update progress bar
        pbar.update(1)

        # Calculate and display the estimated remaining time
        elapsed_time = time.time() - start_time
        frames_processed = i + 1
        avg_time_per_frame = elapsed_time / frames_processed
        remaining_time = avg_time_per_frame * (frame_count - frames_processed)
        pbar.set_postfix(remaining_frames=frame_count - frames_processed, est_time_left=f"{remaining_time:.2f} sec")

# Release resources
cap.release()
out.release()
pbar.close()

print(f"Processed video saved as {output_video_path}.")
