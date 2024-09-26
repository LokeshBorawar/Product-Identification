# Do in colab if got _C error
"""
# Navigate to the groundingdino directory
%cd /content/drive/MyDrive/Colab Notebooks/groundingdino/
import locale
def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding
# Compile the C++ extensions
!python setup.py build_ext --inplace
"""

from groundingdino.util.slconfig import SLConfig
from groundingdino.models import build_model
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import annotate, load_image, predict
from huggingface_hub import hf_hub_download
import torch
import cv2
import os
import shutil


def save_video_frames(video_path):
    # Check if the video path exists
    if not os.path.exists(video_path):
        print(f"Video path '{video_path}' does not exist.")
        return
    
    # Get the video name without the extension and create a folder with the same name
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_folder = os.path.join(os.path.dirname(video_path), video_name)

    # Create a new directory for storing frames
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        return output_folder

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Cannot open video file: {video_path}")
        return

    # Initialize frame count
    frame_count = 0

    while True:
        # Read each frame from the video
        ret, frame = cap.read()

        # Break the loop if no frame is returned (end of video)
        if not ret:
            break

        # Save the frame as an image in the new folder
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:08d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    # Release the video capture object
    cap.release()

    print(f"Frames saved in folder '{output_folder}'.")

    return output_folder

def load_model_hf(repo_id, filename, ckpt_config_filename, device='cuda' if torch.cuda.is_available() else 'cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model   

ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swint_ogc.pth" # groundingdino_swinb_cogcoor.pth, groundingdino_swint_ogc.pth
ckpt_config_filename = "GroundingDINO_SwinT_OGC.cfg.py" # GroundingDINO_SwinB.cfg.py, GroundingDINO_SwinT_OGC.cfg.py

model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)


TEXT_PROMPT = "What is the person grasping in their hands?"#what are in the person's hands?"#"what is in the person's hand?"
BOX_THRESHOLD = 0.23
TEXT_THRESHOLD = 0.21


video_path="asset/HD CCTV Camera video 3MP 4MP iProx CCTV HDCCTVCameras.net retail store.mp4"
output_folder=save_video_frames(video_path)

local_frames_path=output_folder+"/" # "asset/HD CCTV Camera video 3MP 4MP iProx CCTV HDCCTVCameras.net retail store.mp4"
frame_names=os.listdir(local_frames_path)
frame_names.sort()

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
video_writer = cv2.VideoWriter("asset/outputs/"+os.path.basename(video_path), fourcc, fps, (frame_width, frame_height))


class_ob="what"
for frame_name in frame_names:
    local_image_path=local_frames_path+frame_name

    image_source, image = load_image(local_image_path)

    boxes, logits, phrases = predict(
        model=model, 
        image=image, 
        caption=TEXT_PROMPT, 
        box_threshold=BOX_THRESHOLD, 
        text_threshold=TEXT_THRESHOLD
    )

    print(phrases)
    if class_ob not in phrases:#len(boxes) == 0
        print(f"No objects of the '{TEXT_PROMPT}' prompt detected in the image.")
        annotated_frame=image_source[...,::-1] # RGB to BGR
    else:
        # puts 'class_ob' at the last.
        #sorted_indices = sorted(range(len(phrases)), key=lambda i: phrases[i].lower() == class_ob)
        #sorted_phrases = [phrases[i] for i in sorted_indices]
        #sorted_boxes = boxes[sorted_indices]
        #sorted_logits = logits[sorted_indices]

        # Filter tensors based on the indices of 'class_ob'
        indices = [i for i, item in enumerate(phrases) if item.lower() == class_ob]
        sorted_boxes = boxes[indices]
        sorted_logits = logits[indices]
        # Ensure tensors maintain their dimensions if only one "yes" is found
        if sorted_boxes.ndim == 1:  # For the 2D tensor, check if it became 1D and unsqueeze
            sorted_boxes = sorted_boxes.unsqueeze(0)
        if sorted_logits.ndim == 0:  # For the 1D tensor, check if it became a scalar and unsqueeze
            sorted_logits = sorted_logits.unsqueeze(0)
        sorted_phrases=["object"]*len([phrases[i] for i in indices])

        annotated_frame = annotate(image_source=image_source, boxes=sorted_boxes, logits=sorted_logits, phrases=sorted_phrases)
        #annotated_frame = annotated_frame[...,::-1] # BGR to RGB
        print(sorted_phrases)

    video_writer.write(annotated_frame)

video_writer.release()

shutil.rmtree(output_folder)