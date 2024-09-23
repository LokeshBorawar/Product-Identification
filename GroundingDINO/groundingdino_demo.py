from groundingdino.util.slconfig import SLConfig
from groundingdino.models import build_model
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import annotate, load_image, predict
from huggingface_hub import hf_hub_download
import torch
import cv2
import os


def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model   

ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swint_ogc.pth" # groundingdino_swinb_cogcoor.pth, groundingdino_swint_ogc.pth
ckpt_config_filename = "GroundingDINO_SwinT_OGC.cfg.py" # GroundingDINO_SwinB.cfg.py, GroundingDINO_SwinT_OGC.cfg.py

model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)

local_image_path=".asset/redbull2.jpg"
image_source, image = load_image(local_image_path)

TEXT_PROMPT = "hand grasped object. person."#"object held by person"#"grasped object. person. hand."
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

boxes, logits, phrases = predict(
    model=model, 
    image=image, 
    caption=TEXT_PROMPT, 
    box_threshold=BOX_THRESHOLD, 
    text_threshold=TEXT_THRESHOLD
)

print(phrases)
if 'grasped object' not in phrases:#len(boxes) == 0
    print(f"No objects of the '{TEXT_PROMPT}' prompt detected in the image.")
    cv2.imwrite(".asset/outputs/"+os.path.basename(local_image_path),image_source[...,::-1])
else:
    # puts 'grasped object' at the last.
    #sorted_indices = sorted(range(len(phrases)), key=lambda i: phrases[i].lower() == "grasped object")
    #sorted_phrases = [phrases[i] for i in sorted_indices]
    #sorted_boxes = boxes[sorted_indices]
    #sorted_logits = logits[sorted_indices]

    # Filter tensors based on the indices of 'grasped object'
    indices = [i for i, item in enumerate(phrases) if item.lower() == 'grasped object']
    sorted_boxes = boxes[indices]
    sorted_logits = logits[indices]
    # Ensure tensors maintain their dimensions if only one "yes" is found
    if sorted_boxes.ndim == 1:  # For the 2D tensor, check if it became 1D and unsqueeze
        sorted_boxes = sorted_boxes.unsqueeze(0)
    if sorted_logits.ndim == 0:  # For the 1D tensor, check if it became a scalar and unsqueeze
        sorted_logits = sorted_logits.unsqueeze(0)
    sorted_phrases=[phrases[i] for i in indices]

    annotated_frame = annotate(image_source=image_source, boxes=sorted_boxes, logits=sorted_logits, phrases=sorted_phrases)
    #annotated_frame = annotated_frame[...,::-1] # BGR to RGB
    cv2.imwrite(".asset/outputs/"+os.path.basename(local_image_path),annotated_frame)
    print(sorted_phrases)