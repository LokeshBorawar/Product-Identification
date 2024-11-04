import argparse
import cv2
import torch
import os
import shutil

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.transforms.functional import to_pil_image
from threading import Thread
from tqdm import tqdm
from PIL import Image

from dataset import VideoDataset, ZipDataset
from dataset import augmentation as A
from model import MattingBase, MattingRefine
from inference_utils import HomographicAlignment

import torch_directml
device=torch_directml.device()

# --------------- Direct Parameters Setup ---------------

# Define the parameters you want to set
args = argparse.Namespace(
    model_type='mattingrefine',
    model_backbone='resnet101',
    model_backbone_scale=0.25,
    model_refine_mode='sampling',
    model_refine_sample_pixels=80000,
    model_checkpoint="weights/pytorch_resnet101.pth",
    video_src="inputs/rsrc.mp4",
    video_bgr="inputs/bgr-Photoroom.png",
    video_target_bgr=None,
    video_resize=None,
    device=device,
    preprocess_alignment=False,
    output_dir="outputs/",
    output_types=['com', 'fgr', 'pha', 'err', 'ref'],
    output_format='video'
)

# --------------- Utils ---------------

class VideoWriter:
    def __init__(self, path, frame_rate, width, height):
        self.out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height))
        
    def add_batch(self, frames):
        frames = frames.mul(255).byte()
        frames = frames.cpu().permute(0, 2, 3, 1).numpy()
        for i in range(frames.shape[0]):
            frame = frames[i]
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.out.write(frame)
            
class ImageSequenceWriter:
    def __init__(self, path, extension):
        self.path = path
        self.extension = extension
        self.index = 0
        os.makedirs(path)
        
    def add_batch(self, frames):
        Thread(target=self._add_batch, args=(frames, self.index)).start()
        self.index += frames.shape[0]
            
    def _add_batch(self, frames, index):
        frames = frames.cpu()
        for i in range(frames.shape[0]):
            frame = frames[i]
            frame = to_pil_image(frame)
            frame.save(os.path.join(self.path, str(index + i).zfill(5) + '.' + self.extension))

# --------------- Main ---------------

device = torch.device(args.device)

# Load model
if args.model_type == 'mattingbase':
    model = MattingBase(args.model_backbone)
if args.model_type == 'mattingrefine':
    model = MattingRefine(
        args.model_backbone,
        args.model_backbone_scale,
        args.model_refine_mode,
        args.model_refine_sample_pixels
    )

model = model.to(device).eval()
model.load_state_dict(torch.load(args.model_checkpoint, map_location=device), strict=False)

# Load video and background
vid = VideoDataset(args.video_src)
bgr = [Image.open(args.video_bgr).convert('RGB')]
dataset = ZipDataset([vid, bgr], transforms=A.PairCompose([
    A.PairApply(T.Resize(args.video_resize[::-1]) if args.video_resize else nn.Identity()),
    HomographicAlignment() if args.preprocess_alignment else A.PairApply(nn.Identity()),
    A.PairApply(T.ToTensor())
]))

# Create output directory
if os.path.exists(args.output_dir):
    if input(f'Directory {args.output_dir} already exists. Override? [Y/N]: ').lower() == 'y':
        shutil.rmtree(args.output_dir)
    else:
        exit()
os.makedirs(args.output_dir)

# Prepare writers
h = args.video_resize[1] if args.video_resize is not None else vid.height
w = args.video_resize[0] if args.video_resize is not None else vid.width

if args.output_format == 'video':
    if 'com' in args.output_types:
        com_writer = VideoWriter(os.path.join(args.output_dir, 'com.mp4'), vid.frame_rate, w, h)
    if 'pha' in args.output_types:
        pha_writer = VideoWriter(os.path.join(args.output_dir, 'pha.mp4'), vid.frame_rate, w, h)
    if 'fgr' in args.output_types:
        fgr_writer = VideoWriter(os.path.join(args.output_dir, 'fgr.mp4'), vid.frame_rate, w, h)
    if 'err' in args.output_types:
        err_writer = VideoWriter(os.path.join(args.output_dir, 'err.mp4'), vid.frame_rate, w, h)
    if 'ref' in args.output_types:
        ref_writer = VideoWriter(os.path.join(args.output_dir, 'ref.mp4'), vid.frame_rate, w, h)
else:
    if 'com' in args.output_types:
        com_writer = ImageSequenceWriter(os.path.join(args.output_dir, 'com'), 'png')
    if 'pha' in args.output_types:
        pha_writer = ImageSequenceWriter(os.path.join(args.output_dir, 'pha'), 'jpg')
    if 'fgr' in args.output_types:
        fgr_writer = ImageSequenceWriter(os.path.join(args.output_dir, 'fgr'), 'jpg')
    if 'err' in args.output_types:
        err_writer = ImageSequenceWriter(os.path.join(args.output_dir, 'err'), 'jpg')
    if 'ref' in args.output_types:
        ref_writer = ImageSequenceWriter(os.path.join(args.output_dir, 'ref'), 'jpg')

# Conversion loop
with torch.no_grad():
    for input_batch in tqdm(DataLoader(dataset, batch_size=1, pin_memory=True)):
        src, bgr = input_batch
        src = src.to(device, non_blocking=True)
        bgr = bgr.to(device, non_blocking=True)
        
        if args.model_type == 'mattingrefine':
            pha, fgr, _, _, err, ref = model(src, bgr)

        if 'com' in args.output_types:
            # Output composite with black background
            com = fgr * pha + (1 - pha) * torch.tensor([0/255, 0/255, 0/255], device=device).view(1, 3, 1, 1)
            com_writer.add_batch(com)
        if 'pha' in args.output_types:
            pha_writer.add_batch(pha)
        if 'fgr' in args.output_types:
            fgr_writer.add_batch(fgr)
        if 'err' in args.output_types:
            err_writer.add_batch(F.interpolate(err, src.shape[2:], mode='bilinear', align_corners=False))
        if 'ref' in args.output_types:
            ref_writer.add_batch(F.interpolate(ref, src.shape[2:], mode='nearest'))
