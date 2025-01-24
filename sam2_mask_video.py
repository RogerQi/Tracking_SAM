import tracking_SAM.sam2_video_clicker
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import hydra

from sam2.build_sam import build_sam2_video_predictor
import sam2

checkpoint = "./tracking_SAM/third_party/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
sam2_predictor = build_sam2_video_predictor(model_cfg, checkpoint, device="cuda")

assert isinstance(sam2_predictor, sam2.sam2_video_predictor.SAM2VideoPredictor)

# VIDEO_PATH = '/home/roger/cross-embodiment-transformer/data/recordings/1404-human_more_colorpad_left_picking_2025-01-20_22-23-13/episode_0_stereo.mp4'
VIDEO_PATH = '/home/roger/cross-embodiment-transformer/data/recordings/405-pick_on_color_pad_right_far_far-2025_01_13-19_29_04/episode_0_stereo.mp4'

# Read video and dump into a temp folder
import tempfile

video_reader = cv2.VideoCapture(VIDEO_PATH)
video_frames = []
while True:
    ret, frame = video_reader.read()
    if not ret:
        break
    video_frames.append(frame)

MID_IDX = 50

with tempfile.TemporaryDirectory() as temp_folder:
    os.makedirs(temp_folder, exist_ok=True)

    for idx, frame in enumerate(video_frames):
        if idx < MID_IDX:
            continue
        fname = "{}.jpg".format(str(idx).zfill(8))
        cv2.imwrite(os.path.join(temp_folder, fname), frame)

    anno = tracking_SAM.sam2_video_clicker.Annotator(temp_folder, sam2_predictor)

    anno.main()

    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in sam2_predictor.propagate_in_video(anno.inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()[0]
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    
    mask_frames = {}
    for frame_idx in sorted(video_segments.keys()):
        mask_np = np.zeros((video_frames[frame_idx].shape[0], video_frames[frame_idx].shape[1]), dtype=bool)
        for obj_idx in video_segments[frame_idx]:
            mask_np |= video_segments[frame_idx][obj_idx]
        mask_frames[frame_idx + MID_IDX] = mask_np
    
    # Write to video
    out_video_path = "/home/roger/annotate_video_out.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_video_path, fourcc, 30, (video_frames[0].shape[1], video_frames[0].shape[0]))
    for idx in range(len(video_frames)):
        img_bgr_np = video_frames[idx]
        if idx in mask_frames:
            mask_frame = mask_frames[idx]
            img_bgr_np[mask_frame] = 0
        out.write(img_bgr_np)
    out.release()
