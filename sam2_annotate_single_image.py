import tracking_SAM.plt_clicker
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

import sam2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

checkpoint = "./tracking_SAM/third_party/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
sam2_predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

assert isinstance(sam2_predictor, SAM2ImagePredictor)

import pdb; pdb.set_trace()

img = np.array(Image.open("sample_data/DAVIS_bear/images/00000.jpg"))

# with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
#     predictor.set_image(<your_image>)
#     masks, _, _ = predictor.predict(<input_prompts>)

anno = tracking_SAM.plt_clicker.Annotator(img, sam2_predictor)

anno.main()

mask_np_hw = anno.get_mask()

plt.imshow(mask_np_hw)
plt.show()

