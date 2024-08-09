import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from PIL import Image
from tqdm import trange
from segment_anything import sam_model_registry, SamPredictor

def time_sam_clicking_time(test_img_path, predictor, num_trials=100):
    image_np = np.array(Image.open(test_img_path).convert("RGB"))

    input_x = np.random.randint(0, image_np.shape[1])
    input_y = np.random.randint(0, image_np.shape[0])

    input_point = np.array([[input_x, input_y]])  # (x, y); DIFFERENT FROM OPENCV COORDINATE
    input_label = np.ones((input_point.shape[0], ))

    sam_time_list = []
    for _ in trange(num_trials):
        start_cp = time.time()

        predictor.set_image(image_np)

        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )

        end_cp = time.time()
        sam_time_list.append(end_cp - start_cp)
    
    print("Using {} trials".format(num_trials))
    print(f"Average SAM clicking time: {np.mean(sam_time_list):.4f} sec")
    print(f"STD: {np.std(sam_time_list):.4f} sec")


def main():
    sam_checkpoint = "./pretrained_weights/sam_vit_h_4b8939.pth"  # default model
    test_img_paths = [
        "./sample_data/DAVIS_bear/images/00000.jpg"
    ]

    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    for test_img_path in test_img_paths:
        time_sam_clicking_time(test_img_path, predictor)

if __name__ == "__main__":
    main()