import os
import numpy as np
from PIL import Image
import cv2
import time
import argparse

import tracking_SAM

def main(sam_checkpoint, aot_checkpoint, grounding_dino_checkpoint, play_delay):

    test_data_base_dir = './sample_data/DAVIS_bear'

    # Load images
    image_paths_list = sorted([os.path.join(test_data_base_dir, 'images', x) for x in os.listdir(os.path.join(test_data_base_dir, 'images'))])

    image_np_list = [np.array(Image.open(x)) for x in image_paths_list]

    my_tracking_SAM = tracking_SAM.main_tracker(sam_checkpoint, aot_checkpoint, grounding_dino_checkpoint)

    for i in range(len(image_np_list)):
        image_np_rgb = image_np_list[i]
        image_np_bgr = cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2BGR)

        cv2.imshow('Video', image_np_bgr)

        if my_tracking_SAM.is_tracking():
            start_cp = time.time()
            pred_np_hw = my_tracking_SAM.propagate_one_frame(image_np_rgb)
            time_elapsed = time.time() - start_cp
            pred_np_hw = pred_np_hw.astype(np.uint8)
            pred_np_hw[pred_np_hw > 0] = 255

            viz_img = image_np_bgr.copy()
            # Alpha blending to add red mask
            red_overlay = np.dstack((np.zeros_like(pred_np_hw), np.zeros_like(pred_np_hw), pred_np_hw))
            viz_img = cv2.addWeighted(viz_img, 0.5, red_overlay, 0.5, 0)
            # Show time_elapsed on the screen
            str_to_show = f'VOS Latency {time_elapsed:.2f} s'
            cv2.putText(viz_img, str_to_show, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('Tracked', viz_img)

        # Press Q on keyboard to exit
        key_pressed = cv2.waitKey(play_delay) & 0xFF
        if key_pressed == ord('q'):
            break
        elif key_pressed == ord('a'):
            if my_tracking_SAM.is_tracking():
                my_tracking_SAM.reset_engine()
            my_tracking_SAM.annotate_init_frame(image_np_rgb)
        elif key_pressed == ord('d'):
            if my_tracking_SAM.is_tracking():
                my_tracking_SAM.reset_engine()
            my_tracking_SAM.annotate_init_frame(image_np_rgb, method='dino', category_name='bear')
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--sam_checkpoint', type=str, default="./pretrained_weights/sam_vit_h_4b8939.pth")
    parser.add_argument('--aot_checkpoint', type=str, default="./pretrained_weights/AOTT_PRE_YTB_DAV.pth")
    parser.add_argument('--ground_dino_checkpoint', type=str, default="./pretrained_weights/groundingdino_swint_ogc.pth")

    # delay in ms for each image to stay on screen. Low values (e.g., 1) causes the video to pass by quickly.
    parser.add_argument('--play_delay', type=int, default=200)

    args = parser.parse_args()

    main(args.sam_checkpoint, args.aot_checkpoint, args.ground_dino_checkpoint, args.play_delay)
