import os
import numpy as np
from PIL import Image
import time
import torch
from tqdm import tqdm, trange

# Ours
import tracking_SAM.aott

@torch.no_grad()
def time_video_seq(my_vos_tracker, test_data_base_dir, num_trials=100):
    # Call this to clear VOS internal states
    my_vos_tracker.reset_engine()
    # Load images
    image_paths_list = sorted([os.path.join(test_data_base_dir, 'images', x) for x in os.listdir(os.path.join(test_data_base_dir, 'images'))])

    image_np_list = [np.array(Image.open(x)) for x in image_paths_list]

    # Load the initial provided mask
    init_mask_path = os.path.join(test_data_base_dir, 'init_mask', '00000.png')
    init_mask = np.array(Image.open(init_mask_path)).astype(np.uint8)

    init_mask[init_mask > 0] = 1

    all_mask_list = [init_mask]  # The first mask is the initial mask

    reference_frame_time = []
    for _ in trange(num_trials):
        start_cp = time.time()
        my_vos_tracker.add_reference_frame(image_np_list[0], init_mask)
        end_cp = time.time()
        reference_frame_time.append(end_cp - start_cp)

    print("Using {} trials".format(num_trials))
    print('Reference frame time: {:.4f}. Std: {:.4f}'.format(np.mean(reference_frame_time), np.std(reference_frame_time)))

    # Reset reference frames; we will add them again
    my_vos_tracker.reset_engine()
    my_vos_tracker.add_reference_frame(image_np_list[0], init_mask)

    propage_frame_time = []
    for i in trange(1, len(image_np_list)):
        start_cp = time.time()
        cur_frame_np = image_np_list[i]
        cur_mask_np = my_vos_tracker.propagate_one_frame(cur_frame_np)
        all_mask_list.append(cur_mask_np)
        end_cp = time.time()
        propage_frame_time.append(end_cp - start_cp)
    
    print('Propagate frame time: {:.4f}. Std: {:.4f}'.format(np.mean(propage_frame_time), np.std(propage_frame_time)))

def main():
    vos_weight_path = './pretrained_weights/AOTT_PRE_YTB_DAV.pth'

    my_vos_tracker = tracking_SAM.aott.aot_segmenter(vos_weight_path)

    test_folder_list = ['./sample_data/DAVIS_bear']

    for folder in test_folder_list:
        time_video_seq(my_vos_tracker, folder)

if __name__ == '__main__':
    main()
