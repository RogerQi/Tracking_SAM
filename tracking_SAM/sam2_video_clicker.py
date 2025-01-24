import os
import cv2
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import sam2
import tempfile

class Annotator(object):
    def __init__(self, video_folder_path, sam_predictor, save_path=None):
        self.sam_predictor = sam_predictor
        self.save_path = save_path
        self.sam_predictor = sam_predictor
        self.video_folder_path = video_folder_path
        # Initialize
        self.inference_state = self.sam_predictor.init_state(video_path=video_folder_path)

        self.sam_predictor.reset_state(self.inference_state)

        first_img_path = os.path.join(video_folder_path, sorted(os.listdir(video_folder_path))[0])
        img_np = np.array(Image.open(first_img_path).convert('RGB'))
        self.img = img_np.copy()

        self.__reset_annotator_state()

    def __gene_merge(self, r=9, cb=2, b=2, if_first=True):
        # Visualize points and countours of masks
        pred_mask = cv2.merge([self.pred * 255, self.pred * 255, np.zeros_like(self.pred)])
        result = np.uint8(np.clip(self.img * 0.7 + pred_mask * 0.3, 0, 255))
        if b > 0:
            contours, _ = cv2.findContours(self.pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, contours, -1, (255, 255, 255), b)
        for idx, pt in enumerate(self.clicks):
            if self.labels[idx] == 1:
                INNER_COLOR = (0, 255, 0)
            else:
                INNER_COLOR = (255, 0, 0)
            cv2.circle(result, tuple(pt), r, INNER_COLOR, -1)
            # bright outer ring
            cv2.circle(result, tuple(pt), r, (255, 255, 255), cb)
        return result

    def __reset_annotator_state(self):
        self.clicks = np.empty([0, 2], dtype=np.int64)
        self.labels = np.empty([0], dtype=np.int64)
        self.pred = np.zeros(self.img.shape[:2], dtype=np.uint8)
        self.merge = self.__gene_merge()

    def __update(self):
        self.ax1.imshow(self.merge)
        self.fig.canvas.draw()

    def __reset(self):
        self.__reset_annotator_state()
        self.__update()

    def __predict(self):
        # TODO(roger): support multiple instances
        ann_obj_id = 1
        _, out_obj_ids, out_mask_logits = self.sam_predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=0,  # the frame index we interact with
            obj_id=ann_obj_id, # give a unique id to each object we interact with (it can be any integers)
            points=self.clicks,
            labels=self.labels,
        )
        self.pred = (out_mask_logits[0, 0] > 0).cpu().numpy().astype(np.uint8)
        self.merge = self.__gene_merge()
        self.__update()

    def __on_key_press(self, event):
        if event.key == 'ctrl+z':
            self.clicks = self.clicks[:-1, :]
            if len(self.clicks) != 0:
                self.__predict()
            else:
                self.__reset()
        elif event.key == 'ctrl+r':
            self.__reset()
        elif event.key == 'escape':
            plt.close()
        elif event.key == 'enter':
            if self.save_path is not None:
                Image.fromarray(self.pred * 255).save(self.save_path)
                print('save mask in [{}]!'.format(self.save_path))
            plt.close()

    def __on_button_press(self, event):
        if (event.xdata is None) or (event.ydata is None):
            # button press is for mouse. So it should have x/y coordinates
            raise ValueError("Invalid button press event")
        if event.button == 1:  # 1 for left click; 3 for right click
            x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
            self.clicks = np.append(self.clicks, np.array(
                [[x, y]], dtype=np.int64), axis=0)
            self.labels = np.append(self.labels, 1)
            self.__predict()
        elif event.button == 3:
            x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
            self.clicks = np.append(self.clicks, np.array(
                [[x, y]], dtype=np.int64), axis=0)
            self.labels = np.append(self.labels, 0)
            self.__predict()
        

    def main(self):
        self.fig = plt.figure('Annotator', figsize=(10, 7))
        self.fig.canvas.mpl_connect('key_press_event', self.__on_key_press)
        self.fig.canvas.mpl_connect("button_press_event", self.__on_button_press)
        self.fig.suptitle('[RESET]: ctrl+r; [REVOKE]: ctrl+z; [EXIT]: esc; [DONE]: enter', fontsize=14)
        self.ax1 = self.fig.add_subplot(1, 1, 1)
        self.ax1.axis('off')
        self.ax1.imshow(self.merge)
        plt.show()
    
    def get_mask(self):
        return self.pred
