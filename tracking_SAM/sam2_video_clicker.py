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
import imgviz

COLORMAP = imgviz.label_colormap()

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
        mask_viz = np.zeros(self.img.shape, dtype=np.uint8)
        for obj_idx in self.click_label_dict:
            mask_viz[self.pred == obj_idx] = COLORMAP[obj_idx % len(COLORMAP)]
        result = np.uint8(np.clip(self.img * 0.7 + mask_viz * 0.3, 0, 255))
        # if b > 0:
        #     contours, _ = cv2.findContours(self.pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #     cv2.drawContours(result, contours, -1, (255, 255, 255), b)
        for obj_idx in self.click_label_dict:
            for idx, pt in enumerate(self.click_label_dict[obj_idx]["clicks"]):
                if self.click_label_dict[obj_idx]["labels"][idx] == 1:
                    INNER_COLOR = (0, 255, 0)
                else:
                    INNER_COLOR = (255, 0, 0)
                cv2.circle(result, tuple(pt), r, INNER_COLOR, -1)
                # bright outer ring
                cv2.circle(result, tuple(pt), r, (255, 255, 255), cb)
        return result

    def __reset_annotator_state(self):
        self.click_label_dict = {}
        self.cur_obj_idx = 0
        self.__add_new_obj()
        self.pred = np.zeros(self.img.shape[:2], dtype=np.uint8)
        self.merge = self.__gene_merge()
    
    def __add_new_obj(self):
        self.cur_obj_idx += 1
        self.click_label_dict[self.cur_obj_idx] = {
            'clicks': np.empty([0, 2], dtype=np.int64),
            'labels': np.empty([0], dtype=np.int64),
        }

    def __update(self):
        self.ax1.imshow(self.merge)
        self.fig.canvas.draw()

    def __reset(self):
        self.__reset_annotator_state()
        self.__update()

    def __predict(self):
        # Add predictions to the initial frame
        _, out_obj_ids, out_mask_logits = self.sam_predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=0,  # the frame index we interact with
            obj_id=self.cur_obj_idx, # give a unique id to each object we interact with (it can be any integers)
            points=self.click_label_dict[self.cur_obj_idx]["clicks"],
            labels=self.click_label_dict[self.cur_obj_idx]["labels"],
        )
        self.pred = np.zeros_like(self.pred)
        for obj_idx in self.click_label_dict:
            cur_mask_hw = (out_mask_logits[obj_idx - 1, 0] > 0).cpu().numpy()
            self.pred[cur_mask_hw] = obj_idx
            
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
        elif event.key == ' ': # space
            self.__add_new_obj()
            self.fig.suptitle('[RESET]: ctrl+r; [REVOKE]: ctrl+z; [EXIT]: esc; [DONE]: enter; objIDX: {}'.format(self.cur_obj_idx), fontsize=14)
            self.fig.canvas.draw()
        elif event.key == 'enter':
            if self.save_path is not None:
                Image.fromarray(self.pred * 255).save(self.save_path)
                print('save mask in [{}]!'.format(self.save_path))
            plt.close()

    def __on_button_press(self, event):
        if (event.xdata is None) or (event.ydata is None):
            # button press outside of image gives not xdata/ydata
            pass
        if event.button == 1:  # 1 for left click; 3 for right click
            x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
            self.click_label_dict[self.cur_obj_idx]["clicks"] = np.append(self.click_label_dict[self.cur_obj_idx]["clicks"], np.array(
                [[x, y]], dtype=np.int64), axis=0)
            self.click_label_dict[self.cur_obj_idx]["labels"] = np.append(self.click_label_dict[self.cur_obj_idx]["labels"], 1)
            self.__predict()
        elif event.button == 3:
            x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
            self.click_label_dict[self.cur_obj_idx]["clicks"] = np.append(self.click_label_dict[self.cur_obj_idx]["clicks"], np.array(
                [[x, y]], dtype=np.int64), axis=0)
            self.click_label_dict[self.cur_obj_idx]["labels"] = np.append(self.click_label_dict[self.cur_obj_idx]["labels"], 0)
            self.__predict()

    def main(self):
        self.fig = plt.figure('Annotator', figsize=(10, 7))
        self.fig.canvas.mpl_connect('key_press_event', self.__on_key_press)
        self.fig.canvas.mpl_connect("button_press_event", self.__on_button_press)
        self.fig.suptitle('[RESET]: ctrl+r; [REVOKE]: ctrl+z; [EXIT]: esc; [DONE]: enter; objIDX: {}'.format(self.cur_obj_idx), fontsize=14)
        self.ax1 = self.fig.add_subplot(1, 1, 1)
        self.ax1.axis('off')
        self.ax1.imshow(self.merge)
        plt.show()
