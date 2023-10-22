
import cv2
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

# Forbidden  Key: QSFKL


class Annotator(object):
    def __init__(self, img_np, sam_predictor, save_path=None):
        self.sam_predictor = sam_predictor
        self.save_path = save_path
        self.img = img_np.copy()
        self.sam_predictor.set_image(self.img)
        self.clicks = np.empty([0, 2], dtype=np.int64)
        self.pred = np.zeros(self.img.shape[:2], dtype=np.uint8)
        self.merge = self.__gene_merge(self.pred, self.img, self.clicks)

    def __gene_merge(self, pred, img, clicks, r=9, cb=2, b=2, if_first=True):
        pred_mask = cv2.merge([pred * 255, pred * 255, np.zeros_like(pred)])
        result = np.uint8(np.clip(img * 0.7 + pred_mask * 0.3, 0, 255))
        if b > 0:
            contours, _ = cv2.findContours(pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, contours, -1, (255, 255, 255), b)
        for pt in clicks:
            cv2.circle(result, tuple(pt), r, (255, 0, 0), -1)
            cv2.circle(result, tuple(pt), r, (255, 255, 255), cb)
        if if_first and len(clicks) != 0:
            cv2.circle(result, tuple(clicks[0, :2]), r, (0, 255, 0), cb)
        return result

    def __update(self):
        self.ax1.imshow(self.merge)
        self.fig.canvas.draw()

    def __reset(self):
        self.clicks = np.empty([0, 2], dtype=np.int64)
        self.pred = np.zeros(self.img.shape[:2], dtype=np.uint8)
        self.merge = self.__gene_merge(self.pred, self.img, self.clicks)
        self.__update()

    def __predict(self):
        # TODO(roger): support multiple instances and negative clicks
        input_label = np.ones((self.clicks.shape[0], ))
        masks, scores, logits = self.sam_predictor.predict(
            point_coords=self.clicks,
            point_labels=input_label,
            multimask_output=False,
        )
        self.pred = masks[0].astype(np.uint8)
        self.merge = self.__gene_merge(self.pred, self.img, self.clicks)
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
            return
        if event.button == 1:  # 1 for left click; 3 for right click
            x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
            self.clicks = np.append(self.clicks, np.array(
                [[x, y]], dtype=np.int64), axis=0)
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

if __name__ == "__main__":
    from segment_anything import sam_model_registry, SamPredictor
    sam_checkpoint = "../pretrained_weights/sam_vit_h_4b8939.pth"  # default model

    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    img_path = "../sample_data/DAVIS_bear/images/00000.jpg"
    img_np = np.array(Image.open(img_path))
    anno = Annotator(img_np, predictor, save_path="/tmp/00000.png")
    anno.main()

    print("Done!")
    print(anno.get_mask().shape)
