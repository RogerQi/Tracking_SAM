import os
import numpy as np
from PIL import Image
import torch
import tracking_SAM.aott
import tracking_SAM.plt_clicker
from segment_anything import sam_model_registry, SamPredictor

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import predict

class main_tracker:
    def __init__(self, sam_checkpoint, aot_checkpoint, ground_dino_checkpoint,
                 sam_model_type="vit_h", device="cuda"):
        self.device = device
        self.tracking = False

        self.sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)
        self.sam_predictor = SamPredictor(self.sam)

        # Custom wrapper for AOTT
        self.vos_tracker = tracking_SAM.aott.aot_segmenter(aot_checkpoint)

        self.reset_engine()

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        config_file_path = os.path.join(cur_dir,
                                        'third_party',
                                        'GroundingDINO',
                                        'groundingdino',
                                        'config',
                                        'GroundingDINO_SwinT_OGC.py')

        args = SLConfig.fromfile(config_file_path)

        self.dino_model = build_model(args)

        checkpoint = torch.load(ground_dino_checkpoint, map_location='cpu')
        self.dino_model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        self.dino_model.eval()
        self.dino_model = self.dino_model.to(device)
    
    def annotate_init_frame(self, img, method='clicking', category_name='background'):
        """
        Annotate the first frame of the video.

        Args:
            img: numpy array of shape (H, W, 3) and dtype uint8. in RGB format.
            method: 'clicking' or 'dino'. 'clicking' is the default method.
        """
        if method == 'clicking':
            anno = tracking_SAM.plt_clicker.Annotator(img, self.sam_predictor)
            anno.main()  # blocking call
            mask_np_hw = anno.get_mask()
        elif method == 'dino':
            assert category_name != 'background', "Category name must be specified!"
            transform = T.Compose(
                [
                    T.RandomResize([800], max_size=1333),  # not acutally random. It selects from [800].
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

            img_chw, _ = transform(Image.fromarray(img), None)

            # From official groundingdino demo
            BOX_TRESHOLD = 0.3
            TEXT_TRESHOLD = 0.25

            boxes, logits, phrases = predict(
                model=self.dino_model, 
                image=img_chw, 
                caption=category_name, 
                box_threshold=BOX_TRESHOLD, 
                text_threshold=TEXT_TRESHOLD,
                device=self.device
            )

            H, W, _ = img.shape

            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

            self.sam_predictor.set_image(img)
            assert len(boxes_xyxy) == 1
            input_box = boxes_xyxy[0].cpu().numpy()

            masks, _, _ = self.sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )

            mask_np_hw = masks[0].astype(np.uint8)
        else:
            raise NotImplementedError(f"method {method} not implemented!")

        mask_np_hw = mask_np_hw.astype(np.uint8)
        mask_np_hw[mask_np_hw > 0] = 1  # TODO(roger): support multiple objects?

        self.vos_tracker.add_reference_frame(img, mask_np_hw)

        self.tracking = True

    def propagate_one_frame(self, img):
        assert self.tracking, "Please call annotate_init_frame() first!"
        pred_np_hw = self.vos_tracker.propagate_one_frame(img)
        return pred_np_hw
    
    def reset_engine(self):
        self.vos_tracker.reset_engine()
        self.tracking = False
    
    def is_tracking(self):
        return self.tracking
