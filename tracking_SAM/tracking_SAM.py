import numpy as np
import tracking_SAM.aott
import tracking_SAM.plt_clicker
from segment_anything import sam_model_registry, SamPredictor

class main_tracker:
    def __init__(self, sam_checkpoint, aot_checkpoint,
                 sam_model_type="vit_h", device="cuda"):
        self.sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)
        self.sam_predictor = SamPredictor(self.sam)

        self.vos_tracker = tracking_SAM.aott.aot_segmenter(aot_checkpoint)

        self.reset_engine()
    
    def annotate_init_frame(self, img):
        anno = tracking_SAM.plt_clicker.Annotator(img, self.sam_predictor)
        anno.main()  # blocking call
        mask_np_hw = anno.get_mask()

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
