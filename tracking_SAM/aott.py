import importlib
import sys
import os

base_dir = os.path.join(os.path.dirname(__file__), 'third_party/aot_benchmark')

sys.path.insert(0, base_dir)

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from utils.checkpoint import load_network
from networks.models import build_vos_model
from networks.engines import build_engine

import dataloaders.video_transforms as tr

class aot_segmenter:
    def __init__(self, ckpt_path, gpu_id=0):
        cfg = self.get_config(gpu_id, ckpt_path)
        # Pad AOT CFG
        self.gpu_id = gpu_id
        # Load pre-trained model
        self.model = build_vos_model(cfg.MODEL_VOS, cfg).cuda(gpu_id)
        self.model, _ = load_network(self.model, cfg.TEST_CKPT_PATH, gpu_id)
        self.engine = build_engine(cfg.MODEL_ENGINE,
                              phase='eval',
                              aot_model=self.model,
                              gpu_id=gpu_id,
                              long_term_mem_gap=cfg.TEST_LONG_TERM_MEM_GAP)

        # Prepare datasets for each sequence
        self.transform = transforms.Compose([
            tr.MultiRestrictSize(cfg.TEST_MIN_SIZE, cfg.TEST_MAX_SIZE,
                                 cfg.TEST_FLIP, cfg.TEST_MULTISCALE,
                                 cfg.MODEL_ALIGN_CORNERS),
            tr.MultiToTensor()
        ])
        self.cfg = cfg
        
        self.reset_engine()
    
    @staticmethod
    def get_config(gpu_id, ckpt_path):
        exp_name = 'AOT Tool'
        stage = 'pre_ytb_dav'
        model = 'aott'
        gpu_id = gpu_id
        data_path = os.path.join(base_dir, './datasets/Demo')
        output_path = os.path.join(base_dir, './demo_output')
        max_resolution = 480*1.3

        engine_config = importlib.import_module('configs.' + stage)
        cfg = engine_config.EngineConfig(exp_name, model)

        cfg.TEST_GPU_ID = gpu_id

        cfg.TEST_CKPT_PATH = ckpt_path
        cfg.TEST_DATA_PATH = data_path
        cfg.TEST_OUTPUT_PATH = output_path

        cfg.TEST_MIN_SIZE = None
        cfg.TEST_MAX_SIZE = max_resolution * 800. / 480.
        
        return cfg
    
    def preprocess_sample(self, img, label=None):
        """
        Parameters
            - img: (H, W, 3) np.array (RGB ordering; 0-255)
            - label (optional): (H, W) np.array in np.int of same size as img
        
        Return
            - ret_dict: ret_dict ready to be fed into AOT engine
        """
        assert len(img.shape) == 3 and img.shape[2] == 3
        H, W, _ = img.shape
        
        if label is not None:
            obj_idx_list = np.unique(label)
            valid_obj_idx_list = [i for i in obj_idx_list if i != 0]
            assert len(valid_obj_idx_list) > 0, "Not valid label provided"
            novel_obj_idx_list = [i for i in valid_obj_idx_list if i not in self.tracked_obj_idx_list]
            self.obj_num += len(novel_obj_idx_list)
        
        meta_dict = {
            'obj_num': self.obj_num,
            'height': H,
            'width': W,
            'flip': False # no flipping
        }
        
        ret_dict = {
            'current_img': img,
            'meta': meta_dict
        }
        
        if label is not None:
            ret_dict['current_label'] = label
        
        ret_dict = self.transform(ret_dict)[0] # return a list of length 1
        
        ret_dict['current_img'] = ret_dict['current_img'].cuda(self.gpu_id,
                                                               non_blocking=True).float()
        ret_dict['current_img'] = ret_dict['current_img'].reshape((1,) + ret_dict['current_img'].shape)
        
        if 'current_label' in ret_dict:
            ret_dict['current_label'] = ret_dict['current_label'].cuda(self.gpu_id,
                                                                      non_blocking=True)
            ret_dict['current_label'] = ret_dict['current_label'].reshape((1,) + ret_dict['current_label'].shape)
            ret_dict['current_label'] = F.interpolate(ret_dict['current_label'].float(),
                                                     ret_dict['current_img'].shape[2:],
                                                     mode='nearest')
        
        return ret_dict
    
    def add_reference_frame(self, img, label):
        data_dict = self.preprocess_sample(img, label)
        with torch.no_grad():
            self.engine.add_reference_frame(data_dict['current_img'],
                                               data_dict['current_label'],
                                               frame_step=self.frame_cnt,
                                               obj_nums=self.obj_num)
        self.frame_cnt += 1
    
    def propagate_one_frame(self, img):
        data_dict = self.preprocess_sample(img)
        with torch.no_grad():
            # predict segmentation
            self.engine.match_propogate_one_frame(data_dict['current_img'])
            pred_logit = self.engine.decode_current_logits(
                (data_dict['meta']['height'], data_dict['meta']['width']))
            pred_prob = torch.softmax(pred_logit, dim=1)
            pred_label = torch.argmax(pred_prob, dim=1,
                                      keepdim=True).float()
            _pred_label = F.interpolate(pred_label,
                                        size=self.engine.input_size_2d,
                                        mode="nearest")
            # update memory
            self.engine.update_memory(_pred_label)
        self.frame_cnt += 1
        
        return pred_label.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)
    
    def reset_engine(self):
        self.tracked_obj_idx_list = []
        self.obj_num = 0
        self.frame_cnt = 0
        self.model.eval()
        self.engine.restart_engine()