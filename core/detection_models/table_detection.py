import torch
from mmdet.apis import init_detector

import core.config as config


def load_table_detection_model():
    """ Loads the detection model """
    if torch.cuda.is_available():
        print("gpu_mode")
        model = init_detector(config.TABLE_MODEL_CONFIG, config.TABLE_MODEL_GPU, device='cuda:0')
    else:
        print("cpu_mode")
        model = init_detector(config.TABLE_MODEL_CONFIG, config.TABLE_MODEL_GPU, device='cpu', cfg_options={"map_location":'cpu'})
    return model


