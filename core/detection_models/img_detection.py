import torch
import mmdet.apis as apis
import core.config as config

def load_img_detection_model():
    """ Loads the detection model """
    if torch.cuda.is_available():
        print("gpu_mode")
        model = apis.init_detector(config.IMG_MODEL_CONFIG, config.IMG_MODEL_GPU, device='cuda:0')
    else:
        print("cpu_mode")
        model = apis.init_detector(config.IMG_MODEL_CONFIG, config.IMG_MODEL_GPU, device='cpu', cfg_options={"map_location":'cpu'})
    return model