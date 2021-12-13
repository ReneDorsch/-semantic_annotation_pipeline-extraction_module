import os

CURRENT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
TESTFILE = os.path.join(CURRENT_DIRECTORY, 'extraction_modul/testFiles/j.jmrt.2019.12.072.pdf')
TMP_DIRECTORY = os.path.join(CURRENT_DIRECTORY, 'files/tmp')
INPUT_DIRECTORY = os.path.join(TMP_DIRECTORY, 'input/')
IMAGE_DIRECTORY = os.path.join(TMP_DIRECTORY, 'imgs/')
TABLE_DIRECTORY = os.path.join(TMP_DIRECTORY, 'tables/')


METADATA_PATTERNS = os.path.join(CURRENT_DIRECTORY, 'files/meta_data_pattern.json')


TABLE_MODEL_GPU = os.path.join(CURRENT_DIRECTORY, 'detection_models/models/table_model/gpu_table_detection_model.pth')
TABLE_MODEL_CPU = os.path.join(CURRENT_DIRECTORY, 'detection_models/models/table_model/cpu_table_detection_model.pth')
TABLE_MODEL_CONFIG = os.path.join(CURRENT_DIRECTORY,
                                  'detection_models/models/table_model/cascade_mask_rcnn_hrnetv2p_w32_20e_v2.py')

IMG_MODEL_GPU = os.path.join(CURRENT_DIRECTORY, 'detection_models/models/image_model/img_detection_model.pth')
IMG_MODEL_CPU = os.path.join(CURRENT_DIRECTORY, 'detection_models/models/image_model/img_detection_model.pth')
IMG_MODEL_CONFIG = os.path.join(CURRENT_DIRECTORY, 'detection_models/models/image_model/config_model.py')

TABLE_MODEL_CATEGORIES = {
              0: 'Bordered_Table',
              1: 'Cell',
              2: 'Borderless_Table'
            }

IMG_MODEL_CATEGORIES = {
                0: 'Image'
            }
