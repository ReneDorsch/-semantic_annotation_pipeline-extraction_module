import numpy as np
from mmdet.apis import init_detector, inference_detector
from core import config as config


def predict_table_boundaries(model, img_path: str):
    """ Predicts the boundaries of a file. """
    result = inference_detector(model, img_path)
    return result


def get_bbox_class_tuple(result, img_path):
    """ Extracts the Coordinates and the class of the bbox. """
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    bbox_int = bboxes.astype(np.int32)

    x1 = bbox_int[:, 0]
    y1 = bbox_int[:, 1]
    x2 = bbox_int[:, 2]
    y2 = bbox_int[:, 3]
    scores = bboxes[:, 4]

    return zip(x1, y1, x2, y2, scores, labels)


def in_json(results, img, page):
    """ Saves the output of the prediction as a dict. """
    res = []

    for x1, y1, x2, y2, score, label in get_bbox_class_tuple(results, img):
        if score > 0.9:
            res.append({
                'x1': int(x1),
                'y1': int(y1),
                'x2': int(x2),
                'y2': int(y2),
                'page': page
            })
    return res