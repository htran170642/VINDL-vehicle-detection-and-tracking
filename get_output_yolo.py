
import core.utils as utils
from core.yolov4 import YOLOv4, YOLOv3, YOLOv3_tiny, decode
from PIL import Image
from core.config import cfg
import cv2
import numpy as np
import tensorflow as tf


XYSCALE = cfg.YOLO.XYSCALE
STRIDES = np.array(cfg.YOLO.STRIDES)
ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS, False)

def load_model_yolov4(input_size = 608, weights = './data/yolov4_original_last.weights'):

    NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
    input_layer = tf.keras.layers.Input([input_size, input_size, 3])
    feature_maps = YOLOv4(input_layer, NUM_CLASS)
    bbox_tensors = []
    for i, fm in enumerate(feature_maps):
        bbox_tensor = decode(fm, NUM_CLASS, i)
        bbox_tensors.append(bbox_tensor)
    model = tf.keras.Model(input_layer, bbox_tensors)
    utils.load_weights(model, weights)

    model.summary()
    return model


def get_boxes(model, original_image, input_size = 608):

    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_size = original_image.shape[:2]

    image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    pred_bbox = model.predict(image_data)
    pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE)

    bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.25)
    if bboxes is None:
        return [], [], []
    bboxes = utils.nms(bboxes, 0.213, method='nms')
    boxs = list(np.array(bboxes)[:, 0:4])
    confidence = list(np.array(bboxes)[:, 4])
    class_idx = list(np.array(bboxes)[:, 5])

    # image = utils.draw_bbox(original_image, bboxes)
    # image = Image.fromarray(image)
    return boxs, confidence, class_idx