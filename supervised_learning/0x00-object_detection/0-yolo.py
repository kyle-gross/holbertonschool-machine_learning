#!/usr/bin/env python3
"""Contains the class Yolo.
Yolo uses the Yolo v3 algorithm to perform object detection
"""

import tensorflow.keras as K
import numpy as np


class Yolo:
    """Performs object detection with the Yolo v3 algorithm"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Instantiate Yolo object

        Args:
            model_path (str): path to where Darknet Keras model is stored
            classes_path (str): path to list of class names used for Darknet
                model, listed in order of index
            class_t (float): represents box score threshold for initial filter
            nms_t (float): represents the IOU theshold for non-max suppression
            anchors (numpy.ndarray)(outputs,anchor_boxes,2): Contains anchor
                boxes.
                outputs: # of outputs (predictions) made by Darknet model
                anchor_boxes: # of anchor boxes used for each prediction
                2: [anchor_box_width, anchor_box_height]

        Public instance attributes:
            model: the Darknet Keras model
            class_names: list of the class names for the model
            class_t: the box score threshold for the initial filtering
            nms_t: the IOU threshold for non-max suppression
            anchors: the anchor boxes
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path) as f:
            names = f.read().split('\n')
        self.class_names = names
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
