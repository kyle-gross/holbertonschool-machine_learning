#!/usr/bin/env python3
"""Contains the class FaceAlign"""

import numpy as np
import dlib


class FaceAlign:
    """The FaceAlign class"""
    def __init__(self, shape_predictor_path):
        """Instantiate the FaceAlign class.

        Sets the public instance attributes:
            detector: contains dlib's default face detector
            shape_predictor: contains the dlib.shape_predictor
        """
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(shape_predictor_path)
