#!/usr/bin/env python3
"""Contains the class FaceAlign"""

import dlib
import numpy as np


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

    def detect(self, image):
        """Detects a face in an image

        Args:
            image (rank 3 ndarray): contains the image to detect in

        Returns:
            dlib.rectangle containing the boundary box for the face
            or None if failure
        """
        faces = self.detector(image, 1)

        if len(faces) == 1:
            return max(faces, key=lambda rect: rect.width() * rect.height())
        else:
            return None
