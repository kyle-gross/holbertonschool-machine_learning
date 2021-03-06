#!/usr/bin/env python3
"""Contains the class FaceAlign"""

import cv2
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

    def find_landmarks(self, image, detection):
        """Finds facial landmarks

        Args:
            image (ndarray): image of which to find landmarks
            detection (dlib.rectangle): contains boundary box of face

        Returns:
            numpy.ndarray (shape(p, 2)) containing landmark points
                * p: no. landmark points
                * 2: x and y coordinates of point
            None if failure
        """
        points = self.shape_predictor(image, detection)

        return np.array([(float(p.x), float(p.y)) for p in points.parts()])

    def align(self, image, landmark_indices, anchor_points, size=96):
        """Aligns an image for face verification

        Args:
            image (rank 3 ndarray): image to align
            landmark_indices (ndarray)(3,): contains indices of the 3 landmark
                points that should be used for the affine tranformation
            anchor_points (ndarray)(3,2): destination points for the affine
                transformation, scaled to range [0, 1]
            size (int): desired size of the aligned image

        Returns:
            ndarray (size, size, 3): contains aligned image, or None if failure
        """
        bb = self.detect(image)
        if not bb:
            return None

        landmarks = self.find_landmarks(image, bb)
        np_landmarks = np.float32(landmarks)
        points = anchor_points * size

        H = cv2.getAffineTransform((np_landmarks[landmark_indices]), points)
        thumbnail = cv2.warpAffine(image, H, (size, size))

        return thumbnail
